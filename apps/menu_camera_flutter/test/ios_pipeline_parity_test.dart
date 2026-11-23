import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:menu_camera_flutter/models/contracts.dart';
import 'package:menu_camera_flutter/pipeline/canonical_ctc_decoder.dart';
import 'package:menu_camera_flutter/pipeline/lexicon_lookup.dart';
import 'package:menu_camera_flutter/services/ios_coreml_backend.dart';

class _MockIosCoreMLBridge implements IosCoreMLNativeBridge {
  int detectorCalls = 0;
  int recognizerCalls = 0;

  @override
  Future<Map<String, dynamic>> initialize() async => {
        'status': 'ready',
        'backend_id': 'ios_coreml_native',
        'device_name': 'Mock iPhone 15',
        'bundle_id': 'stage4_step1000_raw_2105_v5_lmv2_lexv3_20261123',
        'capabilities': {
          'avfoundation_stream_camera': true,
          'coreml_split_recognizer': true,
          'coreml_obb_1024_detector': true,
          'readonly_sqlite_store': true,
          'compiled_models_present': true,
        },
      };

  @override
  Future<Map<String, dynamic>> runDetectorOBB1024(Uint8List imageBytes) async {
    detectorCalls++;
    // 2 anchors: anchor 0 = dish_horizontal (conf 0.92), anchor 1 = duplicate overlapping (conf 0.60)
    const n = 2;
    final raw = Float32List(7 * n);
    // cx, cy, w, h, p0, p1, angle_rad
    raw[0 * n + 0] = 512.0;
    raw[1 * n + 0] = 300.0;
    raw[2 * n + 0] = 400.0;
    raw[3 * n + 0] = 60.0;
    raw[4 * n + 0] = 0.92;
    raw[5 * n + 0] = 0.05;
    raw[6 * n + 0] = 0.0;

    raw[0 * n + 1] = 514.0;
    raw[1 * n + 1] = 301.0;
    raw[2 * n + 1] = 398.0;
    raw[3 * n + 1] = 60.0;
    raw[4 * n + 1] = 0.60;
    raw[5 * n + 1] = 0.02;
    raw[6 * n + 1] = 0.0;

    return {
      'raw_obb_f32': raw.buffer.asUint8List(),
      'num_anchors': n,
      'letterbox': {'gain': 1.0, 'pad_x': 0.0, 'pad_y': 0.0, 'src_w': 1024.0, 'src_h': 1024.0},
    };
  }

  @override
  Future<Map<String, dynamic>> runRecognizerCrop({
    required Uint8List imageBytes,
    required List<List<double>> polygon,
    required int orientationFlag,
  }) async {
    recognizerCalls++;
    // Emit crisp logits for "季節の先付と前菜" (8 chars, 16 frames interleaved with blank=0)
    // Vocab IDs in test: 1=季, 2=節, 3=の, 4=先, 5=付, 6=と, 7=前, 8=菜, 9=(, 10=（, 11=), 12=）
    const classes = 13;
    const seq = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0];
    final logits = Float32List(seq.length * classes);
    for (var t = 0; t < seq.length; t++) {
      for (var c = 0; c < classes; c++) {
        logits[t * classes + c] = (c == seq[t]) ? 8.0 : -4.0;
      }
    }
    return {
      'multi_window': false,
      'window_count': 1,
      'window_ranges': const [],
      'window_paths': const [],
      'valid_T': seq.length,
      'logits_f32': logits.buffer.asUint8List(),
      'crop_preview_png': Uint8List(0),
    };
  }

  @override
  Future<Map<String, TranslationRecord>> lookupSqlTerms(Set<String> terms) async {
    const db = {
      '季節': TranslationRecord(canonicalJp: '季節', nameEn: 'seasonal'),
      '先付': TranslationRecord(canonicalJp: '先付', nameEn: 'seasonal course starter (sakizuke)'),
      '前菜': TranslationRecord(canonicalJp: '前菜', nameEn: 'appetizers'),
    };
    return {for (final t in terms) if (db.containsKey(t)) t: db[t]!};
  }

  @override
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase) async {
    if (canonicalPhrase == '季節の先付と前菜') {
      return const PhrasePlanRecord(
        canonicalPhrase: '季節の先付と前菜',
        phraseEn: 'Seasonal Starter (Sakizuke) and Appetizers',
        provenanceSource: 'gemini_argon_v1',
      );
    }
    return null;
  }

  @override
  Future<Uint8List> loadBundledAssetBytes(String filename) async =>
      Uint8List.fromList(utf8.encode('{}'));
}

void main() {
  group('Canonical CTC + CharLM V2 R1 + Long-line overlap merger', () {
    test('Parenthesis logit pooling merges fullwidth into ASCII before CTC collapse', () {
      const c2i = {'（': 1, '(': 2, '）': 3, ')': 4, '税': 5};
      const i2c = {1: '（', 2: '(', 3: '）', 4: ')', 5: '税'};
      // Frame 0 has split mass on （ and ( that together beats blank=0
      final raw = [
        [0.4, 0.2, 0.2, -5.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, -5.0, -5.0, 5.0],
        [0.4, -5.0, -5.0, 0.2, 0.2, -5.0],
      ];
      final logp = foldParenthesisAndLogSoftmax(raw, c2i);
      expect(logp[0][1], double.negativeInfinity);
      expect(logp[2][3], double.negativeInfinity);
      expect(canonicalizeOcrText(ctcGreedyDecode(logp, i2c)), '(税)');
    });

    test('CharNGramLMV2 supports dig_unk_v2 and 326 expanded-char Unicode fallback', () {
      final lm = CharNGramLMV2(
        schemaVersion: 2,
        contextTransform: 'dig_unk_v2',
        vocabChars: const ['和', '牛', '円', '0', '1'],
        uni: const {'和': 10, '牛': 10, '円': 20},
        bi: const {'和': {'牛': 10}, '<DIG>': {'円': 15}},
        tri: const {},
        quad: const {},
      );
      // Digits get uniform log(0.1)
      expect(lm.stepLogp('<BOS>', '<BOS>', '<BOS>', '5'), closeTo(lm.uniformDigitLp, 1e-9));
      // Expanded char not in LM vocab uses finite defaultUni fallback (no NaN / crash)
      final lpExp = lm.stepLogp('<BOS>', '<BOS>', '和', '𩸽');
      expect(lpExp.isFinite, isTrue);
      // Digit context maps to <DIG> -> 円 gets boosted
      final lpYenAfterDig = lm.stepLogp('<BOS>', '<BOS>', '8', '円');
      final lpYenAfterBos = lm.stepLogp('<BOS>', '<BOS>', '<BOS>', '円');
      expect(lpYenAfterDig, greaterThan(lpYenAfterBos));
    });

    test('R1 routing uses greedy for <=2 Unicode chars and beam4g_dn for longer lines', () {
      const i2c = {1: '鯛', 2: '刺', 3: '身'};
      final lm = CharNGramLMV2(
        schemaVersion: 2,
        contextTransform: 'dig_unk_v2',
        vocabChars: const ['鯛', '刺', '身'],
        uni: const {'鯛': 5, '刺': 5, '身': 5},
        bi: const {},
        tri: const {},
        quad: const {},
      );
      final shortLogp = [[-5.0, 0.0, -5.0, -5.0]];
      final resShort = decodeWithR1Routing(
        logpTC: shortLogp,
        idToChar: i2c,
        mode: 'charlm_r1',
        alphaLm: 0.30,
        lm: lm,
      );
      expect(resShort.routeUsed, 'r1_short_greedy');
      expect(resShort.text, '鯛');

      final longLogp = [
        [-5.0, 0.0, -5.0, -5.0],
        [0.0, -5.0, -5.0, -5.0],
        [-5.0, -5.0, 0.0, -5.0],
        [0.0, -5.0, -5.0, -5.0],
        [-5.0, -5.0, -5.0, 0.0],
      ];
      final resLong = decodeWithR1Routing(
        logpTC: longLogp,
        idToChar: i2c,
        mode: 'charlm_r1',
        alphaLm: 0.30,
        lm: lm,
      );
      expect(resLong.routeUsed, 'beam4g_dn');
      expect(resLong.text, '鯛刺身');
    });

    test('computeWindowRanges and mergeOwnedWindowTokens match >512px overlap-100', () {
      final ranges = computeWindowRanges(640, maxWidth: 512, overlap: 100);
      expect(ranges, [[0, 512], [128, 640]]);
      final p0 = List<int>.filled(128, 0)..[10] = 1..[100] = 2;
      final p1 = List<int>.filled(128, 0)..[68] = 2..[110] = 3;
      final merged = mergeOwnedWindowTokens([p0, p1], ranges);
      expect(merged, [1, 2, 3]);
    });
  });

  group('Lexicon V3 segmentation + Read-Only SQLite translation parity', () {
    test('Segments 季節の先付と前菜 cleanly while protecting compounds and prices', () async {
      final idx = LexiconV3Index.fromTerms([
        '季節', '先付', '前菜', '胡麻豆腐', '胡麻', '豆腐', '親子丼', '和牛ロース', '未翻訳料理',
      ]);
      final store = InMemoryTranslationStore(
        terms: const {
          '季節': TranslationRecord(canonicalJp: '季節', nameEn: 'seasonal'),
          '先付': TranslationRecord(canonicalJp: '先付', nameEn: 'seasonal course starter (sakizuke)'),
          '前菜': TranslationRecord(canonicalJp: '前菜', nameEn: 'appetizers'),
          '胡麻豆腐': TranslationRecord(canonicalJp: '胡麻豆腐', nameEn: 'sesame tofu'),
          '和牛ロース': TranslationRecord(canonicalJp: '和牛ロース', nameEn: 'wagyu loin'),
          '未翻訳料理': TranslationRecord(canonicalJp: '未翻訳料理', nameEn: ''),
        },
        phrasePlans: const {
          '季節の先付と前菜': PhrasePlanRecord(
            canonicalPhrase: '季節の先付と前菜',
            phraseEn: 'Seasonal Starter (Sakizuke) and Appetizers',
          ),
        },
      );
      final pipe = LexiconV3Pipeline(idx, store);

      final tr1 = await pipe.lookupAndFormat('季節の先付と前菜');
      expect(tr1.components.map((c) => '${c.text}:${c.kind}').toList(),
          ['季節:exact', 'の:particle', '先付:exact', 'と:particle', '前菜:exact']);
      expect(tr1.primaryEnglish, 'Seasonal Starter (Sakizuke) and Appetizers');

      final trComp = await pipe.lookupAndFormat('胡麻豆腐');
      expect(trComp.components.length, 1);
      expect(trComp.components.first.text, '胡麻豆腐');

      final trPrice = await pipe.lookupAndFormat('和牛ロース 1800円');
      expect(trPrice.components.map((c) => '${c.text}:${c.kind}').toList(),
          ['和牛ロース:exact', '1800円:number_or_price']);

      final trAwait = await pipe.lookupAndFormat('未翻訳料理');
      expect(trAwait.translationStatus, 'await_for_translation');
      expect(trAwait.primaryEnglish, 'Await for translation');

      final trUnk = await pipe.lookupAndFormat('謎の未知語');
      expect(trUnk.translationStatus, 'no_lexicon_match');
    });
  });

  group('V5 OBB 1024 rotated NMS + IosCoreMLBackend E2E whole-page flow', () {
    test('IosCoreMLBackend runs fresh capture then cached-logits reprocess cleanly', () async {
      final bridge = _MockIosCoreMLBridge();
      const c2i = {
        '季': 1, '節': 2, 'の': 3, '先': 4, '付': 5, 'と': 6, '前': 7, '菜': 8,
        '(': 9, '（': 10, ')': 11, '）': 12,
      };
      final lm = CharNGramLMV2(
        schemaVersion: 2,
        contextTransform: 'dig_unk_v2',
        vocabChars: c2i.keys.toList(),
        uni: const {'季': 10, '節': 10, 'の': 20, '先': 10, '付': 10, 'と': 20, '前': 10, '菜': 10},
        bi: const {},
        tri: const {},
        quad: const {},
      );
      final lex = LexiconV3Index.fromTerms(['季節', '先付', '前菜']);
      final backend = IosCoreMLBackend(bridge: bridge, charToId: c2i, lm: lm, lexiconIndex: lex);

      final init = await backend.initialize();
      expect(init['status'], 'ready');

      final r1 = await backend.startCapture(
        imageBytes: Uint8List.fromList([1, 2, 3, 4]),
        filename: 'iphone_shutter.png',
        streamWidth: 1024,
        streamHeight: 1024,
        captureSource: 'ios_avfoundation_stream_buffer',
        settings: AppSettings.defaults,
      );
      // Wait for serialized execution
      PageRun cur = r1;
      for (var i = 0; i < 40 && !cur.isTerminal; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
        cur = await backend.fetchRun(r1.runId);
      }
      expect(cur.status, 'completed');
      expect(cur.cacheStatus, 'fresh_full_inference');
      // Overlapping duplicate OBB box suppressed by rotated ProbIoU NMS -> 1 crop
      expect(cur.crops.length, 1);
      expect(cur.crops.first.decodedOcr, '季節の先付と前菜');
      expect(cur.crops.first.translation?.primaryEnglish,
          'Seasonal Starter (Sakizuke) and Appetizers');
      expect(bridge.detectorCalls, 1);
      expect(bridge.recognizerCalls, 1);

      // Reprocess with different alpha (same conf=0.35) MUST reuse cached logits!
      final r2 = await backend.startReprocess(
        captureId: cur.captureId,
        settings: AppSettings.defaults.copyWith(charLmAlpha: 0.15),
      );
      PageRun cur2 = r2;
      for (var i = 0; i < 40 && !cur2.isTerminal; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
        cur2 = await backend.fetchRun(r2.runId);
      }
      expect(cur2.status, 'completed');
      expect(cur2.cacheStatus, 'reused_cached_logits');
      expect(cur2.reusedCachedLogits, isTrue);
      expect(cur2.timingsMs.detectMs, 0.0);
      expect(cur2.timingsMs.recognizeMs, 0.0);
      expect(bridge.detectorCalls, 1);
      expect(bridge.recognizerCalls, 1);
    });
  });
}
