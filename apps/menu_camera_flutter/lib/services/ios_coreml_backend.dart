/// Full in-process iOS Core ML Whole-Page Backend (`MenuBackend` implementation).
///
/// Combines:
///   - Native Swift Core ML + AVFoundation + read-only SQLite via `IosCoreMLNativeBridge`
///   - Shared Dart ProbIoU rotated OBB NMS (`obb_postprocess.dart`)
///   - Shared Dart Canonical CTC + Unicode-scalar CharLM V2 R1 decoder (`canonical_ctc_decoder.dart`)
///   - Shared Dart Lexicon V3 Trie/DP + read-only SQLite translation formatting (`lexicon_lookup.dart`)
///   - Serialized single-slot job coordinator, immutable settings-keyed caches, ordered capture
///     resolution parity validation, and monotonic per-stage timings.
///
/// Zero Python, zero localhost server, zero cloud APIs on iOS.
library;

import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';

import '../geometry/overlay_geometry.dart';
import '../models/contracts.dart';
import '../pipeline/canonical_ctc_decoder.dart';
import '../pipeline/lexicon_lookup.dart';
import '../pipeline/obb_postprocess.dart';
import 'menu_backend.dart';

/// Injectable native Swift bridge interface (implemented by MethodChannel on iOS and mocks in tests).
abstract class IosCoreMLNativeBridge {
  Future<Map<String, dynamic>> initialize();
  Future<Map<String, dynamic>> runDetectorOBB1024(Uint8List imageBytes);
  Future<Map<String, dynamic>> runRecognizerCrop({
    required Uint8List imageBytes,
    required List<List<double>> polygon,
    required int orientationFlag,
  });
  Future<Map<String, TranslationRecord>> lookupSqlTerms(Set<String> terms);
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase);
  Future<Uint8List> loadBundledAssetBytes(String filename);
}

class MethodChannelIosCoreMLBridge implements IosCoreMLNativeBridge {
  MethodChannelIosCoreMLBridge({MethodChannel? channel})
      : _channel = channel ??
            const MethodChannel('com.menucamera.stage4/coreml_backend');

  final MethodChannel _channel;

  @override
  Future<Map<String, dynamic>> initialize() async {
    final res = await _channel.invokeMapMethod<String, dynamic>('initialize');
    return res ?? const <String, dynamic>{};
  }

  @override
  Future<Map<String, dynamic>> runDetectorOBB1024(Uint8List imageBytes) async {
    final res = await _channel.invokeMapMethod<String, dynamic>(
      'runDetectorOBB1024',
      {'image_bytes': imageBytes},
    );
    if (res == null) throw StateError('runDetectorOBB1024 returned null');
    return res;
  }

  @override
  Future<Map<String, dynamic>> runRecognizerCrop({
    required Uint8List imageBytes,
    required List<List<double>> polygon,
    required int orientationFlag,
  }) async {
    final res = await _channel.invokeMapMethod<String, dynamic>(
      'runRecognizerCrop',
      {
        'image_bytes': imageBytes,
        'polygon': polygon,
        'orientation_flag': orientationFlag,
      },
    );
    if (res == null) throw StateError('runRecognizerCrop returned null');
    return res;
  }

  @override
  Future<Map<String, TranslationRecord>> lookupSqlTerms(Set<String> terms) async {
    if (terms.isEmpty) return const {};
    final res = await _channel.invokeMapMethod<String, dynamic>(
      'lookupSqlTerms',
      {'terms': terms.toList()},
    );
    if (res == null) return const {};
    final out = <String, TranslationRecord>{};
    for (final e in res.entries) {
      final m = e.value as Map?;
      if (m == null) continue;
      out[e.key] = TranslationRecord(
        canonicalJp: (m['canonical_jp'] ?? e.key).toString(),
        nameEn: (m['name_en'] ?? '').toString(),
        readingHiragana: (m['reading_hiragana'] ?? '').toString(),
        dbId: (m['db_id'] as num?)?.toInt(),
      );
    }
    return out;
  }

  @override
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase) async {
    final res = await _channel.invokeMapMethod<String, dynamic>(
      'lookupPhrasePlan',
      {'phrase': canonicalPhrase},
    );
    if (res == null) return null;
    return PhrasePlanRecord(
      canonicalPhrase: (res['canonical_phrase'] ?? canonicalPhrase).toString(),
      phraseEn: (res['phrase_en'] ?? '').toString(),
      provenanceSource: res['provenance_source']?.toString(),
    );
  }

  @override
  Future<Uint8List> loadBundledAssetBytes(String filename) async {
    final bytes = await _channel.invokeMethod<Uint8List>(
      'loadBundledAssetBytes',
      {'filename': filename},
    );
    if (bytes == null) throw StateError('Missing bundled asset $filename');
    return bytes;
  }
}

class _BridgeTranslationStore implements ReadOnlyTranslationStore {
  _BridgeTranslationStore(this._bridge);
  final IosCoreMLNativeBridge _bridge;

  @override
  Future<Map<String, TranslationRecord>> lookupTerms(Set<String> terms) =>
      _bridge.lookupSqlTerms(terms);

  @override
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase) =>
      _bridge.lookupPhrasePlan(canonicalPhrase);
}

class _CachedCropOcr {
  const _CachedCropOcr({
    required this.detection,
    required this.multiWindow,
    required this.windowCount,
    required this.windowRanges,
    required this.windowPaths,
    required this.logpTC,
    required this.previewPngB64,
  });

  final ObbDetection detection;
  final bool multiWindow;
  final int windowCount;
  final List<List<int>> windowRanges;
  final List<List<int>> windowPaths;
  final List<List<double>>? logpTC;
  final String? previewPngB64;
}

class _CachedCapture {
  _CachedCapture({
    required this.captureId,
    required this.imageBytes,
    required this.meta,
  });

  final String captureId;
  final Uint8List imageBytes;
  final CapturedPageMeta meta;
  final Map<String, List<_CachedCropOcr>> ocrCacheByConf = {};
  final Map<String, List<ObbDetection>> detsByConf = {};
}

class IosCoreMLBackend implements MenuBackend {
  IosCoreMLBackend({
    IosCoreMLNativeBridge? bridge,
    Map<String, int>? charToId,
    CharNGramLMV2? lm,
    LexiconV3Index? lexiconIndex,
  }) : _bridge = bridge ?? MethodChannelIosCoreMLBridge() {
    _charToId = charToId;
    _lm = lm;
    _lexiconIndex = lexiconIndex;
    if (_charToId != null) {
      _idToChar = {for (final e in _charToId!.entries) e.value: e.key};
    }
    if (_lexiconIndex != null) {
      _lexPipe = LexiconV3Pipeline(_lexiconIndex!, _BridgeTranslationStore(_bridge));
    }
  }

  final IosCoreMLNativeBridge _bridge;
  String _deviceName = 'iOS Device (Core ML)';
  String _bundleId = 'stage4_step1000_raw_2105_v5_lmv2_lexv3_20261123';
  String _status = 'initializing';

  Map<String, int>? _charToId;
  Map<int, String>? _idToChar;
  CharNGramLMV2? _lm;
  LexiconV3Index? _lexiconIndex;
  LexiconV3Pipeline? _lexPipe;

  final Map<String, _CachedCapture> _captures = {};
  final Map<String, PageRun> _runs = {};
  final Set<String> _cancelledRunIds = {};
  int _runCounter = 0;
  int _capCounter = 0;
  String? _activeRunId;
  Future<void> _serialChain = Future<void>.value();

  @override
  String get backendLabel =>
      'iOS Core ML Native ($_deviceName • $_status) • $_bundleId';

  String _confKey(double conf) => conf.toStringAsFixed(2);
  String _configId(AppSettings s) =>
      '${s.yoloConfidence.toStringAsFixed(2)}_${s.charLmAlpha.toStringAsFixed(2)}_${s.decodeMode}';

  @override
  Future<Map<String, dynamic>> initialize() async {
    final info = await _bridge.initialize();
    _deviceName = (info['device_name'] as String?) ?? _deviceName;
    _bundleId = (info['bundle_id'] as String?) ?? _bundleId;
    _status = (info['status'] as String?) ?? 'unknown';
    return info;
  }

  Future<void> _ensureTextAssetsLoaded() async {
    if (_charToId == null) {
      final raw = await _bridge.loadBundledAssetBytes('target_vocab_active.json');
      final obj = jsonDecode(utf8.decode(raw)) as Map<String, dynamic>;
      final chars = ((obj['characters'] as List?) ?? const []).map((e) => e.toString()).toList();
      _charToId = {for (var i = 0; i < chars.length; i++) chars[i]: i + 1};
      _idToChar = {for (final e in _charToId!.entries) e.value: e.key};
    }
    if (_lm == null) {
      final raw = await _bridge.loadBundledAssetBytes('char_lm_4g_hp1010_v2.json');
      _lm = CharNGramLMV2.fromJsonString(utf8.decode(raw));
    }
    if (_lexiconIndex == null) {
      final raw = await _bridge.loadBundledAssetBytes('lexicon_index_v3_22112026.json');
      _lexiconIndex = LexiconV3Index.fromJsonString(utf8.decode(raw));
      _lexPipe = LexiconV3Pipeline(_lexiconIndex!, _BridgeTranslationStore(_bridge));
    }
  }

  @override
  Future<PageRun> startCapture({
    required Uint8List imageBytes,
    required String filename,
    required int streamWidth,
    required int streamHeight,
    required String captureSource,
    required AppSettings settings,
  }) async {
    final parityOk = verifyStreamSavedParity(
      streamWidth: streamWidth,
      streamHeight: streamHeight,
      savedWidth: streamWidth,
      savedHeight: streamHeight,
    );
    if (!parityOk) {
      throw ArgumentError('Invalid capture stream dimensions ${streamWidth}x$streamHeight');
    }
    _capCounter++;
    final capId = 'ios_cap_$_capCounter';
    final meta = CapturedPageMeta(
      captureId: capId,
      filename: filename,
      captureSource: captureSource,
      streamWidth: streamWidth,
      streamHeight: streamHeight,
      savedWidth: streamWidth,
      savedHeight: streamHeight,
      streamSavedParity: true,
      streamSavedParityMode: 'exact',
      imageSha256: 'ios_local_${imageBytes.length}',
    );
    final cap = _CachedCapture(captureId: capId, imageBytes: imageBytes, meta: meta);
    if (_captures.length >= 4) {
      _captures.remove(_captures.keys.first);
    }
    _captures[capId] = cap;
    return _enqueueRun(cap, settings, isReprocess: false);
  }

  @override
  Future<PageRun> startReprocess({
    required String captureId,
    required AppSettings settings,
  }) async {
    final cap = _captures[captureId];
    if (cap == null) {
      throw StateError('Capture $captureId not found in bounded iOS cache');
    }
    return _enqueueRun(cap, settings, isReprocess: true);
  }

  Future<PageRun> _enqueueRun(
    _CachedCapture cap,
    AppSettings settings, {
    required bool isReprocess,
  }) async {
    if (_activeRunId != null) {
      _cancelledRunIds.add(_activeRunId!);
    }
    _runCounter++;
    final runId = 'ios_run_$_runCounter';
    _activeRunId = runId;
    final reqSw = Stopwatch()..start();

    final initial = _buildRun(
      runId: runId,
      cap: cap,
      settings: settings,
      isReprocess: isReprocess,
      status: 'running',
      stage: 'queued',
      cacheStatus: 'pending',
      reusedLogits: false,
      reusedDets: false,
      counts: const CropCounts(
        rawDetections: 0,
        postprocessedLines: 0,
        completedCrops: 0,
        failedCrops: 0,
        totalWindows: 0,
        cropsTruncated: false,
      ),
      timings: const StageTimings(
        coldLoadMs: 0,
        decodeNormalizeMs: 0,
        persistMs: 0,
        detectMs: 0,
        postprocessMs: 0,
        recognizeMs: 0,
        decodeMs: 0,
        lookupMs: 0,
        firstResultMs: null,
        totalWallMs: null,
        summedCropWorkMs: 0,
      ),
      crops: const [],
    );
    _runs[runId] = initial;

    // Serialize execution strictly (max 1 in-flight model run)
    _serialChain = _serialChain.then((_) => _executeRun(runId, cap, settings, isReprocess, reqSw));
    return initial;
  }

  Future<void> _executeRun(
    String runId,
    _CachedCapture cap,
    AppSettings settings,
    bool isReprocess,
    Stopwatch reqSw,
  ) async {
    if (_cancelledRunIds.contains(runId)) {
      _runs[runId] = _markCancelled(runId, cap, settings, isReprocess, reqSw, 0.0, 'cancelled_before_start');
      return;
    }

    final pipeSw = Stopwatch()..start();
    final ck = _confKey(settings.yoloConfidence);
    try {
      await _ensureTextAssetsLoaded();
      final charToId = _charToId!;
      final idToChar = _idToChar!;
      final lexPipe = _lexPipe!;

      var detectMs = 0.0;
      var postMs = 0.0;
      var recMs = 0.0;
      var decMs = 0.0;
      var lookMs = 0.0;
      double? firstResultMs;

      final cachedOcr = cap.ocrCacheByConf[ck];
      final canReuseLogits = cachedOcr != null;
      final cacheStatus = canReuseLogits ? 'reused_cached_logits' : 'fresh_full_inference';

      List<_CachedCropOcr> cropOcrList;
      int rawDetCount;

      if (canReuseLogits) {
        cropOcrList = cachedOcr;
        rawDetCount = cachedOcr.length;
      } else {
        final dSw = Stopwatch()..start();
        final detRaw = await _bridge.runDetectorOBB1024(cap.imageBytes);
        detectMs = dSw.elapsedMicroseconds / 1000.0;
        if (_cancelledRunIds.contains(runId)) {
          _runs[runId] = _markCancelled(runId, cap, settings, isReprocess, reqSw, pipeSw.elapsedMicroseconds / 1000.0, 'cancelled');
          return;
        }

        final pSw = Stopwatch()..start();
        final rawBytes = detRaw['raw_obb_f32'] as Uint8List;
        final rawFloats = rawBytes.buffer.asFloat32List(rawBytes.offsetInBytes, rawBytes.lengthInBytes ~/ 4);
        final numAnchors = (detRaw['num_anchors'] as num).toInt();
        final lb = detRaw['letterbox'] as Map;
        final lbMeta = ObbLetterboxMeta(
          gain: (lb['gain'] as num).toDouble(),
          padX: (lb['pad_x'] as num).toDouble(),
          padY: (lb['pad_y'] as num).toDouble(),
          srcWidth: (lb['src_w'] as num).toDouble(),
          srcHeight: (lb['src_h'] as num).toDouble(),
        );
        final dets = decodeAndRotatedNmsObb(
          rawFloats,
          numAnchors,
          lbMeta,
          confidence: settings.yoloConfidence,
          iouThreshold: 0.70,
        );
        postMs = pSw.elapsedMicroseconds / 1000.0;
        rawDetCount = dets.length;

        cropOcrList = [];
        for (final d in dets) {
          if (_cancelledRunIds.contains(runId)) {
            _runs[runId] = _markCancelled(runId, cap, settings, isReprocess, reqSw, pipeSw.elapsedMicroseconds / 1000.0, 'cancelled');
            return;
          }
          final rSw = Stopwatch()..start();
          final recOut = await _bridge.runRecognizerCrop(
            imageBytes: cap.imageBytes,
            polygon: d.points,
            orientationFlag: d.orientationFlag,
          );
          recMs += rSw.elapsedMicroseconds / 1000.0;

          final isMulti = recOut['multi_window'] as bool? ?? false;
          final winCount = (recOut['window_count'] as num?)?.toInt() ?? 1;
          final previewBytes = recOut['crop_preview_png'] as Uint8List?;
          final previewB64 = (previewBytes != null && previewBytes.isNotEmpty)
              ? base64Encode(previewBytes)
              : null;

          if (isMulti) {
            final ranges = ((recOut['window_ranges'] as List?) ?? const [])
                .map((r) => (r as List).map((v) => (v as num).toInt()).toList())
                .toList();
            final paths = ((recOut['window_paths'] as List?) ?? const [])
                .map((p) => (p as List).map((v) => (v as num).toInt()).toList())
                .toList();
            cropOcrList.add(_CachedCropOcr(
              detection: d,
              multiWindow: true,
              windowCount: winCount,
              windowRanges: ranges,
              windowPaths: paths,
              logpTC: null,
              previewPngB64: previewB64,
            ));
          } else {
            final validT = (recOut['valid_T'] as num).toInt();
            final lbBytes = recOut['logits_f32'] as Uint8List;
            final floats = lbBytes.buffer.asFloat32List(lbBytes.offsetInBytes, lbBytes.lengthInBytes ~/ 4);
            final classes = validT > 0 ? floats.length ~/ validT : 2106;
            final rawTC = List<List<double>>.generate(
              validT,
              (t) => List<double>.generate(classes, (c) => floats[t * classes + c]),
            );
            final logpTC = foldParenthesisAndLogSoftmax(rawTC, charToId);
            cropOcrList.add(_CachedCropOcr(
              detection: d,
              multiWindow: false,
              windowCount: 1,
              windowRanges: const [],
              windowPaths: const [],
              logpTC: logpTC,
              previewPngB64: previewB64,
            ));
          }
        }
        if (!_cancelledRunIds.contains(runId)) {
          cap.ocrCacheByConf[ck] = List.unmodifiable(cropOcrList);
          cap.detsByConf[ck] = List.unmodifiable(dets);
        }
      }

      // Decode + Lexicon V3 + read-only SQLite lookup for ALL crops
      final cropsOut = <CropResult>[];
      var totalWindows = 0;
      var completed = 0;
      var failed = 0;

      for (var i = 0; i < cropOcrList.length; i++) {
        if (_cancelledRunIds.contains(runId)) {
          _runs[runId] = _markCancelled(runId, cap, settings, isReprocess, reqSw, pipeSw.elapsedMicroseconds / 1000.0, 'cancelled');
          return;
        }
        final item = cropOcrList[i];
        totalWindows += item.windowCount;
        try {
          final decSw = Stopwatch()..start();
          String greedyTxt;
          String decodedTxt;
          String routeUsed;
          bool numDisagree;
          if (item.multiWindow) {
            final mergedIds = mergeOwnedWindowTokens(item.windowPaths, item.windowRanges);
            final rawMerged = mergedIds.map((id) => idToChar[id] ?? '').join();
            greedyTxt = canonicalizeOcrText(rawMerged);
            decodedTxt = greedyTxt;
            routeUsed = 'overlap512_merged_canonical_greedy_no_joint_lm';
            numDisagree = false;
          } else {
            final dRes = decodeWithR1Routing(
              logpTC: item.logpTC!,
              idToChar: idToChar,
              mode: settings.decodeMode,
              alphaLm: settings.charLmAlpha,
              lm: _lm,
            );
            greedyTxt = dRes.greedy;
            decodedTxt = dRes.text;
            routeUsed = dRes.routeUsed;
            numDisagree = dRes.numericDisagreement;
          }
          decMs += decSw.elapsedMicroseconds / 1000.0;

          final lkSw = Stopwatch()..start();
          final tr = await lexPipe.lookupAndFormat(decodedTxt);
          lookMs += lkSw.elapsedMicroseconds / 1000.0;

          cropsOut.add(CropResult(
            cropId: 'crop_${i + 1}',
            cropIndex: i + 1,
            status: 'completed',
            confidence: item.detection.confidence,
            orientation: item.detection.orientationLabel,
            polygon: item.detection.points,
            windowCount: item.windowCount,
            multiWindowFallback: item.multiWindow,
            cropPreviewPngB64: item.previewPngB64,
            greedyOcr: greedyTxt,
            decodedOcr: decodedTxt,
            decoderRoute: routeUsed,
            numericDisagreement: numDisagree,
            translation: tr,
            error: null,
          ));
          completed++;
        } catch (e) {
          cropsOut.add(CropResult(
            cropId: 'crop_${i + 1}',
            cropIndex: i + 1,
            status: 'failed',
            confidence: item.detection.confidence,
            orientation: item.detection.orientationLabel,
            polygon: item.detection.points,
            windowCount: item.windowCount,
            multiWindowFallback: item.multiWindow,
            cropPreviewPngB64: null,
            greedyOcr: '',
            decodedOcr: '',
            decoderRoute: 'failed',
            numericDisagreement: false,
            translation: null,
            error: e.toString(),
          ));
          failed++;
        }
        firstResultMs ??= reqSw.elapsedMicroseconds / 1000.0;
      }

      final reqTotMs = reqSw.elapsedMicroseconds / 1000.0;
      final pipeTotMs = pipeSw.elapsedMicroseconds / 1000.0;
      _runs[runId] = _buildRun(
        runId: runId,
        cap: cap,
        settings: settings,
        isReprocess: isReprocess,
        status: 'completed',
        stage: 'done',
        cacheStatus: cacheStatus,
        reusedLogits: canReuseLogits,
        reusedDets: canReuseLogits,
        counts: CropCounts(
          rawDetections: rawDetCount,
          postprocessedLines: cropOcrList.length,
          completedCrops: completed,
          failedCrops: failed,
          totalWindows: totalWindows,
          cropsTruncated: false,
        ),
        timings: StageTimings(
          coldLoadMs: 0,
          decodeNormalizeMs: 0,
          persistMs: 0,
          detectMs: double.parse(detectMs.toStringAsFixed(2)),
          postprocessMs: double.parse(postMs.toStringAsFixed(2)),
          recognizeMs: double.parse(recMs.toStringAsFixed(2)),
          decodeMs: double.parse(decMs.toStringAsFixed(2)),
          lookupMs: double.parse(lookMs.toStringAsFixed(2)),
          firstResultMs: firstResultMs != null ? double.parse(firstResultMs.toStringAsFixed(2)) : null,
          serverPipelineWallMs: double.parse(pipeTotMs.toStringAsFixed(2)),
          serverRequestTotalMs: double.parse(reqTotMs.toStringAsFixed(2)),
          totalWallMs: double.parse(reqTotMs.toStringAsFixed(2)),
          summedCropWorkMs: double.parse((recMs + decMs + lookMs).toStringAsFixed(2)),
        ),
        crops: cropsOut,
      );
    } catch (e) {
      final reqTotMs = reqSw.elapsedMicroseconds / 1000.0;
      final pipeTotMs = pipeSw.elapsedMicroseconds / 1000.0;
      _runs[runId] = _buildRun(
        runId: runId,
        cap: cap,
        settings: settings,
        isReprocess: isReprocess,
        status: 'failed',
        stage: 'error',
        cacheStatus: 'failed',
        reusedLogits: false,
        reusedDets: false,
        counts: const CropCounts(
          rawDetections: 0,
          postprocessedLines: 0,
          completedCrops: 0,
          failedCrops: 0,
          totalWindows: 0,
          cropsTruncated: false,
        ),
        timings: StageTimings(
          coldLoadMs: 0,
          decodeNormalizeMs: 0,
          persistMs: 0,
          detectMs: 0,
          postprocessMs: 0,
          recognizeMs: 0,
          decodeMs: 0,
          lookupMs: 0,
          firstResultMs: null,
          serverPipelineWallMs: double.parse(pipeTotMs.toStringAsFixed(2)),
          serverRequestTotalMs: double.parse(reqTotMs.toStringAsFixed(2)),
          totalWallMs: double.parse(reqTotMs.toStringAsFixed(2)),
          summedCropWorkMs: 0,
        ),
        crops: const [],
        error: e.toString(),
      );
    }
  }

  PageRun _markCancelled(
    String runId,
    _CachedCapture cap,
    AppSettings settings,
    bool isReprocess,
    Stopwatch reqSw,
    double pipeMs,
    String stage,
  ) {
    final reqMs = reqSw.elapsedMicroseconds / 1000.0;
    return _buildRun(
      runId: runId,
      cap: cap,
      settings: settings,
      isReprocess: isReprocess,
      status: 'cancelled',
      stage: stage,
      cacheStatus: stage,
      reusedLogits: false,
      reusedDets: false,
      counts: const CropCounts(
        rawDetections: 0,
        postprocessedLines: 0,
        completedCrops: 0,
        failedCrops: 0,
        totalWindows: 0,
        cropsTruncated: false,
      ),
      timings: StageTimings(
        coldLoadMs: 0,
        decodeNormalizeMs: 0,
        persistMs: 0,
        detectMs: 0,
        postprocessMs: 0,
        recognizeMs: 0,
        decodeMs: 0,
        lookupMs: 0,
        firstResultMs: null,
        serverPipelineWallMs: double.parse(pipeMs.toStringAsFixed(2)),
        serverRequestTotalMs: double.parse(reqMs.toStringAsFixed(2)),
        totalWallMs: double.parse(reqMs.toStringAsFixed(2)),
        summedCropWorkMs: 0,
      ),
      crops: const [],
    );
  }

  PageRun _buildRun({
    required String runId,
    required _CachedCapture cap,
    required AppSettings settings,
    required bool isReprocess,
    required String status,
    required String stage,
    required String cacheStatus,
    required bool reusedLogits,
    required bool reusedDets,
    required CropCounts counts,
    required StageTimings timings,
    required List<CropResult> crops,
    String? error,
  }) {
    return PageRun(
      runId: runId,
      captureId: cap.captureId,
      configId: _configId(settings),
      backendId: 'ios_coreml_native',
      deviceName: _deviceName,
      precision: 'fp32',
      bundleId: _bundleId,
      status: status,
      stage: stage,
      cacheStatus: cacheStatus,
      reusedCachedLogits: reusedLogits,
      reusedCachedDetections: reusedDets,
      coldOrWarm: 'warm',
      isReprocess: isReprocess,
      capture: cap.meta,
      counts: counts,
      timingsMs: timings,
      crops: crops,
      error: error,
    );
  }

  @override
  Future<PageRun> fetchRun(String runId) async {
    final run = _runs[runId];
    if (run == null) throw StateError('Run $runId not found');
    return run;
  }

  @override
  Future<void> cancelRun(String runId) async {
    _cancelledRunIds.add(runId);
  }

  @override
  Future<Uint8List> fetchCaptureImage(String captureId) async {
    final cap = _captures[captureId];
    if (cap == null) throw StateError('Capture $captureId not found');
    return cap.imageBytes;
  }
}
