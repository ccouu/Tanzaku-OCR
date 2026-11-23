/// Shared Stage 4 Canonical CTC + Unicode-Scalar 4-Gram CharLM V2 R1 Decoder.
///
/// Exact Dart port of `src/recognition/canonical_ctc.py`, `src/recognition/char_lm_beam.py`,
/// and `src/recognition/long_line_inference.py` (`owned_tokens` + `window_ranges`).
/// Zero Python/cloud dependency; runs in-process on iOS and is unit-tested on Windows
/// against frozen Python golden fixtures.
library;

import 'dart:convert';
import 'dart:math' as math;

const double _negInf = double.negativeInfinity;

double _logSumExp2(double a, double b) {
  if (a == _negInf) return b;
  if (b == _negInf) return a;
  if (a > b) {
    return a + math.log(1.0 + math.exp(b - a));
  }
  return b + math.log(1.0 + math.exp(a - b));
}

/// Narrow digit + parenthesis canonicalization (no global NFKC/confusable rewrite).
String canonicalizeOcrText(String input) {
  final sb = StringBuffer();
  for (final cp in input.runes) {
    if (cp >= 0xFF10 && cp <= 0xFF19) {
      // Fullwidth digits ０-９ -> ASCII 0-9
      sb.writeCharCode(cp - 0xFF10 + 0x30);
    } else if (cp == 0xFF08) {
      // Fullwidth （ -> (
      sb.writeCharCode(0x28);
    } else if (cp == 0xFF09) {
      // Fullwidth ） -> )
      sb.writeCharCode(0x29);
    } else {
      sb.writeCharCode(cp);
    }
  }
  return sb.toString().trim();
}

bool _isAsciiDigit(String ch) {
  if (ch.length != 1) return false;
  final cu = ch.codeUnitAt(0);
  return cu >= 0x30 && cu <= 0x39;
}

/// Pool fullwidth/ASCII parenthesis probability mass BEFORE CTC collapse/log-softmax.
/// Preserves exact 2,106-class IDs and blank=0 without dropping or remapping classes.
List<List<double>> foldParenthesisAndLogSoftmax(
  List<List<double>> rawLogitsTC,
  Map<String, int> charToId,
) {
  final wideOpen = charToId['（'];
  final asciiOpen = charToId['('];
  final wideClose = charToId['）'];
  final asciiClose = charToId[')'];

  return rawLogitsTC.map((frame) {
    final x = List<double>.from(frame);
    if (wideOpen != null && asciiOpen != null) {
      x[asciiOpen] = _logSumExp2(x[wideOpen], x[asciiOpen]);
      x[wideOpen] = _negInf;
    }
    if (wideClose != null && asciiClose != null) {
      x[asciiClose] = _logSumExp2(x[wideClose], x[asciiClose]);
      x[wideClose] = _negInf;
    }
    // Stable log_softmax over all classes
    var maxV = _negInf;
    for (final v in x) {
      if (v > maxV) maxV = v;
    }
    var sumExp = 0.0;
    for (final v in x) {
      if (v > _negInf) {
        sumExp += math.exp(v - maxV);
      }
    }
    final lse = maxV + math.log(sumExp);
    for (var i = 0; i < x.length; i++) {
      x[i] = x[i] == _negInf ? _negInf : (x[i] - lse);
    }
    return x;
  }).toList();
}

/// Canonical CTC greedy collapse from log-probabilities `[T, C]`.
String ctcGreedyDecode(
  List<List<double>> logpTC,
  Map<int, String> idToChar,
) {
  final sb = StringBuffer();
  var prev = 0;
  for (final frame in logpTC) {
    var bestIdx = 0;
    var bestVal = frame[0];
    for (var i = 1; i < frame.length; i++) {
      if (frame[i] > bestVal) {
        bestVal = frame[i];
        bestIdx = i;
      }
    }
    if (bestIdx != 0 && bestIdx != prev) {
      final ch = idToChar[bestIdx];
      if (ch != null && ch.isNotEmpty) {
        sb.write(ch);
      }
    }
    prev = bestIdx;
  }
  return sb.toString();
}

/// Immutable 4-gram Character LM (Schema V2, `dig_unk_v2`, 326 expanded-char Unicode fallback).
class CharNGramLMV2 {
  CharNGramLMV2({
    required this.schemaVersion,
    required this.contextTransform,
    required this.vocabChars,
    required this.uni,
    required this.bi,
    required this.tri,
    required this.quad,
  }) : assert(schemaVersion == 2, 'Only schema_version=2 CharLM is supported') {
    if (contextTransform != 'dig_unk_v2') {
      throw FormatException(
        'Incompatible CharLM context_transform=$contextTransform; expected dig_unk_v2',
      );
    }
    vocabSize = math.max(1, vocabChars.length);
    var totalUni = 0;
    for (final v in uni.values) {
      totalUni += v;
    }
    defaultUni = 1.0 / (totalUni + vocabSize);
    pUni = {
      for (final ch in vocabChars)
        ch: ((uni[ch] ?? 0) + 1.0) / (totalUni + vocabSize),
    };
    biTot = {
      for (final e in bi.entries)
        e.key: e.value.values.fold<int>(0, (a, b) => a + b),
    };
    triTot = {
      for (final e in tri.entries)
        e.key: e.value.values.fold<int>(0, (a, b) => a + b),
    };
    quadTot = {
      for (final e in quad.entries)
        e.key: e.value.values.fold<int>(0, (a, b) => a + b),
    };
  }

  factory CharNGramLMV2.fromJsonString(String jsonText) {
    final obj = jsonDecode(jsonText) as Map<String, dynamic>;
    final sv = (obj['schema_version'] as num?)?.toInt() ?? 0;
    final ctx = (obj['context_transform'] as String?) ?? '';
    final vc = ((obj['vocab_chars'] as List?) ?? const [])
        .map((e) => e.toString())
        .toList();
    Map<String, int> parseFlat(dynamic raw) {
      final m = (raw as Map?) ?? const {};
      return {for (final e in m.entries) e.key.toString(): (e.value as num).toInt()};
    }

    Map<String, Map<String, int>> parseNested(dynamic raw) {
      final m = (raw as Map?) ?? const {};
      return {
        for (final e in m.entries) e.key.toString(): parseFlat(e.value),
      };
    }

    return CharNGramLMV2(
      schemaVersion: sv,
      contextTransform: ctx,
      vocabChars: vc,
      uni: parseFlat(obj['uni']),
      bi: parseNested(obj['bi']),
      tri: parseNested(obj['tri']),
      quad: parseNested(obj['quad']),
    );
  }

  final int schemaVersion;
  final String contextTransform;
  final List<String> vocabChars;
  final Map<String, int> uni;
  final Map<String, Map<String, int>> bi;
  final Map<String, Map<String, int>> tri;
  final Map<String, Map<String, int>> quad;

  late final int vocabSize;
  late final double defaultUni;
  late final Map<String, double> pUni;
  late final Map<String, int> biTot;
  late final Map<String, int> triTot;
  late final Map<String, int> quadTot;
  final double uniformDigitLp = math.log(0.1);

  double stepLogp(
    String p3,
    String p2,
    String p1,
    String ch, {
    bool digitNeutral = true,
  }) {
    if (digitNeutral) {
      if (_isAsciiDigit(ch)) return uniformDigitLp;
      if (_isAsciiDigit(p1)) p1 = '<DIG>';
      if (_isAsciiDigit(p2)) p2 = '<DIG>';
      if (_isAsciiDigit(p3)) p3 = '<DIG>';
    }
    final v = vocabSize.toDouble();
    final pu = pUni[ch] ?? defaultUni;
    final bt = biTot[p1] ?? 0;
    final pb = bt > 0 ? ((bi[p1]?[ch] ?? 0) + 0.5 * pu * v) / (bt + 0.5 * v) : pu;
    final k2 = '$p2\t$p1';
    final tt = triTot[k2] ?? 0;
    final pt = tt > 0 ? ((tri[k2]?[ch] ?? 0) + 0.5 * pb * v) / (tt + 0.5 * v) : pb;
    final k3 = '$p3\t$p2\t$p1';
    final qt = quadTot[k3] ?? 0;
    final pq = qt > 0 ? ((quad[k3]?[ch] ?? 0) + 0.5 * pt * v) / (qt + 0.5 * v) : pt;
    final mix = 0.45 * pq + 0.30 * pt + 0.15 * pb + 0.10 * pu;
    return math.log(mix < 1e-12 ? 1e-12 : mix);
  }
}

class _BeamState {
  _BeamState(this.pb, this.pnb, this.lms);
  double pb;
  double pnb;
  double lms;
}

/// True CTC prefix beam search matching `ctc_beam_decode_stateless` in `char_lm_beam.py`.
String ctcPrefixBeamDecode(
  List<List<double>> logpTC,
  Map<int, String> idToChar, {
  CharNGramLMV2? lm,
  double alphaLm = 0.30,
  int beamWidth = 5,
  int topkT = 6,
  bool digitNeutral = true,
}) {
  final useLm = lm != null && alphaLm > 0.0;
  var beams = <String, _BeamState>{'': _BeamState(0.0, _negInf, 0.0)};
  final lmCache = <String, double>{};

  for (final frame in logpTC) {
    final pBlk = frame[0];
    // Top-k non-blank classes
    final indexed = <MapEntry<int, double>>[];
    for (var i = 1; i < frame.length; i++) {
      if (frame[i] > _negInf) {
        indexed.add(MapEntry(i, frame[i]));
      }
    }
    indexed.sort((a, b) => b.value.compareTo(a.value));
    final kEff = math.min(topkT, indexed.length);
    final cands = indexed.sublist(0, kEff);

    final nextBeams = <String, _BeamState>{};
    void acc(String pref, double pb, double pnb, double lms) {
      final cur = nextBeams[pref];
      if (cur == null) {
        nextBeams[pref] = _BeamState(pb, pnb, lms);
      } else {
        cur.pb = _logSumExp2(cur.pb, pb);
        cur.pnb = _logSumExp2(cur.pnb, pnb);
      }
    }

    for (final entry in beams.entries) {
      final pref = entry.key;
      final st = entry.value;
      final ptot = _logSumExp2(st.pb, st.pnb);
      acc(pref, ptot + pBlk, _negInf, st.lms);

      final runes = pref.runes.toList();
      final lastCh = runes.isNotEmpty ? String.fromCharCode(runes.last) : null;
      final p3 = runes.length >= 3 ? String.fromCharCode(runes[runes.length - 3]) : '<BOS>';
      final p2 = runes.length >= 2 ? String.fromCharCode(runes[runes.length - 2]) : '<BOS>';
      final p1 = runes.isNotEmpty ? String.fromCharCode(runes.last) : '<BOS>';

      for (final cand in cands) {
        final ch = idToChar[cand.key];
        if (ch == null || ch.isEmpty) continue;
        final lpC = cand.value;
        if (ch == lastCh) {
          acc(pref, _negInf, st.pnb + lpC, st.lms);
          if (st.pb > _negInf) {
            var sl = 0.0;
            if (useLm) {
              final key = '$p3\u0000$p2\u0000$p1\u0000$ch';
              sl = lmCache.putIfAbsent(
                key,
                () => lm.stepLogp(p3, p2, p1, ch, digitNeutral: digitNeutral),
              );
            }
            acc('$pref$ch', _negInf, st.pb + lpC, st.lms + sl);
          }
        } else {
          var sl = 0.0;
          if (useLm) {
            final key = '$p3\u0000$p2\u0000$p1\u0000$ch';
            sl = lmCache.putIfAbsent(
              key,
              () => lm.stepLogp(p3, p2, p1, ch, digitNeutral: digitNeutral),
            );
          }
          acc('$pref$ch', _negInf, ptot + lpC, st.lms + sl);
        }
      }
    }

    final ranked = nextBeams.entries.toList()
      ..sort((a, b) {
        final sa = _logSumExp2(a.value.pb, a.value.pnb) +
            (useLm ? alphaLm * a.value.lms : 0.0);
        final sb = _logSumExp2(b.value.pb, b.value.pnb) +
            (useLm ? alphaLm * b.value.lms : 0.0);
        return sb.compareTo(sa);
      });
    beams = {
      for (final e in ranked.take(beamWidth)) e.key: e.value,
    };
  }

  var bestPref = '';
  var bestScore = _negInf;
  for (final e in beams.entries) {
    final s = _logSumExp2(e.value.pb, e.value.pnb) +
        (useLm ? alphaLm * e.value.lms : 0.0);
    if (s > bestScore) {
      bestScore = s;
      bestPref = e.key;
    }
  }
  return bestPref;
}

class DecodedLineResult {
  const DecodedLineResult({
    required this.greedy,
    required this.text,
    required this.routeUsed,
    required this.numericDisagreement,
  });

  final String greedy;
  final String text;
  final String routeUsed;
  final bool numericDisagreement;
}

final RegExp _numSpanRe = RegExp(r'[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?');

/// Decode logits with R1 short-greedy routing (`<= 2` Unicode code points -> greedy).
DecodedLineResult decodeWithR1Routing({
  required List<List<double>> logpTC,
  required Map<int, String> idToChar,
  required String mode, // 'charlm_r1' | 'greedy'
  required double alphaLm,
  CharNGramLMV2? lm,
  int beamWidth = 5,
  int topkT = 6,
  int shortGreedyMaxChars = 2,
}) {
  final greedyRaw = ctcGreedyDecode(logpTC, idToChar);
  final greedyCanon = canonicalizeOcrText(greedyRaw);
  if (mode == 'greedy' || lm == null) {
    return DecodedLineResult(
      greedy: greedyCanon,
      text: greedyCanon,
      routeUsed: 'greedy',
      numericDisagreement: false,
    );
  }
  if (greedyCanon.runes.length <= shortGreedyMaxChars) {
    return DecodedLineResult(
      greedy: greedyCanon,
      text: greedyCanon,
      routeUsed: 'r1_short_greedy',
      numericDisagreement: false,
    );
  }
  final beamRaw = ctcPrefixBeamDecode(
    logpTC,
    idToChar,
    lm: lm,
    alphaLm: alphaLm,
    beamWidth: beamWidth,
    topkT: topkT,
    digitNeutral: true,
  );
  final beamCanon = canonicalizeOcrText(beamRaw);
  final gNums = _numSpanRe.allMatches(greedyCanon).map((m) => m.group(0)).toList();
  final bNums = _numSpanRe.allMatches(beamCanon).map((m) => m.group(0)).toList();
  var disagree = gNums.length != bNums.length;
  if (!disagree) {
    for (var i = 0; i < gNums.length; i++) {
      if (gNums[i] != bNums[i]) {
        disagree = true;
        break;
      }
    }
  }
  return DecodedLineResult(
    greedy: greedyCanon,
    text: beamCanon,
    routeUsed: 'beam4g_dn',
    numericDisagreement: disagree,
  );
}

/// Exact `window_ranges` from `src/recognition/long_line_inference.py` for >512px lines.
List<List<int>> computeWindowRanges(
  int width, {
  int maxWidth = 512,
  int overlap = 100,
}) {
  if (width <= maxWidth) return [[0, width]];
  final step = maxWidth - overlap;
  final out = <List<int>>[];
  var left = 0;
  while (true) {
    final right = math.min(left + maxWidth, width);
    out.add([left, right]);
    if (right >= width) break;
    left += step;
    if (width - left < maxWidth) {
      left = width - maxWidth;
    }
  }
  return out;
}

/// Exact `owned_tokens` coordinate-based merger from `src/recognition/long_line_inference.py`.
List<int> mergeOwnedWindowTokens(
  List<List<int>> windowPaths,
  List<List<int>> ranges,
) {
  final merged = <int>[];
  var prevTok = 0;
  for (var wIdx = 0; wIdx < windowPaths.length; wIdx++) {
    final path = windowPaths[wIdx];
    final left = ranges[wIdx][0];
    final ownStart = wIdx == 0 ? left : (ranges[wIdx - 1][1] + left) ~/ 2;
    final ownEnd = wIdx + 1 == ranges.length
        ? ranges[wIdx][1]
        : (ranges[wIdx][1] + ranges[wIdx + 1][0]) ~/ 2;
    for (var t = 0; t < path.length; t++) {
      final centerX = left + t * 4 + 2;
      if (centerX < ownStart || centerX >= ownEnd) continue;
      final tok = path[t];
      if (tok != 0 && tok != prevTok) {
        merged.add(tok);
      }
      prevTok = tok;
    }
  }
  return merged;
}
