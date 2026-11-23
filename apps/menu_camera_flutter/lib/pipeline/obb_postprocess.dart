/// Shared Stage 4 V5 YOLO11n-OBB 1024 ProbIoU Rotated NMS & Inverse Letterbox.
///
/// Exact Dart port of `decode_and_rotated_nms_obb` in `src/stage4/coreml_export.py`.
/// Preserves class mapping (`0: dish_horizontal`, `1: dish_vertical`), radian angles,
/// and rotated ProbIoU suppression (never axis-aligned Vision NMS).
library;

import 'dart:math' as math;
import 'dart:typed_data';

const Map<int, String> kDetectorClassMap = {
  0: 'dish_horizontal',
  1: 'dish_vertical',
};

class ObbLetterboxMeta {
  const ObbLetterboxMeta({
    required this.gain,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  final double gain;
  final double padX;
  final double padY;
  final double srcWidth;
  final double srcHeight;
}

class ObbDetection {
  const ObbDetection({
    required this.id,
    required this.classId,
    required this.label,
    required this.confidence,
    required this.xywhrSrc,
    required this.points,
  });

  final String id;
  final int classId;
  final String label;
  final double confidence;
  final List<double> xywhrSrc; // [cx, cy, w, h, angle_rad]
  final List<List<double>> points; // 4 clockwise corners in source pixels

  int get orientationFlag => classId == 1 ? 1 : 0;
  String get orientationLabel => classId == 1 ? 'vertical' : 'horizontal';
}

List<List<double>> xywhrToCorners(
  double cx,
  double cy,
  double w,
  double h,
  double angleRad,
) {
  final cosA = math.cos(angleRad);
  final sinA = math.sin(angleRad);
  final v1x = (w / 2.0) * cosA;
  final v1y = (w / 2.0) * sinA;
  final v2x = -(h / 2.0) * sinA;
  final v2y = (h / 2.0) * cosA;
  return [
    [cx + v1x + v2x, cy + v1y + v2y],
    [cx + v1x - v2x, cy + v1y - v2y],
    [cx - v1x - v2x, cy - v1y - v2y],
    [cx - v1x + v2x, cy - v1y + v2y],
  ];
}

double probIouPair(List<double> b1, List<double> b2, {double eps = 1e-7}) {
  final x1 = b1[0], y1 = b1[1], w1 = b1[2], h1 = b1[3], r1 = b1[4];
  final x2 = b2[0], y2 = b2[1], w2 = b2[2], h2 = b2[3], r2 = b2[4];

  final a1 = (w1 * w1) / 12.0;
  final bh1 = (h1 * h1) / 12.0;
  final c1 = math.cos(r1), s1 = math.sin(r1);
  final covA1 = a1 * c1 * c1 + bh1 * s1 * s1;
  final covB1 = a1 * s1 * s1 + bh1 * c1 * c1;
  final covC1 = (a1 - bh1) * c1 * s1;

  final a2 = (w2 * w2) / 12.0;
  final bh2 = (h2 * h2) / 12.0;
  final c2 = math.cos(r2), s2 = math.sin(r2);
  final covA2 = a2 * c2 * c2 + bh2 * s2 * s2;
  final covB2 = a2 * s2 * s2 + bh2 * c2 * c2;
  final covC2 = (a2 - bh2) * c2 * s2;

  final covA = covA1 + covA2;
  final covB = covB1 + covB2;
  final covC = covC1 + covC2;
  final denom = covA * covB - covC * covC + eps;

  final dx = x1 - x2;
  final dy = y1 - y2;
  final t1 = ((covA * dy * dy + covB * dx * dx) / denom) * 0.25;
  final t2 = ((covC * (-dx) * dy) / denom) * 0.5;
  final det1 = math.max(covA1 * covB1 - covC1 * covC1, 0.0);
  final det2 = math.max(covA2 * covB2 - covC2 * covC2, 0.0);
  final t3 = math.log(math.max(denom / (4.0 * math.sqrt(det1 * det2) + eps), 1.0)) * 0.5;
  final bd = (t1 + t2 + t3).clamp(eps, 100.0);
  final hd = math.sqrt(math.max(1.0 - math.exp(-bd), eps));
  return 1.0 - hd;
}

/// Decode raw `[7, N]` (channel-first flattened Float32List of length `7 * numAnchors`)
/// V5 YOLO11n-OBB output with class-aware ProbIoU rotated NMS and inverse letterbox.
List<ObbDetection> decodeAndRotatedNmsObb(
  Float32List rawObbChw,
  int numAnchors,
  ObbLetterboxMeta meta, {
  double confidence = 0.35,
  double iouThreshold = 0.70,
  int maxDet = 300,
}) {
  final cands = <({List<double> box, double conf, int clsId})>[];
  for (var i = 0; i < numAnchors; i++) {
    final p0 = rawObbChw[4 * numAnchors + i];
    final p1 = rawObbChw[5 * numAnchors + i];
    final clsId = p1 > p0 ? 1 : 0;
    final conf = clsId == 1 ? p1 : p0;
    if (conf < confidence) continue;
    final cx = rawObbChw[0 * numAnchors + i];
    final cy = rawObbChw[1 * numAnchors + i];
    final w = rawObbChw[2 * numAnchors + i];
    final h = rawObbChw[3 * numAnchors + i];
    final ang = rawObbChw[6 * numAnchors + i];
    cands.add((box: [cx, cy, w, h, ang], conf: conf, clsId: clsId));
  }

  cands.sort((a, b) => b.conf.compareTo(a.conf));
  final selected = <({List<double> box, double conf, int clsId})>[];
  for (final c in cands) {
    if (selected.length >= maxDet) break;
    var suppressed = false;
    for (final s in selected) {
      if (s.clsId != c.clsId) continue;
      if (probIouPair(c.box, s.box) > iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) selected.add(c);
  }

  final out = <ObbDetection>[];
  for (var idx = 0; idx < selected.length; idx++) {
    final s = selected[idx];
    final cxSrc = (s.box[0] - meta.padX) / meta.gain;
    final cySrc = (s.box[1] - meta.padY) / meta.gain;
    final wSrc = s.box[2] / meta.gain;
    final hSrc = s.box[3] / meta.gain;
    final ang = s.box[4];
    final pts = xywhrToCorners(cxSrc, cySrc, wSrc, hSrc, ang)
        .map((p) => [
              p[0].clamp(0.0, meta.srcWidth),
              p[1].clamp(0.0, meta.srcHeight),
            ])
        .toList();
    out.add(ObbDetection(
      id: 'raw_${idx.toString().padLeft(3, '0')}',
      classId: s.clsId,
      label: kDetectorClassMap[s.clsId] ?? 'dish_horizontal',
      confidence: s.conf,
      xywhrSrc: [cxSrc, cySrc, wSrc, hSrc, ang],
      points: pts,
    ));
  }
  return out;
}
