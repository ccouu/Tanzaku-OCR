/// Pure Dart geometry transforms for source-pixel <-> viewport polygon mapping and hit testing.
/// Supports BoxFit.contain (still image with letterbox) and BoxFit.cover (aspect-fill preview).
library;

import 'dart:math' as math;
import 'dart:ui';

class ImageViewportTransform {
  const ImageViewportTransform({
    required this.sourceSize,
    required this.viewportSize,
    required this.scale,
    required this.offset,
  });

  final Size sourceSize;
  final Size viewportSize;
  final double scale;
  final Offset offset;

  /// Compute exact transform for BoxFit.contain (letterboxed still view).
  factory ImageViewportTransform.contain({
    required Size sourceSize,
    required Size viewportSize,
  }) {
    if (sourceSize.width <= 0 || sourceSize.height <= 0) {
      return ImageViewportTransform(
        sourceSize: sourceSize,
        viewportSize: viewportSize,
        scale: 1.0,
        offset: Offset.zero,
      );
    }
    final sx = viewportSize.width / sourceSize.width;
    final sy = viewportSize.height / sourceSize.height;
    final s = math.min(sx, sy);
    final dx = (viewportSize.width - sourceSize.width * s) / 2.0;
    final dy = (viewportSize.height - sourceSize.height * s) / 2.0;
    return ImageViewportTransform(
      sourceSize: sourceSize,
      viewportSize: viewportSize,
      scale: s,
      offset: Offset(dx, dy),
    );
  }

  /// Compute exact transform for BoxFit.cover (aspect-fill camera preview).
  factory ImageViewportTransform.cover({
    required Size sourceSize,
    required Size viewportSize,
  }) {
    if (sourceSize.width <= 0 || sourceSize.height <= 0) {
      return ImageViewportTransform(
        sourceSize: sourceSize,
        viewportSize: viewportSize,
        scale: 1.0,
        offset: Offset.zero,
      );
    }
    final sx = viewportSize.width / sourceSize.width;
    final sy = viewportSize.height / sourceSize.height;
    final s = math.max(sx, sy);
    final dx = (viewportSize.width - sourceSize.width * s) / 2.0;
    final dy = (viewportSize.height - sourceSize.height * s) / 2.0;
    return ImageViewportTransform(
      sourceSize: sourceSize,
      viewportSize: viewportSize,
      scale: s,
      offset: Offset(dx, dy),
    );
  }

  Offset sourceToViewport(Offset src) =>
      Offset(src.dx * scale + offset.dx, src.dy * scale + offset.dy);

  Offset viewportToSource(Offset vp) =>
      Offset((vp.dx - offset.dx) / scale, (vp.dy - offset.dy) / scale);

  List<Offset> polygonToViewport(List<List<double>> poly) =>
      poly.map((pt) => sourceToViewport(Offset(pt[0], pt[1]))).toList();
}

/// Ray-casting point-in-polygon check in source-pixel coordinates.
bool pointInPolygon(Offset point, List<List<double>> polygon) {
  if (polygon.length < 3) return false;
  var inside = false;
  for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    final xi = polygon[i][0];
    final yi = polygon[i][1];
    final xj = polygon[j][0];
    final yj = polygon[j][1];
    final intersects = ((yi > point.dy) != (yj > point.dy)) &&
        (point.dx < (xj - xi) * (point.dy - yi) / ((yj - yi) + 1e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

/// Verify ordered stream buffer pixel parity against saved still dimensions.
///
/// Requires exact (streamWidth == savedWidth && streamHeight == savedHeight),
/// OR allows swapped (streamWidth == savedHeight && streamHeight == savedWidth)
/// ONLY when a known 90/270-degree rotation or EXIF axis-swap orientation (5..8)
/// justifies the swap. Rejects misreported stream dimensions.
bool verifyStreamSavedParity({
  required int streamWidth,
  required int streamHeight,
  required int savedWidth,
  required int savedHeight,
  int rotationDegrees = 0,
  int? exifOrientation,
}) {
  if (streamWidth <= 0 || streamHeight <= 0 || savedWidth <= 0 || savedHeight <= 0) {
    return false;
  }
  final rot = ((rotationDegrees % 360) + 360) % 360;
  if (rot != 0 && rot != 90 && rot != 180 && rot != 270) {
    return false;
  }
  final rotationSwapsAxes = (rot == 90 || rot == 270) ||
      (exifOrientation != null && const {5, 6, 7, 8}.contains(exifOrientation));

  if (streamWidth == savedWidth && streamHeight == savedHeight) {
    if (rotationSwapsAxes && streamWidth != streamHeight) {
      return false;
    }
    return true;
  }
  if (streamWidth == savedHeight && streamHeight == savedWidth) {
    return rotationSwapsAxes;
  }
  return false;
}
