/// Stub webcam controller for non-web platforms without a native webcam plugin.
/// Explicitly reports unsupported camera rather than faking a camera.
library;

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:material_ui/material_ui.dart';
import 'ios_native_camera_controller.dart';
import 'webcam_capture.dart';

class _StubWebcamController implements WebcamController {
  @override
  bool get isSupported => false;

  @override
  bool get isReady => false;

  @override
  String? get errorMessage =>
      'Live webcam capture is unavailable in this runtime. Use Import Menu Image below.';

  @override
  int get streamWidth => 0;

  @override
  int get streamHeight => 0;

  @override
  Future<void> start() async {}

  @override
  Widget buildPreview() => const Center(
        child: Text(
          'Camera preview unavailable on this runtime.\nUse Import Menu Image to run full-page inference.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.white70),
        ),
      );

  @override
  Future<WebcamFrame> captureExactStreamFrame() {
    throw UnsupportedError('Camera not available on this runtime');
  }

  @override
  void dispose() {}
}

WebcamController createPlatformWebcamController() {
  if (defaultTargetPlatform == TargetPlatform.iOS) {
    return IosNativeCameraController();
  }
  return _StubWebcamController();
}
