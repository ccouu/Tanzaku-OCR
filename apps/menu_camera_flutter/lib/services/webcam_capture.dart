/// Cross-platform webcam stream preview & exact stream-buffer shutter capture contract.
library;

import 'dart:typed_data';
import 'package:flutter/widgets.dart';

import 'webcam_capture_stub.dart'
    if (dart.library.js_interop) 'webcam_capture_web.dart';

class WebcamFrame {
  const WebcamFrame({
    required this.pngBytes,
    required this.streamWidth,
    required this.streamHeight,
  });

  final Uint8List pngBytes;
  final int streamWidth;
  final int streamHeight;
}

abstract class WebcamController {
  bool get isSupported;
  bool get isReady;
  String? get errorMessage;
  int get streamWidth;
  int get streamHeight;

  Future<void> start();
  Widget buildPreview();
  Future<WebcamFrame> captureExactStreamFrame();
  void dispose();
}

WebcamController createWebcamController() => createPlatformWebcamController();
