/// Native iOS AVFoundation camera controller backed by CoreMLMenuPlugin FlutterTexture.
/// Captures exact preview stream buffer pixels on shutter; zero live YOLO/OCR inference.
library;

import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';

import 'webcam_capture.dart';

class IosNativeCameraController implements WebcamController {
  IosNativeCameraController({MethodChannel? channel})
      : _channel = channel ??
            const MethodChannel('com.menucamera.stage4/coreml_backend');

  final MethodChannel _channel;
  int _textureId = -1;
  int _streamWidth = 0;
  int _streamHeight = 0;
  bool _ready = false;
  String? _error;

  @override
  bool get isSupported => true;

  @override
  bool get isReady => _ready;

  @override
  String? get errorMessage => _error;

  @override
  int get streamWidth => _streamWidth;

  @override
  int get streamHeight => _streamHeight;

  @override
  Future<void> start() async {
    try {
      final res =
          await _channel.invokeMapMethod<String, dynamic>('startCameraPreview');
      if (res == null) {
        _error = 'startCameraPreview returned null';
        _ready = false;
        return;
      }
      _textureId = (res['texture_id'] as num?)?.toInt() ?? -1;
      _streamWidth = (res['stream_width'] as num?)?.toInt() ?? 0;
      _streamHeight = (res['stream_height'] as num?)?.toInt() ?? 0;
      _ready = _textureId >= 0;
      _error = _ready ? null : 'Camera texture registration failed';
    } catch (e) {
      _error = 'iOS camera start failed: $e';
      _ready = false;
    }
  }

  @override
  Widget buildPreview() {
    if (!_ready || _textureId < 0) {
      return Center(
        child: Text(
          _error ?? 'Initializing iOS AVFoundation preview...',
          style: const TextStyle(color: Colors.white70),
        ),
      );
    }
    return Texture(textureId: _textureId);
  }

  @override
  Future<WebcamFrame> captureExactStreamFrame() async {
    final res = await _channel
        .invokeMapMethod<String, dynamic>('capturePreviewStreamFrame');
    if (res == null) {
      throw StateError('capturePreviewStreamFrame returned null');
    }
    final bytes = res['png_bytes'] as Uint8List?;
    final sw = (res['stream_width'] as num?)?.toInt() ?? 0;
    final sh = (res['stream_height'] as num?)?.toInt() ?? 0;
    if (bytes == null || bytes.isEmpty || sw <= 0 || sh <= 0) {
      throw StateError('Invalid preview stream frame captured (${sw}x$sh)');
    }
    _streamWidth = sw;
    _streamHeight = sh;
    return WebcamFrame(pngBytes: bytes, streamWidth: sw, streamHeight: sh);
  }

  @override
  void dispose() {
    _ready = false;
    _channel.invokeMethod<dynamic>('stopCameraPreview').catchError((_) {});
  }
}
