/// Flutter Web HTML5 webcam preview + exact stream-buffer shutter capture
/// using `video.videoWidth` x `video.videoHeight` (NOT CSS viewport dimensions).
library;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';
import 'dart:ui_web' as ui_web;

import 'package:material_ui/material_ui.dart';
import 'package:web/web.dart' as web;

import 'webcam_capture.dart';

int _viewSeq = 0;

class _WebWebcamController implements WebcamController {
  _WebWebcamController() : _viewType = 'stage4-webcam-view-${_viewSeq++}';

  final String _viewType;
  web.HTMLVideoElement? _video;
  web.MediaStream? _stream;
  bool _ready = false;
  bool _registered = false;
  String? _error;

  @override
  bool get isSupported => true;

  @override
  bool get isReady => _ready && (_video?.videoWidth ?? 0) > 0;

  @override
  String? get errorMessage => _error;

  @override
  int get streamWidth => _video?.videoWidth ?? 0;

  @override
  int get streamHeight => _video?.videoHeight ?? 0;

  @override
  Future<void> start() async {
    _error = null;
    try {
      final video = web.HTMLVideoElement()
        ..autoplay = true
        ..muted = true
        ..playsInline = true
        ..style.width = '100%'
        ..style.height = '100%'
        ..style.objectFit = 'cover';
      _video = video;

      if (!_registered) {
        ui_web.platformViewRegistry.registerViewFactory(_viewType, (int _) => video);
        _registered = true;
      }

      final constraints = web.MediaStreamConstraints(
        video: {'facingMode': 'environment'}.jsify()!,
        audio: false.toJS,
      );
      final stream = await web.window.navigator.mediaDevices
          .getUserMedia(constraints)
          .toDart;
      _stream = stream;
      video.srcObject = stream;
      await video.play().toDart;

      // Wait briefly for videoWidth/videoHeight metadata to populate
      for (var i = 0; i < 20 && video.videoWidth == 0; i++) {
        await Future<void>.delayed(const Duration(milliseconds: 50));
      }
      _ready = video.videoWidth > 0 && video.videoHeight > 0;
      if (!_ready) {
        _error = 'Camera started but reported 0x0 stream dimensions.';
      }
    } catch (e) {
      _ready = false;
      _error = 'Camera permission denied or no webcam found ($e). Use Import Menu Image below.';
    }
  }

  @override
  Widget buildPreview() {
    if (!_ready || _video == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _error ?? 'Initializing camera stream...',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white70, fontSize: 15),
          ),
        ),
      );
    }
    return HtmlElementView(viewType: _viewType);
  }

  @override
  Future<WebcamFrame> captureExactStreamFrame() async {
    final video = _video;
    if (video == null || video.videoWidth <= 0 || video.videoHeight <= 0) {
      throw StateError('Camera stream is not ready');
    }
    // Capture exact negotiated video stream dimensions, NEVER CSS widget size!
    final w = video.videoWidth;
    final h = video.videoHeight;
    final canvas = web.HTMLCanvasElement()
      ..width = w
      ..height = h;
    final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D;
    ctx.drawImage(video, 0, 0, w, h);
    final dataUrl = canvas.toDataURL('image/png');
    final comma = dataUrl.indexOf(',');
    final pngBytes = Uint8List.fromList(base64Decode(dataUrl.substring(comma + 1)));
    return WebcamFrame(pngBytes: pngBytes, streamWidth: w, streamHeight: h);
  }

  @override
  void dispose() {
    final tracks = _stream?.getTracks().toDart;
    if (tracks != null) {
      for (final t in tracks) {
        t.stop();
      }
    }
    _stream = null;
    _video = null;
    _ready = false;
  }
}

WebcamController createPlatformWebcamController() => _WebWebcamController();
