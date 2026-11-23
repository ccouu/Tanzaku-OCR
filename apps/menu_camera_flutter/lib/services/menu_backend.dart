/// Platform backend interface shared by DesktopReferenceBackend (localhost Python)
/// and future IosCoreMLBackend (Swift/Core ML plugin on iPhone 15).
library;

import 'dart:convert';

import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

import '../models/contracts.dart';
import 'ios_coreml_backend.dart';

export 'ios_coreml_backend.dart' show IosCoreMLBackend, IosCoreMLNativeBridge;

/// Select platform backend explicitly: iOS routes to native Core ML plugin (never localhost Python);
/// Windows/Web desktop routes to local Python reference backend.
MenuBackend createPlatformMenuBackend({
  String? desktopBaseUrl,
  http.Client? client,
}) {
  if (!kIsWeb && defaultTargetPlatform == TargetPlatform.iOS) {
    return IosCoreMLBackend();
  }
  return DesktopReferenceBackend(baseUrl: desktopBaseUrl, client: client);
}

abstract class MenuBackend {
  String get backendLabel;
  Future<Map<String, dynamic>> initialize();
  Future<PageRun> startCapture({
    required Uint8List imageBytes,
    required String filename,
    required int streamWidth,
    required int streamHeight,
    required String captureSource,
    required AppSettings settings,
  });
  Future<PageRun> startReprocess({
    required String captureId,
    required AppSettings settings,
  });
  Future<PageRun> fetchRun(String runId);
  Future<void> cancelRun(String runId);
  Future<Uint8List> fetchCaptureImage(String captureId);
}

class DesktopReferenceBackend implements MenuBackend {
  DesktopReferenceBackend({
    String? baseUrl,
    http.Client? client,
  })  : _baseUrl = baseUrl ??
            (kIsWeb ? Uri.base.origin : 'http://127.0.0.1:8790'),
        _client = client ?? http.Client();

  final String _baseUrl;
  final http.Client _client;
  String? _token;
  String _deviceName = 'Local Python Backend';
  String _bundleId = '';

  @override
  String get backendLabel =>
      'Desktop Python Reference ($_deviceName)${_bundleId.isNotEmpty ? ' • $_bundleId' : ''}';

  Map<String, String> get _authHeaders => {
        'X-Stage4-Token': ?_token,
      };

  @override
  Future<Map<String, dynamic>> initialize() async {
    final resp = await _client.get(Uri.parse('$_baseUrl/api/bootstrap'));
    if (resp.statusCode != 200) {
      throw Exception(
          'Desktop backend bootstrap failed (${resp.statusCode}): ${resp.body}');
    }
    final data = jsonDecode(resp.body) as Map<String, dynamic>;
    _token = data['session_token'] as String?;
    _deviceName = (data['device_name'] as String?) ?? 'Local Python';
    _bundleId = (data['bundle_id'] as String?) ?? '';
    return data;
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
    if (_token == null) await initialize();
    final resp = await _client.post(
      Uri.parse('$_baseUrl/api/capture'),
      headers: {
        ..._authHeaders,
        'Content-Type': 'application/octet-stream',
        'X-Filename': filename,
        'X-Stream-Width': streamWidth.toString(),
        'X-Stream-Height': streamHeight.toString(),
        'X-Capture-Source': captureSource,
        'X-Confidence': settings.yoloConfidence.toStringAsFixed(2),
        'X-Alpha-Lm': settings.charLmAlpha.toStringAsFixed(2),
        'X-Decode-Mode': settings.decodeMode,
      },
      body: imageBytes,
    );
    if (resp.statusCode != 202 && resp.statusCode != 200) {
      throw Exception('Capture failed (${resp.statusCode}): ${resp.body}');
    }
    return PageRun.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  @override
  Future<PageRun> startReprocess({
    required String captureId,
    required AppSettings settings,
  }) async {
    if (_token == null) await initialize();
    final resp = await _client.post(
      Uri.parse('$_baseUrl/api/reprocess'),
      headers: {
        ..._authHeaders,
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'capture_id': captureId,
        'confidence': settings.yoloConfidence,
        'alpha_lm': settings.charLmAlpha,
        'decode_mode': settings.decodeMode,
      }),
    );
    if (resp.statusCode != 202 && resp.statusCode != 200) {
      throw Exception('Reprocess failed (${resp.statusCode}): ${resp.body}');
    }
    return PageRun.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  @override
  Future<PageRun> fetchRun(String runId) async {
    if (_token == null) await initialize();
    final resp = await _client.get(
      Uri.parse('$_baseUrl/api/runs/$runId'),
      headers: _authHeaders,
    );
    if (resp.statusCode != 200) {
      throw Exception('Fetch run failed (${resp.statusCode}): ${resp.body}');
    }
    return PageRun.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  @override
  Future<void> cancelRun(String runId) async {
    if (_token == null) return;
    await _client.post(
      Uri.parse('$_baseUrl/api/runs/$runId/cancel'),
      headers: _authHeaders,
    );
  }

  @override
  Future<Uint8List> fetchCaptureImage(String captureId) async {
    if (_token == null) await initialize();
    final resp = await _client.get(
      Uri.parse('$_baseUrl/api/captures/$captureId/image'),
      headers: _authHeaders,
    );
    if (resp.statusCode != 200) {
      throw Exception('Fetch capture image failed (${resp.statusCode})');
    }
    return resp.bodyBytes;
  }
}
