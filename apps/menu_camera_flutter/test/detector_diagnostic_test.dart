import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:menu_camera_flutter/models/contracts.dart';
import 'package:menu_camera_flutter/pipeline/lexicon_lookup.dart';
import 'package:menu_camera_flutter/screens/capture_home_screen.dart';
import 'package:menu_camera_flutter/screens/detector_diagnostic_screen.dart';
import 'package:menu_camera_flutter/services/menu_backend.dart';
import 'package:menu_camera_flutter/services/settings_store.dart';
import 'package:menu_camera_flutter/services/webcam_capture.dart';

final Uint8List _tinyPng = Uint8List.fromList(base64Decode(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
));
final String _tinyPngB64 = base64Encode(_tinyPng);

class _StubIosBridge implements IosCoreMLNativeBridge {
  @override
  Future<Map<String, dynamic>> initialize() async => {
        'status': 'ready',
        'backend_id': 'ios_coreml_native',
        'device_name': 'iPhone 15',
        'bundle_id': 'stage4_step1000_raw_2105_v5_lmv2_lexv3_20261123',
      };

  @override
  Future<Map<String, dynamic>> runDetectorOBB1024(Uint8List imageBytes) async =>
      throw StateError('Simulated Core ML detector failure');

  @override
  Future<Map<String, dynamic>> runRecognizerCrop({
    required Uint8List imageBytes,
    required List<List<double>> polygon,
    required int orientationFlag,
  }) async =>
      const {};

  @override
  Future<Map<String, TranslationRecord>> lookupSqlTerms(
          Set<String> terms) async =>
      const {};

  @override
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase) async =>
      null;

  @override
  Future<Uint8List> loadBundledAssetBytes(String filename) async =>
      Uint8List.fromList(utf8.encode('{}'));
}

class _FakeFailingWebcam implements WebcamController {
  @override
  bool get isSupported => true;
  @override
  bool get isReady => true;
  @override
  String? get errorMessage => null;
  @override
  int get streamWidth => 1024;
  @override
  int get streamHeight => 1024;
  @override
  Future<void> start() async {}
  @override
  Widget buildPreview() => const ColoredBox(color: Colors.black);
  @override
  Future<WebcamFrame> captureExactStreamFrame() async => WebcamFrame(
        pngBytes: _tinyPng,
        streamWidth: 1024,
        streamHeight: 1024,
      );
  @override
  void dispose() {}
}

class _FailingRunBackend implements MenuBackend {
  @override
  String get backendLabel => 'Failing Test Backend';

  @override
  Future<Map<String, dynamic>> initialize() async => {'status': 'ok'};

  @override
  Future<PageRun> startCapture({
    required Uint8List imageBytes,
    required String filename,
    required int streamWidth,
    required int streamHeight,
    required String captureSource,
    required AppSettings settings,
  }) async =>
      PageRun(
        runId: 'fail_run_1',
        captureId: 'cap_fail',
        configId: 'cfg_1',
        backendId: 'ios_coreml_native',
        deviceName: 'iPhone 15',
        precision: 'fp32',
        bundleId: 'test_bundle',
        status: 'failed',
        stage: 'detecting',
        cacheStatus: 'failed',
        coldOrWarm: 'warm',
        isReprocess: false,
        capture: const CapturedPageMeta(
          captureId: 'cap_fail',
          filename: 'shutter.png',
          captureSource: 'ios_avfoundation_stream_buffer',
          streamWidth: 1024,
          streamHeight: 1024,
          savedWidth: 1024,
          savedHeight: 1024,
          streamSavedParity: true,
          imageSha256: 'abc',
        ),
        counts: const CropCounts(
          rawDetections: 0,
          postprocessedLines: 0,
          completedCrops: 0,
          failedCrops: 0,
          totalWindows: 0,
          cropsTruncated: false,
        ),
        timingsMs: const StageTimings(
          coldLoadMs: 0,
          decodeNormalizeMs: 0,
          persistMs: 0,
          detectMs: 12,
          postprocessMs: 0,
          recognizeMs: 0,
          decodeMs: 0,
          lookupMs: 0,
          firstResultMs: null,
          totalWallMs: 14,
          summedCropWorkMs: 0,
        ),
        crops: const [],
        error: 'Core ML OBB prediction failed: non-finite output tensor',
      );

  @override
  Future<PageRun> startReprocess({
    required String captureId,
    required AppSettings settings,
  }) =>
      throw UnimplementedError();

  @override
  Future<PageRun> fetchRun(String runId) => throw UnimplementedError();

  @override
  Future<void> cancelRun(String runId) async {}

  @override
  Future<Uint8List> fetchCaptureImage(String captureId) async => _tinyPng;
}

void main() {
  group('DetectorDiagnosticScreen UI & failure visibility', () {
    testWidgets(
        'No implicit auto-test; capture button disabled when null; bundled run renders passed report & previews',
        (tester) async {
      var calls = 0;
      double? gotConf;
      bool? gotCpuOnly;
      Uint8List? gotBytes;

      final backend = IosCoreMLBackend(bridge: _StubIosBridge());
      await tester.pumpWidget(MaterialApp(
        home: DetectorDiagnosticScreen(
          backend: backend,
          capturedImageBytes: null,
          diagnosticRunner: (bytes, {required confidence, bool cpuOnly = false}) async {
            calls++;
            gotBytes = bytes;
            gotConf = confidence;
            gotCpuOnly = cpuOnly;
            return {
              'status': 'passed',
              'stage': 'done',
              'error': null,
              'effective_confidence': confidence,
              'post_nms_count': 15,
              'expected_reference_boxes': 15,
              'source_preview_png_b64': _tinyPngB64,
              'detector_input_png_b64': _tinyPngB64,
              'decoded_boxes': [
                {
                  'points': [
                    [10, 10],
                    [200, 10],
                    [200, 50],
                    [10, 50]
                  ],
                  'confidence': 0.88,
                  'class_id': 0,
                }
              ],
              'native': {
                'compute_units': cpuOnly ? 'cpuOnly' : 'all',
                'output_shape': [1, 7, 21504],
                'output_dtype': 'float32',
                'finite_count': 150528,
                'nonfinite_count': 0,
                'max_class_confidence': 0.91,
              },
            };
          },
        ),
      ));
      await tester.pumpAndSettle();

      // Strictly NO implicit auto-test on open
      expect(calls, 0);
      expect(find.byKey(const Key('diag_status_card')), findsNothing);
      expect(find.text('0.35'), findsWidgets);

      // Capture button is disabled when capturedImageBytes is null
      final capBtn = tester.widget<OutlinedButton>(
          find.byKey(const Key('diag_run_capture_btn')));
      expect(capBtn.onPressed, isNull);

      // Toggle CPU-only ON and tap bundled test
      await tester.tap(find.byKey(const Key('diag_cpu_only_switch')));
      await tester.pump();
      await tester.runAsync(() async {
        await tester.tap(find.byKey(const Key('diag_run_bundled_btn')));
        await Future<void>.delayed(const Duration(milliseconds: 60));
      });
      await tester.pump();

      expect(calls, 1);
      expect(gotBytes, isNull);
      expect(gotConf, 0.35);
      expect(gotCpuOnly, isTrue);

      expect(find.byKey(const Key('diag_status_card')), findsOneWidget);
      expect(find.textContaining('Status: passed'), findsOneWidget);
      expect(find.textContaining('Post-NMS Boxes: 15 (expected 15)'),
          findsOneWidget);
      expect(find.byKey(const Key('diag_native_metrics_box')), findsOneWidget);
      expect(find.byKey(const Key('diag_source_preview_box')), findsOneWidget);
      expect(find.byKey(const Key('diag_detector_input_preview_box')),
          findsOneWidget);
      expect(find.byKey(const Key('diag_copy_json_btn')), findsOneWidget);
      expect(find.byKey(const Key('diag_save_json_btn')), findsOneWidget);
    });

    testWidgets(
        'Failed diagnostic report displays error prominently and keeps partial native metrics',
        (tester) async {
      final backend = IosCoreMLBackend(bridge: _StubIosBridge());
      await tester.pumpWidget(MaterialApp(
        home: DetectorDiagnosticScreen(
          backend: backend,
          capturedImageBytes: _tinyPng,
          diagnosticRunner: (bytes, {required confidence, bool cpuOnly = false}) async {
            return {
              'status': 'failed',
              'stage': 'postprocess',
              'error': 'Output tensor contained 150528 non-finite values',
              'effective_confidence': confidence,
              'post_nms_count': 0,
              'expected_reference_boxes': null,
              'decoded_boxes': const [],
              'native': {
                'compute_units': 'all',
                'nonfinite_count': 150528,
                'max_class_confidence': 0.0,
              },
            };
          },
        ),
      ));
      await tester.pumpAndSettle();

      final capBtn = tester.widget<OutlinedButton>(
          find.byKey(const Key('diag_run_capture_btn')));
      expect(capBtn.onPressed, isNotNull);

      await tester.tap(find.byKey(const Key('diag_run_capture_btn')));
      await tester.pumpAndSettle();

      expect(find.textContaining('Status: failed'), findsOneWidget);
      expect(find.textContaining('Stage: postprocess'), findsOneWidget);
      expect(find.byKey(const Key('diag_error_text')), findsOneWidget);
      expect(
          find.textContaining('Output tensor contained 150528 non-finite values'),
          findsWidgets);
      expect(find.byKey(const Key('diag_native_metrics_box')), findsOneWidget);
    });
  });

  group('CaptureHomeScreen iOS diagnostics entry & failed run banner', () {
    testWidgets(
        'iOS backend shows diagnostics button & non-Desktop banner; opens diagnostic screen',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      final iosBackend = IosCoreMLBackend(bridge: _StubIosBridge());
      await tester.pumpWidget(MaterialApp(
        home: CaptureHomeScreen(
          backend: iosBackend,
          settingsStore: SettingsStore(),
          webcamController: _FakeFailingWebcam(),
        ),
      ));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('diagnostics_btn')), findsOneWidget);
      expect(find.textContaining('Desktop Demo'), findsNothing);
      expect(find.textContaining('iOS On-Device Core ML'), findsOneWidget);

      await tester.tap(find.byKey(const Key('diagnostics_btn')));
      await tester.pumpAndSettle();
      expect(find.text('iOS Detector Diagnostics'), findsOneWidget);
    });

    testWidgets('Terminal failed PageRun shows prominent run.error banner',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      await tester.pumpWidget(MaterialApp(
        home: CaptureHomeScreen(
          backend: _FailingRunBackend(),
          settingsStore: SettingsStore(),
          webcamController: _FakeFailingWebcam(),
        ),
      ));
      await tester.pumpAndSettle();

      // Non-iOS backend does NOT show iOS diagnostics button
      expect(find.byKey(const Key('diagnostics_btn')), findsNothing);

      await tester.tap(find.byKey(const Key('shutter_btn')));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('run_failed_error_banner')), findsOneWidget);
      expect(
          find.textContaining(
              'Core ML OBB prediction failed: non-finite output tensor'),
          findsOneWidget);
      expect(find.textContaining('detecting'), findsWidgets);
    });
  });
}
