import 'dart:convert';
import 'dart:typed_data';
import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:menu_camera_flutter/geometry/overlay_geometry.dart';
import 'package:menu_camera_flutter/main.dart';
import 'package:menu_camera_flutter/models/contracts.dart';
import 'package:menu_camera_flutter/screens/capture_home_screen.dart';
import 'package:menu_camera_flutter/services/menu_backend.dart';
import 'package:menu_camera_flutter/services/settings_store.dart';
import 'package:menu_camera_flutter/services/webcam_capture.dart';

// Valid 1x1 PNG bytes for deterministic widget tests
final Uint8List _tinyPng = Uint8List.fromList(base64Decode(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
));

class _FakeWebcam implements WebcamController {
  _FakeWebcam({this.captureDelay = Duration.zero});

  final Duration captureDelay;

  @override
  bool get isSupported => true;
  @override
  bool get isReady => true;
  @override
  String? get errorMessage => null;
  @override
  int get streamWidth => 1280;
  @override
  int get streamHeight => 720;
  @override
  Future<void> start() async {}
  @override
  Widget buildPreview() => const ColoredBox(color: Colors.black);
  @override
  Future<WebcamFrame> captureExactStreamFrame() async {
    if (captureDelay > Duration.zero) {
      await Future<void>.delayed(captureDelay);
    }
    return WebcamFrame(
      pngBytes: _tinyPng,
      streamWidth: 1280,
      streamHeight: 720,
    );
  }

  @override
  void dispose() {}
}

class _FakeBackend implements MenuBackend {
  _FakeBackend({this.uploadDelay = Duration.zero});

  final Duration uploadDelay;
  int captureCalls = 0;
  int reprocessCalls = 0;
  int cancelCalls = 0;
  String? lastCancelledRunId;

  PageRun _makeRun(String runId, String status, String cacheStatus) => PageRun(
        runId: runId,
        captureId: 'cap_001',
        configId: 'cfg_001',
        backendId: 'fake_backend',
        deviceName: 'FakeGPU',
        precision: 'fp32',
        bundleId: 'test_bundle',
        status: status,
        stage: status == 'completed' ? 'done' : 'recognizing',
        cacheStatus: cacheStatus,
        reusedCachedLogits: cacheStatus == 'reused_cached_logits',
        reusedCachedDetections: cacheStatus == 'reused_cached_logits',
        coldOrWarm: 'warm',
        isReprocess: cacheStatus == 'reused_cached_logits',
        capture: const CapturedPageMeta(
          captureId: 'cap_001',
          filename: 'shutter.png',
          captureSource: 'webcam_stream_buffer',
          streamWidth: 1280,
          streamHeight: 720,
          savedWidth: 1280,
          savedHeight: 720,
          streamSavedParity: true,
          streamSavedParityMode: 'exact',
          imageSha256: 'abc',
        ),
        counts: const CropCounts(
          rawDetections: 1,
          postprocessedLines: 1,
          completedCrops: 1,
          failedCrops: 0,
          totalWindows: 1,
          cropsTruncated: false,
        ),
        timingsMs: const StageTimings(
          coldLoadMs: 0,
          decodeNormalizeMs: 4,
          persistMs: 5,
          requestIngestMs: 9,
          detectMs: 20,
          postprocessMs: 3,
          recognizeMs: 18,
          decodeMs: 6,
          lookupMs: 2,
          firstResultMs: 49,
          serverPipelineWallMs: 49,
          serverRequestTotalMs: 58,
          totalWallMs: 58,
          summedCropWorkMs: 26,
        ),
        crops: [
          CropResult(
            cropId: 'crop_01',
            cropIndex: 1,
            status: 'completed',
            confidence: 0.92,
            orientation: 'horizontal',
            polygon: const [
              [0, 0],
              [1280, 0],
              [1280, 720],
              [0, 720],
            ],
            windowCount: 1,
            multiWindowFallback: false,
            cropPreviewPngB64: base64Encode(_tinyPng),
            greedyOcr: '季節の先付と前菜',
            decodedOcr: '季節の先付と前菜',
            decoderRoute: 'beam4g_dn',
            numericDisagreement: false,
            translation: const CropTranslation(
              primaryEnglish: 'Seasonal Starter (Sakizuke) and Appetizers',
              reviewedWholePhraseEn:
                  'Seasonal Starter (Sakizuke) and Appetizers',
              translationStatus: 'exact_db_english',
              components: [
                TranslationComponent(
                    text: '季節', kind: 'exact', english: 'seasonal'),
                TranslationComponent(text: 'の', kind: 'particle', english: ''),
                TranslationComponent(
                    text: '先付',
                    kind: 'exact',
                    english: 'seasonal course starter (sakizuke)'),
              ],
              suggestions: [],
            ),
            error: null,
          ),
        ],
        error: null,
      );

  @override
  String get backendLabel => 'Fake Reference Backend';
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
  }) async {
    if (uploadDelay > Duration.zero) {
      await Future<void>.delayed(uploadDelay);
    }
    captureCalls++;
    return _makeRun('run_$captureCalls', 'completed', 'fresh_full_inference');
  }

  @override
  Future<PageRun> startReprocess({
    required String captureId,
    required AppSettings settings,
  }) async {
    reprocessCalls++;
    return _makeRun(
        'reproc_$reprocessCalls', 'completed', 'reused_cached_logits');
  }

  @override
  Future<PageRun> fetchRun(String runId) async =>
      _makeRun(runId, 'completed', 'fresh_full_inference');

  @override
  Future<void> cancelRun(String runId) async {
    cancelCalls++;
    lastCancelledRunId = runId;
  }

  @override
  Future<Uint8List> fetchCaptureImage(String captureId) async => _tinyPng;
}

void main() {
  group('Geometry & stream-resolution parity', () {
    test('BoxFit.contain and BoxFit.cover round-trip source <-> viewport', () {
      final contain = ImageViewportTransform.contain(
        sourceSize: const Size(1280, 720),
        viewportSize: const Size(640, 640),
      );
      const srcPt = Offset(640, 360);
      final vp = contain.sourceToViewport(srcPt);
      expect(vp.dx, closeTo(320, 1e-6));
      expect(vp.dy, closeTo(320, 1e-6));
      final back = contain.viewportToSource(vp);
      expect(back.dx, closeTo(srcPt.dx, 1e-6));
      expect(back.dy, closeTo(srcPt.dy, 1e-6));

      final cover = ImageViewportTransform.cover(
        sourceSize: const Size(1280, 720),
        viewportSize: const Size(400, 800),
      );
      final vp2 = cover.sourceToViewport(srcPt);
      final back2 = cover.viewportToSource(vp2);
      expect(back2.dx, closeTo(srcPt.dx, 1e-6));
      expect(back2.dy, closeTo(srcPt.dy, 1e-6));
    });

    test('pointInPolygon handles rotated quads accurately', () {
      const diamond = [
        [50.0, 0.0],
        [100.0, 50.0],
        [50.0, 100.0],
        [0.0, 50.0],
      ];
      expect(pointInPolygon(const Offset(50, 50), diamond), isTrue);
      expect(pointInPolygon(const Offset(5, 5), diamond), isFalse);
    });

    test('verifyStreamSavedParity enforces ordered WxH and justified 90/270 rotation only', () {
      // Exact match passes
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 1280,
              savedHeight: 720),
          isTrue);
      // Swapped WITHOUT justified 90/270 rotation MUST fail
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 720,
              savedHeight: 1280),
          isFalse);
      // Swapped WITH justified 90 or 270 rotation passes
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 720,
              savedHeight: 1280,
              rotationDegrees: 90),
          isTrue);
      // Swapped WITH EXIF orientation 6 passes
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 720,
              savedHeight: 1280,
              exifOrientation: 6),
          isTrue);
      // Rotation=90 claiming unswapped non-square dims MUST fail
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 1280,
              savedHeight: 720,
              rotationDegrees: 90),
          isFalse);
      // Mismatched dimensions fail
      expect(
          verifyStreamSavedParity(
              streamWidth: 1280,
              streamHeight: 720,
              savedWidth: 640,
              savedHeight: 360),
          isFalse);
    });
  });

  group('Settings persistence & reset', () {
    test('SettingsStore saves, loads, and resets defaults', () async {
      SharedPreferences.setMockInitialValues({});
      final store = SettingsStore();
      expect(await store.load(), AppSettings.defaults);
      const custom = AppSettings(
          yoloConfidence: 0.50, charLmAlpha: 0.15, decodeMode: 'greedy');
      await store.save(custom);
      expect(await store.load(), custom);
      expect(await store.reset(), AppSettings.defaults);
    });
  });

  group('Capture-first UI flow', () {
    testWidgets(
        'Shutter -> results -> tap cached popup -> dismiss -> settings -> reprocess -> retake',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      final backend = _FakeBackend();
      final webcam = _FakeWebcam();
      await tester.pumpWidget(MaterialApp(
        home: Builder(
          builder: (_) => MenuCameraApp(
            backend: backend,
            settingsStore: SettingsStore(),
          ),
        ),
      ));
      // Rebuild with injected webcam controller directly on CaptureHomeScreen
      await tester.pumpWidget(MaterialApp(
        home: CaptureHomeScreen(
          backend: backend,
          settingsStore: SettingsStore(),
          webcamController: webcam,
        ),
      ));
      await tester.pumpAndSettle();

      // Zero live inference before shutter!
      expect(backend.captureCalls, 0);
      expect(find.byKey(const Key('shutter_btn')), findsOneWidget);
      expect(find.byKey(const Key('import_image_btn')), findsOneWidget);

      // Press shutter -> triggers single whole-page job
      await tester.tap(find.byKey(const Key('shutter_btn')));
      await tester.pumpAndSettle();
      expect(backend.captureCalls, 1);
      expect(find.byKey(const Key('client_end_to_end_text')), findsOneWidget);
      expect(find.byKey(const Key('page_total_time_text')), findsOneWidget);
      expect(find.text('Cache: Fresh Full Inference'), findsOneWidget);

      // Tap center of still image -> opens cached popup without re-running OCR
      await tester.tap(find.byKey(const Key('still_image_canvas')));
      await tester.pumpAndSettle();
      expect(find.text('季節の先付と前菜'), findsOneWidget);
      expect(find.text('Seasonal Starter (Sakizuke) and Appetizers'),
          findsOneWidget);
      expect(backend.captureCalls, 1);

      // Dismiss popup returns to still page
      await tester.tap(find.byKey(const Key('popup_dismiss_btn')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('retake_btn')), findsOneWidget);

      // Change settings -> explicit Reprocess button appears -> click Reprocess
      await tester.tap(find.byKey(const Key('settings_btn')));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Canonical Greedy'));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('settings_apply_btn')));
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('reprocess_btn')), findsOneWidget);
      await tester.tap(find.byKey(const Key('reprocess_btn')));
      await tester.pumpAndSettle();
      expect(backend.reprocessCalls, 1);
      expect(find.text('Cache: Reused Cached Logits'), findsOneWidget);
      expect(find.textContaining('Client Reprocess→Result:'), findsOneWidget);

      // Retake returns to preview
      await tester.tap(find.byKey(const Key('retake_btn')));
      await tester.pumpAndSettle();
      expect(find.byKey(const Key('shutter_btn')), findsOneWidget);
    });

    testWidgets(
        'Frontend monotonic Stopwatch accounts for delayed capture + upload separately from server wall',
        (tester) async {
      SharedPreferences.setMockInitialValues({});
      final backend =
          _FakeBackend(uploadDelay: const Duration(milliseconds: 60));
      final webcam =
          _FakeWebcam(captureDelay: const Duration(milliseconds: 60));

      await tester.runAsync(() async {
        await tester.pumpWidget(MaterialApp(
          home: CaptureHomeScreen(
            backend: backend,
            settingsStore: SettingsStore(),
            webcamController: webcam,
          ),
        ));
        await Future<void>.delayed(const Duration(milliseconds: 20));
        await tester.pump();

        await tester.tap(find.byKey(const Key('shutter_btn')));
        // Wait for real delayed capture (60ms) + delayed upload (60ms)
        await Future<void>.delayed(const Duration(milliseconds: 180));
        await tester.pump();
      });

      // Expand timing details to inspect client monotonic breakdown
      await tester.tap(find.byKey(const Key('toggle_timings_btn')));
      await tester.pump();

      final detailsWidget = tester.widget<Text>(
          find.byKey(const Key('client_timing_details_text')));
      final text = detailsWidget.data ?? '';
      expect(text, contains('action=Shutter→Result'));
      final capMatch = RegExp(r'capture/encode=([0-9.]+)').firstMatch(text);
      final e2eMatch = RegExp(r'endToEnd=([0-9.]+)').firstMatch(text);
      expect(capMatch, isNotNull);
      expect(e2eMatch, isNotNull);
      final capMs = double.parse(capMatch!.group(1)!);
      final e2eMs = double.parse(e2eMatch!.group(1)!);
      // Real monotonic stopwatch must account for both capture (60ms) and upload (60ms)
      expect(capMs, greaterThanOrEqualTo(45.0));
      expect(e2eMs, greaterThanOrEqualTo(100.0));
      expect(find.textContaining('Server Req:'), findsOneWidget);
      expect(find.textContaining('Pipeline:'), findsOneWidget);
      expect(find.text('Cache: Fresh Full Inference'), findsOneWidget);
    });
  });
}
