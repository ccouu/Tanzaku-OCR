/// Main Capture-First Stage 4 Screen:
/// - Preview Mode: live webcam preview (exact videoWidth x videoHeight shutter snapshot)
///   + Import Menu Image button + Settings button. Zero continuous live YOLO/OCR.
/// - Result Mode: frozen still image with interactive tappable polygons, real-time
///   progress & Cancel, prominent whole-page wall time + expandable stage/cache timings,
///   cached JP+English popup on tap, explicit Reprocess on settings change, and Retake.
library;

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:material_ui/material_ui.dart';

import '../geometry/overlay_geometry.dart';
import '../models/contracts.dart';
import '../services/menu_backend.dart';
import '../services/settings_store.dart';
import '../services/webcam_capture.dart';
import '../widgets/crop_popup.dart';
import '../widgets/settings_sheet.dart';
import 'detector_diagnostic_screen.dart';

class CaptureHomeScreen extends StatefulWidget {
  const CaptureHomeScreen({
    super.key,
    required this.backend,
    required this.settingsStore,
    this.webcamController,
  });

  final MenuBackend backend;
  final SettingsStore settingsStore;
  final WebcamController? webcamController;

  @override
  State<CaptureHomeScreen> createState() => _CaptureHomeScreenState();
}

class _CaptureHomeScreenState extends State<CaptureHomeScreen> {
  late final WebcamController _webcam;
  AppSettings _settings = AppSettings.defaults;
  AppSettings? _runSettings;
  String _backendBanner = 'Connecting to local Python reference backend...';
  String? _errorBanner;

  // Still-image / active job state
  Uint8List? _stillImageBytes;
  Size? _stillImageSize;
  PageRun? _activeRun;
  String? _expectedRunId;
  String? _selectedCropId;
  bool _timingsExpanded = false;
  Timer? _pollTimer;

  // Frontend monotonic Stopwatch (independent of server clocks — zero cross-process subtraction)
  Stopwatch? _clientStopwatch;
  String _clientActionLabel = 'Shutter→Result';
  double? _clientCaptureEncodeMs;
  double? _clientUploadAckMs;
  double? _clientEndToEndMs;

  @override
  void initState() {
    super.initState();
    _backendBanner = widget.backend is IosCoreMLBackend
        ? 'Initializing iOS On-Device Core ML backend...'
        : 'Connecting to local Python reference backend...';
    _webcam = widget.webcamController ?? createWebcamController();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final loaded = await widget.settingsStore.load();
    if (!mounted) return;
    setState(() => _settings = loaded);

    try {
      await widget.backend.initialize();
      if (!mounted) return;
      final isIosNative = widget.backend is IosCoreMLBackend;
      final platformLabel = isIosNative
          ? 'iOS On-Device Core ML'
          : (kIsWeb ? 'Flutter Web Desktop Demo' : 'Flutter Desktop Demo');
      setState(() {
        _backendBanner = '$platformLabel • ${widget.backend.backendLabel}';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _errorBanner = 'Backend connection error: $e');
    }

    if (_webcam.isSupported) {
      await _webcam.start();
      if (mounted) setState(() {});
    }
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _webcam.dispose();
    super.dispose();
  }

  Future<Size> _decodeImageSize(Uint8List bytes, int hintW, int hintH) async {
    if (hintW > 0 && hintH > 0) {
      return Size(hintW.toDouble(), hintH.toDouble());
    }
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final sz = Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    frame.image.dispose();
    codec.dispose();
    return sz;
  }

  double _elapsedMs(Stopwatch sw) => sw.elapsedMicroseconds / 1000.0;

  void _finishClientStopwatch(Stopwatch sw) {
    if (sw.isRunning) {
      sw.stop();
    }
    _clientEndToEndMs = _elapsedMs(sw);
  }

  Future<void> _submitImage({
    required Uint8List bytes,
    required String filename,
    required int streamWidth,
    required int streamHeight,
    required String source,
    required Stopwatch clientSw,
    required String actionLabel,
    required double captureEncodeMs,
  }) async {
    _pollTimer?.cancel();
    if (_activeRun != null && !_activeRun!.isTerminal) {
      unawaited(widget.backend.cancelRun(_activeRun!.runId));
    }
    final sz = await _decodeImageSize(bytes, streamWidth, streamHeight);
    final snapSettings = _settings;
    setState(() {
      _errorBanner = null;
      _stillImageBytes = bytes;
      _stillImageSize = sz;
      _selectedCropId = null;
      _runSettings = snapSettings;
      _activeRun = null;
      _expectedRunId = null;
      _clientStopwatch = clientSw;
      _clientActionLabel = actionLabel;
      _clientCaptureEncodeMs = captureEncodeMs;
      _clientUploadAckMs = null;
      _clientEndToEndMs = null;
    });

    try {
      final run = await widget.backend.startCapture(
        imageBytes: bytes,
        filename: filename,
        streamWidth: streamWidth > 0 ? streamWidth : sz.width.toInt(),
        streamHeight: streamHeight > 0 ? streamHeight : sz.height.toInt(),
        captureSource: source,
        settings: snapSettings,
      );
      if (!mounted) return;
      final ackMs = _elapsedMs(clientSw);
      setState(() {
        _activeRun = run;
        _expectedRunId = run.runId;
        _clientUploadAckMs = ackMs;
        if (run.isTerminal) {
          _finishClientStopwatch(clientSw);
        }
      });
      if (!run.isTerminal) {
        _startPolling(run.runId);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _finishClientStopwatch(clientSw);
        _errorBanner = 'Capture failed: $e';
      });
    }
  }

  Future<void> _onShutterPressed() async {
    // Start frontend monotonic stopwatch BEFORE PNG encoding / upload
    final sw = Stopwatch()..start();
    try {
      final frame = await _webcam.captureExactStreamFrame();
      final encodeMs = _elapsedMs(sw);
      await _submitImage(
        bytes: frame.pngBytes,
        filename: 'shutter_${DateTime.now().millisecondsSinceEpoch}.png',
        streamWidth: frame.streamWidth,
        streamHeight: frame.streamHeight,
        source: 'webcam_stream_buffer',
        clientSw: sw,
        actionLabel: 'Shutter→Result',
        captureEncodeMs: encodeMs,
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _clientStopwatch = sw;
        _clientActionLabel = 'Shutter→Result';
        _finishClientStopwatch(sw);
        _errorBanner = 'Shutter capture failed: $e';
      });
    }
  }

  Future<void> _onImportImage() async {
    final picked = await FilePicker.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['png', 'jpg', 'jpeg', 'webp'],
    );
    if (picked.isEmpty) return;
    // Start frontend monotonic stopwatch BEFORE reading file bytes / upload
    final sw = Stopwatch()..start();
    final f = picked.first;
    final bytes = await f.xFile.readAsBytes();
    final readMs = _elapsedMs(sw);
    if (bytes.isEmpty) {
      sw.stop();
      setState(() => _errorBanner = 'Selected file could not be read.');
      return;
    }
    await _submitImage(
      bytes: bytes,
      filename: f.name,
      streamWidth: 0,
      streamHeight: 0,
      source: 'desktop_import',
      clientSw: sw,
      actionLabel: 'Import→Result',
      captureEncodeMs: readMs,
    );
  }

  Future<void> _onReprocess() async {
    final cur = _activeRun;
    if (cur == null) return;
    _pollTimer?.cancel();
    if (!cur.isTerminal) {
      unawaited(widget.backend.cancelRun(cur.runId));
    }
    final sw = Stopwatch()..start();
    final snapSettings = _settings;
    setState(() {
      _errorBanner = null;
      _selectedCropId = null;
      _runSettings = snapSettings;
      _clientStopwatch = sw;
      _clientActionLabel = 'Reprocess→Result';
      _clientCaptureEncodeMs = 0.0;
      _clientUploadAckMs = null;
      _clientEndToEndMs = null;
    });
    try {
      final run = await widget.backend.startReprocess(
        captureId: cur.captureId,
        settings: snapSettings,
      );
      if (!mounted) return;
      final ackMs = _elapsedMs(sw);
      setState(() {
        _activeRun = run;
        _expectedRunId = run.runId;
        _clientUploadAckMs = ackMs;
        if (run.isTerminal) {
          _finishClientStopwatch(sw);
        }
      });
      if (!run.isTerminal) {
        _startPolling(run.runId);
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _finishClientStopwatch(sw);
        _errorBanner = 'Reprocess failed: $e';
      });
    }
  }

  void _startPolling(String runId) {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(milliseconds: 250), (t) async {
      if (!mounted || _expectedRunId != runId) {
        t.cancel();
        return;
      }
      try {
        final updated = await widget.backend.fetchRun(runId);
        if (!mounted || _expectedRunId != runId) {
          t.cancel();
          return;
        }
        setState(() {
          _activeRun = updated;
          if (updated.isTerminal && _clientStopwatch != null) {
            _finishClientStopwatch(_clientStopwatch!);
          }
        });
        if (updated.isTerminal) {
          t.cancel();
        }
      } catch (e) {
        t.cancel();
        if (mounted && _expectedRunId == runId) {
          setState(() {
            if (_clientStopwatch != null) {
              _finishClientStopwatch(_clientStopwatch!);
            }
            _errorBanner = 'Polling error: $e';
          });
        }
      }
    });
  }

  Future<void> _onCancelRun() async {
    final cur = _activeRun;
    if (cur == null || cur.isTerminal) return;
    await widget.backend.cancelRun(cur.runId);
    if (!mounted) return;
    final refreshed = await widget.backend.fetchRun(cur.runId);
    if (mounted && _expectedRunId == cur.runId) {
      setState(() {
        _activeRun = refreshed;
        if (refreshed.isTerminal && _clientStopwatch != null) {
          _finishClientStopwatch(_clientStopwatch!);
        }
      });
    }
  }

  void _onRetake() {
    _pollTimer?.cancel();
    _clientStopwatch?.stop();
    final cur = _activeRun;
    if (cur != null && !cur.isTerminal) {
      unawaited(widget.backend.cancelRun(cur.runId));
    }
    setState(() {
      _stillImageBytes = null;
      _stillImageSize = null;
      _activeRun = null;
      _expectedRunId = null;
      _selectedCropId = null;
      _runSettings = null;
      _errorBanner = null;
      _clientStopwatch = null;
      _clientCaptureEncodeMs = null;
      _clientUploadAckMs = null;
      _clientEndToEndMs = null;
    });
  }

  void _openSettings() {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (_) => SettingsSheet(
        initial: _settings,
        onApply: (updated) async {
          await widget.settingsStore.save(updated);
          if (mounted) setState(() => _settings = updated);
        },
        onReset: () async {
          final def = await widget.settingsStore.reset();
          if (mounted) setState(() => _settings = def);
        },
      ),
    );
  }

  void _openCropPopup(CropResult crop) {
    setState(() => _selectedCropId = crop.cropId);
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (_) => CropDetailPopup(crop: crop),
    );
  }

  void _openDiagnostics() {
    final backend = widget.backend;
    if (backend is! IosCoreMLBackend) return;
    Navigator.of(context).push(
      MaterialPageRoute<void>(
        builder: (_) => DetectorDiagnosticScreen(
          backend: backend,
          capturedImageBytes: _stillImageBytes,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final inResultView = _stillImageBytes != null;
    final settingsChanged =
        inResultView && _runSettings != null && _runSettings != _settings;
    final isIosBackend = widget.backend is IosCoreMLBackend;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Menu Camera MVP (Stage 4)'),
        backgroundColor: const Color(0xFF0F172A),
        foregroundColor: Colors.white,
        actions: [
          if (inResultView && settingsChanged)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: FilledButton.tonalIcon(
                key: const Key('reprocess_btn'),
                onPressed: _onReprocess,
                icon: const Icon(Icons.refresh, size: 18),
                label: const Text('Reprocess'),
              ),
            ),
          if (isIosBackend)
            IconButton(
              key: const Key('diagnostics_btn'),
              icon: const Icon(Icons.bug_report_outlined),
              tooltip: 'Detector Diagnostics',
              onPressed: _openDiagnostics,
            ),
          IconButton(
            key: const Key('settings_btn'),
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
            onPressed: _openSettings,
          ),
        ],
      ),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            color: const Color(0xFF1E293B),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    _backendBanner,
                    style: const TextStyle(color: Colors.white70, fontSize: 12),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Text(
                  'conf=${_settings.yoloConfidence.toStringAsFixed(2)} | ${_settings.decodeMode == "greedy" ? "Greedy" : "CharLM α=${_settings.charLmAlpha.toStringAsFixed(2)}"}',
                  style: const TextStyle(color: Colors.amberAccent, fontSize: 12),
                ),
              ],
            ),
          ),
          if (_errorBanner != null)
            Container(
              width: double.infinity,
              color: Colors.red.shade900,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              child: Text(_errorBanner!,
                  style: const TextStyle(color: Colors.white, fontSize: 13)),
            ),
          Expanded(
            child: inResultView ? _buildResultView() : _buildPreviewView(),
          ),
        ],
      ),
    );
  }

  Widget _buildPreviewView() {
    return Container(
      color: Colors.black,
      child: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                _webcam.buildPreview(),
                if (_webcam.isReady)
                  Positioned(
                    top: 12,
                    left: 12,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        'Live stream: ${_webcam.streamWidth}×${_webcam.streamHeight} (no live inference)',
                        style:
                            const TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Container(
            width: double.infinity,
            color: const Color(0xFF0F172A),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Wrap(
              alignment: WrapAlignment.center,
              spacing: 14,
              runSpacing: 10,
              children: [
                OutlinedButton.icon(
                  key: const Key('import_image_btn'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                    side: const BorderSide(color: Colors.white54),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                  ),
                  onPressed: _onImportImage,
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Import Menu Image'),
                ),
                FilledButton.icon(
                  key: const Key('shutter_btn'),
                  style: FilledButton.styleFrom(
                    backgroundColor: const Color(0xFF10B981),
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 12),
                  ),
                  onPressed: _webcam.isReady ? _onShutterPressed : null,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Shutter (Capture Menu)',
                      style: TextStyle(fontWeight: FontWeight.w700)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultView() {
    final run = _activeRun;
    final imgBytes = _stillImageBytes!;
    final srcSize = _stillImageSize ?? const Size(1, 1);

    return Column(
      children: [
        _buildTimingHeader(run),
        if (run != null && run.status == 'failed')
          Container(
            key: const Key('run_failed_error_banner'),
            width: double.infinity,
            color: const Color(0xFF991B1B),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
            child: SelectableText(
              'Run failed (stage: ${run.stage}): ${run.error ?? "Unknown error"}',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w600,
                fontSize: 13,
              ),
            ),
          ),
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              final vp = Size(constraints.maxWidth, constraints.maxHeight);
              final xform = ImageViewportTransform.contain(
                sourceSize: srcSize,
                viewportSize: vp,
              );
              return InteractiveViewer(
                minScale: 1.0,
                maxScale: 5.0,
                child: GestureDetector(
                  key: const Key('still_image_canvas'),
                  onTapUp: (details) {
                    if (run == null || run.crops.isEmpty) return;
                    final srcPt = xform.viewportToSource(details.localPosition);
                    for (final c in run.crops) {
                      if (pointInPolygon(srcPt, c.polygon)) {
                        _openCropPopup(c);
                        return;
                      }
                    }
                  },
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.memory(imgBytes, fit: BoxFit.contain),
                      if (run != null)
                        CustomPaint(
                          painter: _BoxesPainter(
                            crops: run.crops,
                            transform: xform,
                            selectedCropId: _selectedCropId,
                          ),
                        ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
        Container(
          color: const Color(0xFF0F172A),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(
            children: [
              OutlinedButton.icon(
                key: const Key('retake_btn'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white,
                  side: const BorderSide(color: Colors.white54),
                ),
                onPressed: _onRetake,
                icon: const Icon(Icons.camera_alt_outlined),
                label: const Text('Retake'),
              ),
              const SizedBox(width: 10),
              if (run != null && !run.isTerminal)
                FilledButton.tonalIcon(
                  key: const Key('cancel_run_btn'),
                  onPressed: _onCancelRun,
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('Cancel'),
                ),
              const Spacer(),
              if (run != null && run.status == 'completed')
                Text(
                  run.crops.isEmpty
                      ? 'No menu text lines detected'
                      : 'Tap any highlighted box for cached JP + English',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
            ],
          ),
        ),
      ],
    );
  }

  String _formatCacheBadge(PageRun run) {
    if (run.cacheStatus == 'reused_cached_logits' || run.reusedCachedLogits) {
      return 'Cache: Reused Cached Logits';
    }
    if (run.cacheStatus == 'reused_raw_detections_fresh_ocr' ||
        run.reusedCachedDetections) {
      return 'Cache: Reused Detections + Fresh OCR';
    }
    if (run.cacheStatus == 'fresh_full_inference') {
      return 'Cache: Fresh Full Inference';
    }
    return 'Cache: ${run.cacheStatus}';
  }

  Widget _buildTimingHeader(PageRun? run) {
    final liveClientMs = _clientEndToEndMs ??
        (_clientStopwatch != null ? _elapsedMs(_clientStopwatch!) : null);
    final clientText = liveClientMs != null
        ? '${(liveClientMs / 1000.0).toStringAsFixed(2)}s'
        : '...';

    if (run == null) {
      return Container(
        width: double.infinity,
        color: const Color(0xFFF1F5F9),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Client $_clientActionLabel: $clientText (uploading...)',
              key: const Key('client_end_to_end_text'),
              style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
            ),
            const SizedBox(height: 4),
            const LinearProgressIndicator(minHeight: 4),
          ],
        ),
      );
    }
    final t = run.timingsMs;
    final c = run.counts;
    final doneN = c.completedCrops + c.failedCrops;
    final totalN = c.postprocessedLines;
    final srvReqMs = t.serverRequestTotalMs ?? t.totalWallMs;
    final srvPipeMs = t.serverPipelineWallMs;
    final srvReqText = srvReqMs != null
        ? '${(srvReqMs / 1000.0).toStringAsFixed(2)}s'
        : 'running (${run.stage})...';
    final srvPipeText = srvPipeMs != null
        ? '${(srvPipeMs / 1000.0).toStringAsFixed(2)}s'
        : '...';
    final cacheBadge = _formatCacheBadge(run);

    return Container(
      width: double.infinity,
      color: const Color(0xFFF1F5F9),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 12,
            runSpacing: 4,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              Text(
                'Client $_clientActionLabel: $clientText',
                key: const Key('client_end_to_end_text'),
                style:
                    const TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
              ),
              Text(
                'Server Req: $srvReqText (Pipeline: $srvPipeText)',
                key: const Key('page_total_time_text'),
                style:
                    const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
              ),
              Container(
                key: const Key('cache_status_badge'),
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: run.reusedCachedLogits
                      ? const Color(0xFFDCFCE7)
                      : const Color(0xFFE2E8F0),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  cacheBadge,
                  style: const TextStyle(
                      fontSize: 11, fontWeight: FontWeight.w600),
                ),
              ),
              Text(
                'Lines: $doneN/$totalN (${c.totalWindows} win${c.failedCrops > 0 ? ", ${c.failedCrops} failed" : ""})',
                style: const TextStyle(fontSize: 13),
              ),
              TextButton.icon(
                key: const Key('toggle_timings_btn'),
                onPressed: () =>
                    setState(() => _timingsExpanded = !_timingsExpanded),
                icon: Icon(
                    _timingsExpanded ? Icons.expand_less : Icons.expand_more,
                    size: 18),
                label: const Text('Details'),
              ),
            ],
          ),
          if (!run.isTerminal)
            LinearProgressIndicator(
              value: totalN > 0 ? doneN / totalN : null,
              minHeight: 4,
            ),
          if (_timingsExpanded) ...[
            const Divider(height: 12),
            Wrap(
              spacing: 16,
              runSpacing: 4,
              children: [
                Text(
                  'Client Monotonic (ms): action=$_clientActionLabel | capture/encode=${_clientCaptureEncodeMs?.toStringAsFixed(1) ?? "-"} | uploadAck=${_clientUploadAckMs?.toStringAsFixed(1) ?? "-"} | endToEnd=${liveClientMs?.toStringAsFixed(1) ?? "-"}',
                  key: const Key('client_timing_details_text'),
                  style: const TextStyle(fontSize: 12),
                ),
                Text(
                    'Status: ${run.status} (${run.coldOrWarm}, ${run.cacheStatus})',
                    style: const TextStyle(fontSize: 12)),
                Text(
                    'Capture: stream ${run.capture.streamWidth}×${run.capture.streamHeight} (rot=${run.capture.streamRotation}°) / saved ${run.capture.savedWidth}×${run.capture.savedHeight} (parity=${run.capture.streamSavedParity}, ${run.capture.streamSavedParityMode})',
                    style: const TextStyle(fontSize: 12)),
                Text(
                    'Hardware: ${run.deviceName} (${run.precision.toUpperCase()}) • ${run.backendId}',
                    style: const TextStyle(fontSize: 12)),
                Text(
                    'Server Monotonic (ms): reqTotal=${srvReqMs ?? "-"} | pipelineWall=${srvPipeMs ?? "-"} | ingest=${t.requestIngestMs} (read=${t.requestReadMs}, dec=${t.decodeNormalizeMs}, persist=${t.persistMs}) | queue=${t.queueWaitMs} | cold=${t.coldLoadMs} | detect=${t.detectMs} | post=${t.postprocessMs} | rec=${t.recognizeMs} | lm=${t.decodeMs} | lookup=${t.lookupMs} | first=${t.firstResultMs ?? "-"} | sumCrop=${t.summedCropWorkMs}',
                    style: const TextStyle(fontSize: 12)),
                Text(
                    'Detections: raw=${c.rawDetections} → post=${c.postprocessedLines} | runId=${run.runId} | configId=${run.configId}',
                    style: const TextStyle(fontSize: 12)),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

class _BoxesPainter extends CustomPainter {
  const _BoxesPainter({
    required this.crops,
    required this.transform,
    required this.selectedCropId,
  });

  final List<CropResult> crops;
  final ImageViewportTransform transform;
  final String? selectedCropId;

  @override
  void paint(Canvas canvas, Size size) {
    for (final c in crops) {
      if (c.polygon.length < 3) continue;
      final pts = transform.polygonToViewport(c.polygon);
      final path = Path()..moveTo(pts.first.dx, pts.first.dy);
      for (var i = 1; i < pts.length; i++) {
        path.lineTo(pts[i].dx, pts[i].dy);
      }
      path.close();

      final isSelected = c.cropId == selectedCropId;
      final isFailed = c.status == 'failed';
      final strokeColor = isFailed
          ? Colors.redAccent
          : (isSelected ? Colors.amberAccent : const Color(0xFF10B981));

      canvas.drawPath(
        path,
        Paint()
          ..color = strokeColor.withValues(alpha: isSelected ? 0.30 : 0.14)
          ..style = PaintingStyle.fill,
      );
      canvas.drawPath(
        path,
        Paint()
          ..color = strokeColor
          ..strokeWidth = isSelected ? 3.0 : 1.8
          ..style = PaintingStyle.stroke,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _BoxesPainter old) =>
      old.crops != crops ||
      old.selectedCropId != selectedCropId ||
      old.transform.scale != transform.scale ||
      old.transform.offset != transform.offset;
}
