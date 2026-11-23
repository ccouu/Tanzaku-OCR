/// iOS Core ML Detector Diagnostic Screen (detection-only; skips OCR/LM/lexicon).
///
/// Allows on-device debugging of V5 YOLO11n-OBB 1024 Core ML detection on either:
///   - Bundled `menu-101.jpg` reference sample (expected 15 boxes at conf=0.35), or
///   - Current captured/imported still image bytes (disabled when null).
///
/// No implicit auto-test runs when opened. Displays status/stage/error, structured
/// native tensor & threshold metrics, source preview with decoded OBB polygons,
/// actual 1024x1024 detector input preview, and Copy / Save JSON report actions.
library;

import 'dart:convert';
import 'dart:ui' as ui;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/services.dart';
import 'package:material_ui/material_ui.dart';

import '../geometry/overlay_geometry.dart';
import '../services/ios_coreml_backend.dart';

typedef DetectorDiagnosticRunner = Future<Map<String, dynamic>> Function(
  Uint8List? imageBytes, {
  required double confidence,
  bool cpuOnly,
});

class DetectorDiagnosticScreen extends StatefulWidget {
  const DetectorDiagnosticScreen({
    super.key,
    required this.backend,
    this.capturedImageBytes,
    this.diagnosticRunner,
  });

  final IosCoreMLBackend backend;
  final Uint8List? capturedImageBytes;
  final DetectorDiagnosticRunner? diagnosticRunner;

  @override
  State<DetectorDiagnosticScreen> createState() =>
      _DetectorDiagnosticScreenState();
}

class _DetectorDiagnosticScreenState extends State<DetectorDiagnosticScreen> {
  double _confidence = 0.35;
  bool _cpuOnly = false;
  bool _includePreviewsInCopy = false;
  bool _running = false;
  String _runningTarget = '';
  Map<String, dynamic>? _report;
  String? _exceptionError;

  Uint8List? _sourcePreviewBytes;
  Size? _sourcePreviewSize;
  Uint8List? _detectorInputBytes;
  List<_DiagBox> _decodedBoxes = const [];

  late final TextEditingController _confController;

  @override
  void initState() {
    super.initState();
    _confController = TextEditingController(text: _confidence.toStringAsFixed(2));
  }

  @override
  void dispose() {
    _confController.dispose();
    super.dispose();
  }

  void _setConfidence(double v) {
    final clamped = double.parse(v.clamp(0.01, 0.90).toStringAsFixed(2));
    setState(() {
      _confidence = clamped;
      _confController.text = clamped.toStringAsFixed(2);
    });
  }

  Future<Size?> _decodeImageSize(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final sz = Size(frame.image.width.toDouble(), frame.image.height.toDouble());
      frame.image.dispose();
      codec.dispose();
      return sz;
    } catch (_) {
      return null;
    }
  }

  Uint8List? _tryDecodeBase64(dynamic raw) {
    if (raw is! String || raw.trim().isEmpty) return null;
    try {
      return base64Decode(raw.trim());
    } catch (_) {
      return null;
    }
  }

  List<_DiagBox> _parseBoxes(dynamic raw) {
    if (raw is! List) return const [];
    final out = <_DiagBox>[];
    for (final item in raw) {
      if (item is! Map) continue;
      final ptsRaw = item['points'];
      if (ptsRaw is! List) continue;
      final pts = <List<double>>[];
      for (final p in ptsRaw) {
        if (p is List && p.length >= 2) {
          pts.add([(p[0] as num).toDouble(), (p[1] as num).toDouble()]);
        }
      }
      if (pts.length >= 3) {
        out.add(_DiagBox(
          points: pts,
          confidence: (item['confidence'] as num?)?.toDouble() ?? 0.0,
          classId: (item['class_id'] as num?)?.toInt() ?? 0,
        ));
      }
    }
    return out;
  }

  Future<void> _runDiagnostic({required bool useBundledSample}) async {
    if (_running) return;
    final Uint8List? inputBytes =
        useBundledSample ? null : widget.capturedImageBytes;
    if (!useBundledSample && inputBytes == null) return;

    setState(() {
      _running = true;
      _runningTarget =
          useBundledSample ? 'bundled menu-101.jpg' : 'current capture';
      _exceptionError = null;
      _report = null;
      _sourcePreviewBytes = null;
      _sourcePreviewSize = null;
      _detectorInputBytes = null;
      _decodedBoxes = const [];
    });

    try {
      final Map<String, dynamic> result;
      if (widget.diagnosticRunner != null) {
        result = await widget.diagnosticRunner!(
          inputBytes,
          confidence: _confidence,
          cpuOnly: _cpuOnly,
        );
      } else {
        result = await widget.backend.runDetectorDiagnostic(
          inputBytes,
          confidence: _confidence,
          cpuOnly: _cpuOnly,
        );
      }

      final srcBytes = _tryDecodeBase64(result['source_preview_png_b64']);
      final detBytes = _tryDecodeBase64(result['detector_input_png_b64']);
      final srcSize = srcBytes != null ? await _decodeImageSize(srcBytes) : null;
      final boxes = _parseBoxes(result['decoded_boxes']);

      if (!mounted) return;
      setState(() {
        _running = false;
        _report = result;
        _sourcePreviewBytes = srcBytes;
        _sourcePreviewSize = srcSize;
        _detectorInputBytes = detBytes;
        _decodedBoxes = boxes;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _running = false;
        _exceptionError = e.toString();
        _report = {
          'status': 'failed',
          'stage': 'dart_invocation',
          'error': e.toString(),
          'effective_confidence': _confidence,
          'cpu_only': _cpuOnly,
        };
      });
    }
  }

  String _buildExportJson({required bool includePreviews}) {
    final rep = _report;
    if (rep == null) return '{}';
    final copy = Map<String, dynamic>.from(rep);
    if (!includePreviews) {
      if (copy['source_preview_png_b64'] is String) {
        copy['source_preview_png_b64'] =
            '<omitted ${(copy['source_preview_png_b64'] as String).length} base64 chars>';
      }
      if (copy['detector_input_png_b64'] is String) {
        copy['detector_input_png_b64'] =
            '<omitted ${(copy['detector_input_png_b64'] as String).length} base64 chars>';
      }
    }
    return const JsonEncoder.withIndent('  ').convert(copy);
  }

  Future<void> _copyReport() async {
    final text = _buildExportJson(includePreviews: _includePreviewsInCopy);
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          'Copied diagnostic JSON (${text.length} chars, previews ${_includePreviewsInCopy ? "included" : "omitted"}).',
        ),
      ),
    );
  }

  Future<void> _saveReport() async {
    final text = _buildExportJson(includePreviews: _includePreviewsInCopy);
    final bytes = Uint8List.fromList(utf8.encode(text));
    try {
      final savedPath = await FilePicker.saveFile(
        dialogTitle: 'Save Detector Diagnostic JSON',
        fileName:
            'detector_diagnostic_${DateTime.now().millisecondsSinceEpoch}.json',
        bytes: bytes,
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            savedPath != null
                ? 'Saved JSON report to $savedPath'
                : 'Save cancelled (you can also use Copy JSON Report).',
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Save JSON not available on this device ($e). Use Copy JSON Report instead.',
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final hasCapture = widget.capturedImageBytes != null;
    final rep = _report;
    final status = (rep?['status'] ?? '').toString();
    final stage = (rep?['stage'] ?? '').toString();
    final errText = (rep?['error'] ?? _exceptionError)?.toString();
    final isFailed = status == 'failed' || (errText != null && errText.isNotEmpty);
    final nativeMap = rep?['native'] is Map
        ? Map<String, dynamic>.from(rep!['native'] as Map)
        : null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('iOS Detector Diagnostics'),
        backgroundColor: const Color(0xFF0F172A),
        foregroundColor: Colors.white,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          key: const Key('diag_scroll_view'),
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Instructions card
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF1F5F9),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: const Color(0xFFCBD5E1)),
                ),
                child: const Text(
                  'Detection-only diagnostic (skips OCR, CharLM, and lexicon). '
                  'No test runs automatically.\n'
                  'How to report: tap "Run Bundled menu-101 Test" (expected 15 boxes at conf=0.35), '
                  'optionally retry with CPU-only ON or lower confidence, then tap "Copy JSON Report" '
                  '(or "Save JSON") and paste the exact JSON and error text.',
                  style: TextStyle(fontSize: 12.5, height: 1.35),
                ),
              ),
              const SizedBox(height: 12),

              // Controls card
              Card(
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                  side: const BorderSide(color: Color(0xFFCBD5E1)),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Text(
                            'Confidence threshold:',
                            style: TextStyle(fontWeight: FontWeight.w600),
                          ),
                          const SizedBox(width: 10),
                          SizedBox(
                            width: 78,
                            child: TextField(
                              key: const Key('diag_conf_input'),
                              controller: _confController,
                              keyboardType: const TextInputType.numberWithOptions(
                                decimal: true,
                              ),
                              decoration: const InputDecoration(
                                isDense: true,
                                border: OutlineInputBorder(),
                                contentPadding: EdgeInsets.symmetric(
                                    horizontal: 8, vertical: 6),
                              ),
                              onSubmitted: (txt) {
                                final parsed = double.tryParse(txt.trim());
                                if (parsed != null) _setConfidence(parsed);
                              },
                            ),
                          ),
                          const Spacer(),
                          Text(
                            _confidence.toStringAsFixed(2),
                            key: const Key('diag_conf_value_text'),
                            style: const TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 15,
                            ),
                          ),
                        ],
                      ),
                      Slider(
                        key: const Key('diag_conf_slider'),
                        value: _confidence,
                        min: 0.01,
                        max: 0.90,
                        divisions: 89,
                        label: _confidence.toStringAsFixed(2),
                        onChanged: _running ? null : _setConfidence,
                      ),
                      Wrap(
                        spacing: 8,
                        children: [
                          for (final preset in const [0.05, 0.20, 0.35, 0.50])
                            ChoiceChip(
                              label: Text(preset.toStringAsFixed(2)),
                              selected: (_confidence - preset).abs() < 0.005,
                              onSelected:
                                  _running ? null : (_) => _setConfidence(preset),
                            ),
                        ],
                      ),
                      const Divider(height: 16),
                      SwitchListTile(
                        key: const Key('diag_cpu_only_switch'),
                        contentPadding: EdgeInsets.zero,
                        title: const Text('CPU-only Core ML compute units'),
                        subtitle: const Text(
                          'Forces MLComputeUnits.cpuOnly to isolate ANE/GPU vs CPU behavior',
                          style: TextStyle(fontSize: 12),
                        ),
                        value: _cpuOnly,
                        onChanged: _running
                            ? null
                            : (v) => setState(() => _cpuOnly = v),
                      ),
                      const SizedBox(height: 8),
                      Wrap(
                        spacing: 10,
                        runSpacing: 10,
                        children: [
                          FilledButton.icon(
                            key: const Key('diag_run_bundled_btn'),
                            onPressed: _running
                                ? null
                                : () => _runDiagnostic(useBundledSample: true),
                            icon: const Icon(Icons.science_outlined),
                            label: const Text('Run Bundled menu-101 Test'),
                          ),
                          OutlinedButton.icon(
                            key: const Key('diag_run_capture_btn'),
                            onPressed: (_running || !hasCapture)
                                ? null
                                : () => _runDiagnostic(useBundledSample: false),
                            icon: const Icon(Icons.image_search_outlined),
                            label: Text(
                              hasCapture
                                  ? 'Run Current Capture'
                                  : 'Run Current Capture (no still)',
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              if (_running) ...[
                const SizedBox(height: 12),
                Container(
                  key: const Key('diag_running_indicator'),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: const Color(0xFFEFF6FF),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: const Color(0xFF93C5FD)),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Running detector diagnostic on $_runningTarget (conf=${_confidence.toStringAsFixed(2)}, cpuOnly=$_cpuOnly)...',
                        style: const TextStyle(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 8),
                      const LinearProgressIndicator(minHeight: 4),
                    ],
                  ),
                ),
              ],

              if (rep != null) ...[
                const SizedBox(height: 12),
                // Status / Stage / Error Card
                Container(
                  key: const Key('diag_status_card'),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isFailed
                        ? const Color(0xFFFEF2F2)
                        : const Color(0xFFECFDF5),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: isFailed
                          ? const Color(0xFFFCA5A5)
                          : const Color(0xFF6EE7B7),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Wrap(
                        spacing: 12,
                        runSpacing: 4,
                        children: [
                          Text(
                            'Status: ${status.isEmpty ? "unknown" : status}',
                            key: const Key('diag_status_text'),
                            style: TextStyle(
                              fontWeight: FontWeight.w700,
                              fontSize: 15,
                              color: isFailed
                                  ? const Color(0xFF991B1B)
                                  : const Color(0xFF065F46),
                            ),
                          ),
                          Text(
                            'Stage: ${stage.isEmpty ? "-" : stage}',
                            key: const Key('diag_stage_text'),
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                          Text(
                            'Post-NMS Boxes: ${rep['post_nms_count'] ?? _decodedBoxes.length}'
                            '${rep['expected_reference_boxes'] != null ? " (expected ${rep['expected_reference_boxes']})" : ""}',
                            key: const Key('diag_box_count_text'),
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                        ],
                      ),
                      if (rep['effective_confidence'] != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 4),
                          child: Text(
                            'Effective conf: ${rep['effective_confidence']} | Compute units: ${nativeMap?['compute_units'] ?? (_cpuOnly ? "cpuOnly" : "all")}',
                            style: const TextStyle(fontSize: 12.5),
                          ),
                        ),
                      if (errText != null && errText.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.only(top: 8),
                          child: SelectableText(
                            'Error: $errText',
                            key: const Key('diag_error_text'),
                            style: const TextStyle(
                              color: Color(0xFFB91C1C),
                              fontWeight: FontWeight.w700,
                              fontSize: 13,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),

                const SizedBox(height: 12),
                // Copy / Save actions
                Wrap(
                  spacing: 10,
                  runSpacing: 8,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    FilledButton.tonalIcon(
                      key: const Key('diag_copy_json_btn'),
                      onPressed: _copyReport,
                      icon: const Icon(Icons.copy, size: 18),
                      label: const Text('Copy JSON Report'),
                    ),
                    OutlinedButton.icon(
                      key: const Key('diag_save_json_btn'),
                      onPressed: _saveReport,
                      icon: const Icon(Icons.save_alt, size: 18),
                      label: const Text('Save JSON'),
                    ),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Checkbox(
                          value: _includePreviewsInCopy,
                          onChanged: (v) => setState(
                              () => _includePreviewsInCopy = v ?? false),
                        ),
                        const Text('Include base64 previews',
                            style: TextStyle(fontSize: 12)),
                      ],
                    ),
                  ],
                ),

                // Native metrics summary
                if (nativeMap != null && nativeMap.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  const Text(
                    'Native Core ML Metrics',
                    style: TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    key: const Key('diag_native_metrics_box'),
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF8FAFC),
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: const Color(0xFFCBD5E1)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        for (final e in nativeMap.entries)
                          Padding(
                            padding: const EdgeInsets.symmetric(vertical: 2),
                            child: SelectableText(
                              '${e.key}: ${e.value is Map || e.value is List ? jsonEncode(e.value) : e.value}',
                              style: const TextStyle(
                                fontFamily: 'monospace',
                                fontSize: 12,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],

                // Source preview with OBB polygons
                if (_sourcePreviewBytes != null) ...[
                  const SizedBox(height: 14),
                  Text(
                    'Source Preview + Decoded OBB Polygons (${_decodedBoxes.length} boxes)',
                    style: const TextStyle(
                        fontWeight: FontWeight.w700, fontSize: 14),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    key: const Key('diag_source_preview_box'),
                    height: 320,
                    decoration: BoxDecoration(
                      color: Colors.black,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: LayoutBuilder(
                      builder: (context, constraints) {
                        final srcSz =
                            _sourcePreviewSize ?? const Size(1024, 1024);
                        final xform = ImageViewportTransform.contain(
                          sourceSize: srcSz,
                          viewportSize:
                              Size(constraints.maxWidth, constraints.maxHeight),
                        );
                        return Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.memory(_sourcePreviewBytes!,
                                fit: BoxFit.contain),
                            CustomPaint(
                              painter: _DiagBoxesPainter(
                                boxes: _decodedBoxes,
                                transform: xform,
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ),
                ],

                // Actual 1024x1024 detector input preview
                if (_detectorInputBytes != null) ...[
                  const SizedBox(height: 14),
                  const Text(
                    'Actual 1024×1024 Detector Input Preview (Letterboxed)',
                    style: TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
                  ),
                  const SizedBox(height: 6),
                  Container(
                    key: const Key('diag_detector_input_preview_box'),
                    height: 260,
                    decoration: BoxDecoration(
                      color: Colors.black,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Image.memory(_detectorInputBytes!,
                        fit: BoxFit.contain),
                  ),
                ],

                // Full JSON report
                const SizedBox(height: 14),
                const Text(
                  'Full Diagnostic JSON Report',
                  style: TextStyle(fontWeight: FontWeight.w700, fontSize: 14),
                ),
                const SizedBox(height: 6),
                Container(
                  key: const Key('diag_json_report_box'),
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFF0F172A),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: SelectableText(
                    _buildExportJson(includePreviews: false),
                    style: const TextStyle(
                      color: Color(0xFFE2E8F0),
                      fontFamily: 'monospace',
                      fontSize: 11.5,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _DiagBox {
  const _DiagBox({
    required this.points,
    required this.confidence,
    required this.classId,
  });

  final List<List<double>> points;
  final double confidence;
  final int classId;
}

class _DiagBoxesPainter extends CustomPainter {
  const _DiagBoxesPainter({
    required this.boxes,
    required this.transform,
  });

  final List<_DiagBox> boxes;
  final ImageViewportTransform transform;

  @override
  void paint(Canvas canvas, Size size) {
    for (final b in boxes) {
      if (b.points.length < 3) continue;
      final pts = transform.polygonToViewport(b.points);
      final path = Path()..moveTo(pts.first.dx, pts.first.dy);
      for (var i = 1; i < pts.length; i++) {
        path.lineTo(pts[i].dx, pts[i].dy);
      }
      path.close();
      final color = b.classId == 1
          ? const Color(0xFFF59E0B)
          : const Color(0xFF10B981);
      canvas.drawPath(
        path,
        Paint()
          ..color = color.withValues(alpha: 0.16)
          ..style = PaintingStyle.fill,
      );
      canvas.drawPath(
        path,
        Paint()
          ..color = color
          ..strokeWidth = 2.0
          ..style = PaintingStyle.stroke,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _DiagBoxesPainter old) =>
      old.boxes != boxes ||
      old.transform.scale != transform.scale ||
      old.transform.offset != transform.offset;
}
