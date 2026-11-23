/// Immediate Stage 4 Settings sheet: YOLO confidence, CharLM alpha ("Language-model weight"),
/// Decoder mode (CharLM R1 vs Canonical Greedy), and Reset defaults.
library;

import 'package:material_ui/material_ui.dart';
import '../models/contracts.dart';

class SettingsSheet extends StatefulWidget {
  const SettingsSheet({
    super.key,
    required this.initial,
    required this.onApply,
    required this.onReset,
  });

  final AppSettings initial;
  final ValueChanged<AppSettings> onApply;
  final VoidCallback onReset;

  @override
  State<SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<SettingsSheet> {
  late double _conf;
  late double _alpha;
  late String _mode;

  @override
  void initState() {
    super.initState();
    _conf = widget.initial.yoloConfidence;
    _alpha = widget.initial.charLmAlpha;
    _mode = widget.initial.decodeMode;
  }

  @override
  Widget build(BuildContext context) {
    final isGreedy = _mode == 'greedy';
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text(
                  'Pipeline Settings',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                TextButton(
                  key: const Key('settings_reset_btn'),
                  onPressed: () {
                    setState(() {
                      _conf = AppSettings.defaults.yoloConfidence;
                      _alpha = AppSettings.defaults.charLmAlpha;
                      _mode = AppSettings.defaults.decodeMode;
                    });
                    widget.onReset();
                  },
                  child: const Text('Reset Defaults'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'YOLO Detection Confidence: ${_conf.toStringAsFixed(2)}',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            const Text(
              'Applies to next capture or explicit Reprocess. Detector confidence is not OCR certainty.',
              style: TextStyle(fontSize: 12, color: Colors.black54),
            ),
            Slider(
              key: const Key('settings_conf_slider'),
              value: _conf,
              min: 0.10,
              max: 0.90,
              divisions: 16,
              label: _conf.toStringAsFixed(2),
              onChanged: (v) => setState(() => _conf = (v * 20).round() / 20.0),
            ),
            const SizedBox(height: 8),
            const Text(
              'Decoder Mode',
              style: TextStyle(fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 6),
            SegmentedButton<String>(
              key: const Key('settings_decoder_mode'),
              segments: const [
                ButtonSegment(value: 'charlm_r1', label: Text('CharLM R1')),
                ButtonSegment(value: 'greedy', label: Text('Canonical Greedy')),
              ],
              selected: {_mode},
              onSelectionChanged: (s) => setState(() => _mode = s.first),
            ),
            const SizedBox(height: 12),
            Text(
              'Language-model weight (alpha): ${_alpha.toStringAsFixed(2)}${isGreedy ? ' (ignored in Greedy)' : ''}',
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: isGreedy ? Colors.black38 : Colors.black87,
              ),
            ),
            const Text(
              'Higher values favor likely Japanese words and can alter correct text or numbers. Beam width 5 / top-k 6 fixed. Alpha 0.00 runs plain beam (not greedy).',
              style: TextStyle(fontSize: 12, color: Colors.black54),
            ),
            Slider(
              key: const Key('settings_alpha_slider'),
              value: _alpha,
              min: 0.00,
              max: 0.60,
              divisions: 12,
              label: _alpha.toStringAsFixed(2),
              onChanged: isGreedy
                  ? null
                  : (v) => setState(() => _alpha = (v * 20).round() / 20.0),
            ),
            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                key: const Key('settings_apply_btn'),
                onPressed: () {
                  widget.onApply(AppSettings(
                    yoloConfidence: _conf,
                    charLmAlpha: _alpha,
                    decodeMode: _mode,
                  ));
                  Navigator.of(context).pop();
                },
                child: const Text('Save Settings'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
