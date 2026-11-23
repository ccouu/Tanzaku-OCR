/// Cached crop detail popup: crop preview, decoded Japanese OCR, reviewed whole-phrase
/// or component English, distinct Await / No lexicon match states, numeric disagreement flag,
/// and optional compact Suggestions drawer. Never triggers OCR on tap.
library;

import 'dart:convert';
import 'package:material_ui/material_ui.dart';
import '../models/contracts.dart';

class CropDetailPopup extends StatelessWidget {
  const CropDetailPopup({super.key, required this.crop});

  final CropResult crop;

  @override
  Widget build(BuildContext context) {
    final tr = crop.translation;
    final hasPreview =
        crop.cropPreviewPngB64 != null && crop.cropPreviewPngB64!.isNotEmpty;

    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Text(
                  'Line #${crop.cropIndex} (${crop.orientation}, det ${(crop.confidence * 100).toStringAsFixed(0)}%)',
                  style: const TextStyle(
                      fontSize: 15, fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                IconButton(
                  key: const Key('popup_dismiss_btn'),
                  icon: const Icon(Icons.close),
                  tooltip: 'Dismiss back to still image',
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ],
            ),
            if (hasPreview) ...[
              const SizedBox(height: 8),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Image.memory(
                  base64Decode(crop.cropPreviewPngB64!),
                  height: 76,
                  fit: BoxFit.contain,
                ),
              ),
            ],
            const SizedBox(height: 12),
            if (crop.status == 'failed') ...[
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Crop processing failed (isolated): ${crop.error ?? "unknown error"}',
                  style: TextStyle(color: Colors.red.shade900),
                ),
              ),
            ] else ...[
              const Text('Decoded Japanese (selected decoder)',
                  style: TextStyle(fontSize: 12, color: Colors.black54)),
              const SizedBox(height: 2),
              SelectableText(
                crop.decodedOcr.isEmpty ? '(No readable text)' : crop.decodedOcr,
                key: const Key('popup_jp_text'),
                style:
                    const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
              ),
              if (crop.numericDisagreement) ...[
                const SizedBox(height: 6),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.amber.shade50,
                    border: Border.all(color: Colors.amber.shade700),
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(
                    'Numeric disagreement with greedy OCR: greedy="${crop.greedyOcr}" vs decoded="${crop.decodedOcr}". Verify price visually.',
                    style:
                        TextStyle(fontSize: 12, color: Colors.amber.shade900),
                  ),
                ),
              ],
              if (crop.multiWindowFallback) ...[
                const SizedBox(height: 4),
                Text(
                  'Long line (${crop.windowCount} overlap512 windows): merged canonical greedy (joint LM unavailable across windows).',
                  style: const TextStyle(fontSize: 11, color: Colors.black54),
                ),
              ],
              const SizedBox(height: 12),
              const Text('English Translation',
                  style: TextStyle(fontSize: 12, color: Colors.black54)),
              const SizedBox(height: 2),
              SelectableText(
                tr?.primaryEnglish ?? 'No readable text',
                key: const Key('popup_en_text'),
                style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFF0F4C81)),
              ),
              if (tr != null && tr.components.isNotEmpty) ...[
                const SizedBox(height: 12),
                const Text('Components & Grammar Breakdown',
                    style: TextStyle(fontSize: 12, color: Colors.black54)),
                const SizedBox(height: 6),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: tr.components.map((c) {
                    final isPart = c.kind == 'particle';
                    final isAwait = c.kind == 'await_for_translation';
                    final isUnm = c.kind == 'unmatched' ||
                        c.kind == 'unmatched_with_suggestion';
                    final bg = isPart
                        ? Colors.blueGrey.shade50
                        : (isAwait
                            ? Colors.amber.shade50
                            : (isUnm
                                ? Colors.grey.shade200
                                : Colors.teal.shade50));
                    return Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: bg,
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(color: Colors.black12),
                      ),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '${c.text}${isPart ? " (particle)" : ""}',
                            style: const TextStyle(
                                fontWeight: FontWeight.w600, fontSize: 13),
                          ),
                          if (c.english.isNotEmpty)
                            Text(
                              c.english,
                              style: const TextStyle(
                                  fontSize: 11, color: Colors.black87),
                            ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ],
              if (tr != null && tr.suggestions.isNotEmpty) ...[
                const SizedBox(height: 12),
                ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  title: Text(
                    'Dictionary Suggestions (${tr.suggestions.length}) — not auto-applied',
                    style: const TextStyle(fontSize: 13, color: Colors.black54),
                  ),
                  children: tr.suggestions
                      .map((s) => ListTile(
                            dense: true,
                            contentPadding: EdgeInsets.zero,
                            title: Text(
                                '${s.sourceSpan} → ${s.suggestedTerm} (${s.suggestedEnglish})'),
                            subtitle: Text(s.matchType),
                          ))
                      .toList(),
                ),
              ],
            ],
            const SizedBox(height: 8),
            Text(
              'Route: ${crop.decoderRoute} • Cached result (tap does not rerun OCR)',
              style: const TextStyle(fontSize: 11, color: Colors.black45),
            ),
          ],
        ),
      ),
    );
  }
}
