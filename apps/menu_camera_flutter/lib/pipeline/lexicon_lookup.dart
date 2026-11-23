/// Shared Stage 4 Lexicon V3 Trie + Protected Spans + Particle Guard + Read-Only Translation Lookup.
///
/// Exact Dart port of the core segmentation & translation formatting semantics in
/// `src/recognition/lexicon_lookup/` and `src/stage4/desktop_backend.py`:
///   - O(L) exact-match pre-pass on the 12,471-entry Lexicon V3 index (never starved by fuzzy DFS)
///   - Protected number/price spans (`1800円`, `¥1,280`)
///   - Particle protection (`でのをにとはがもや` — preserves `季節/の/先付/と/前菜` while protecting compounds
///     like `胡麻豆腐`, `親子丼`, `幕の内`)
///   - Auto-correct OFF; read-only SQLite overrides & reviewed phrase plans via `TranslationStore`.
library;

import 'dart:convert';

import '../models/contracts.dart';

const Set<String> kParticleChars = {'で', 'の', 'を', 'に', 'と', 'は', 'が', 'も', 'や'};
final RegExp _protectedPriceRe = RegExp(r'[¥￥]?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?(?:円|税込|税抜)?');
final RegExp _symbolOnlyRe = RegExp(r'^[\s0-9.,¥￥円()（）/\-~〜・&＆+%!！?？:：]+$');

class TranslationRecord {
  const TranslationRecord({
    required this.canonicalJp,
    required this.nameEn,
    this.readingHiragana = '',
    this.dbId,
  });

  final String canonicalJp;
  final String nameEn;
  final String readingHiragana;
  final int? dbId;
}

class PhrasePlanRecord {
  const PhrasePlanRecord({
    required this.canonicalPhrase,
    required this.phraseEn,
    this.provenanceSource,
  });

  final String canonicalPhrase;
  final String phraseEn;
  final String? provenanceSource;
}

/// Abstract read-only translation store (implemented by native iOS SQLite or in-memory test store).
abstract class ReadOnlyTranslationStore {
  Future<Map<String, TranslationRecord>> lookupTerms(Set<String> terms);
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase);
}

class InMemoryTranslationStore implements ReadOnlyTranslationStore {
  InMemoryTranslationStore({
    Map<String, TranslationRecord>? terms,
    Map<String, PhrasePlanRecord>? phrasePlans,
  })  : _terms = terms ?? const {},
        _phrasePlans = phrasePlans ?? const {};

  final Map<String, TranslationRecord> _terms;
  final Map<String, PhrasePlanRecord> _phrasePlans;

  @override
  Future<Map<String, TranslationRecord>> lookupTerms(Set<String> terms) async {
    return {
      for (final t in terms)
        if (_terms.containsKey(t)) t: _terms[t]!,
    };
  }

  @override
  Future<PhrasePlanRecord?> lookupPhrasePlan(String canonicalPhrase) async =>
      _phrasePlans[canonicalPhrase];
}

class _TrieNode {
  final Map<int, _TrieNode> children = {};
  String? terminalTerm;
  int? dbId;
}

class LexiconV3Index {
  LexiconV3Index._(this._root, this.entryCount);

  factory LexiconV3Index.fromJsonString(String jsonText) {
    final obj = jsonDecode(jsonText) as Map<String, dynamic>;
    final entries = (obj['entries'] as List?) ?? const [];
    final root = _TrieNode();
    var count = 0;
    for (final raw in entries) {
      if (raw is! Map) continue;
      final term = (raw['canonical_term'] ?? raw['display_term'] ?? '').toString().trim();
      if (term.isEmpty) continue;
      var cur = root;
      for (final cp in term.runes) {
        cur = cur.children.putIfAbsent(cp, _TrieNode.new);
      }
      cur.terminalTerm = term;
      cur.dbId = (raw['db_id'] as num?)?.toInt();
      count++;
    }
    return LexiconV3Index._(root, count);
  }

  factory LexiconV3Index.fromTerms(Iterable<String> terms) {
    final root = _TrieNode();
    var count = 0;
    for (final raw in terms) {
      final term = raw.trim();
      if (term.isEmpty) continue;
      var cur = root;
      for (final cp in term.runes) {
        cur = cur.children.putIfAbsent(cp, _TrieNode.new);
      }
      cur.terminalTerm = term;
      count++;
    }
    return LexiconV3Index._(root, count);
  }

  final _TrieNode _root;
  final int entryCount;

  /// Find all exact lexicon matches starting at Unicode code-point index `start` in `cps`.
  List<ExactSpan> exactMatchesAt(List<int> cps, int start) {
    final out = <ExactSpan>[];
    var cur = _root;
    for (var i = start; i < cps.length; i++) {
      final nxt = cur.children[cps[i]];
      if (nxt == null) break;
      cur = nxt;
      if (cur.terminalTerm != null) {
        out.add(ExactSpan(start, i + 1, cur.terminalTerm!, cur.dbId));
      }
    }
    return out;
  }
}

class ExactSpan {
  const ExactSpan(this.start, this.end, this.term, this.dbId);
  final int start;
  final int end;
  final String term;
  final int? dbId;
}

class LexiconSegment {
  const LexiconSegment(this.start, this.end, this.text, this.kind, {this.dbId});
  final int start;
  final int end;
  final String text;
  final String kind; // 'exact' | 'protected' | 'particle' | 'symbol' | 'unmatched' | 'whitespace'
  final int? dbId;
}

class LexiconV3Pipeline {
  LexiconV3Pipeline(this.index, this.store);

  final LexiconV3Index index;
  final ReadOnlyTranslationStore store;

  /// Segment `decodedOcr` using bounded DP prioritizing longest exact lexicon matches,
  /// protected price/number spans, and particles between exact terms (preserving compounds).
  List<LexiconSegment> segment(String decodedOcr) {
    final cps = decodedOcr.runes.toList();
    final n = cps.length;
    if (n == 0) return const [];

    String slice(int a, int b) => String.fromCharCodes(cps.sublist(a, b));

    // Precompute protected price/numeric spans by code-point index
    final protectedAt = <int, int>{};
    // Map UTF-16 offset -> code-point offset
    final cuToCp = <int, int>{};
    var cuIdx = 0;
    for (var cpIdx = 0; cpIdx < n; cpIdx++) {
      cuToCp[cuIdx] = cpIdx;
      cuIdx += cps[cpIdx] > 0xFFFF ? 2 : 1;
    }
    cuToCp[cuIdx] = n;
    for (final m in _protectedPriceRe.allMatches(decodedOcr)) {
      final s = cuToCp[m.start];
      final e = cuToCp[m.end];
      if (s != null && e != null && e > s) {
        protectedAt[s] = e;
      }
    }

    // DP: maximize coverage score (prefer longer exact lexicon matches over split pieces)
    final dpScore = List<double>.filled(n + 1, -1e9);
    final dpChoice = List<LexiconSegment?>.filled(n + 1, null);
    dpScore[n] = 0.0;

    for (var i = n - 1; i >= 0; i--) {
      final ch = String.fromCharCode(cps[i]);
      // 1. Whitespace (always classify as whitespace, never symbol)
      if (ch.trim().isEmpty) {
        dpScore[i] = dpScore[i + 1];
        dpChoice[i] = LexiconSegment(i, i + 1, ch, 'whitespace');
        continue;
      }
      // 2. Exact lexicon matches (prefer longest whole compound: length^2 bonus!)
      for (final m in index.exactMatchesAt(cps, i)) {
        final len = m.end - m.start;
        final sc = (len * 10.0 + len * len * 1.5) + dpScore[m.end];
        if (sc > dpScore[i]) {
          dpScore[i] = sc;
          dpChoice[i] = LexiconSegment(m.start, m.end, m.term, 'exact', dbId: m.dbId);
        }
      }
      // 3. Protected number/price span
      final pEnd = protectedAt[i];
      if (pEnd != null) {
        final len = pEnd - i;
        final sc = (len * 9.0 + len * len * 1.2) + dpScore[pEnd];
        if (sc > dpScore[i]) {
          dpScore[i] = sc;
          dpChoice[i] = LexiconSegment(i, pEnd, slice(i, pEnd), 'protected');
        }
      }
      // 4. Particle (single char between terms)
      if (kParticleChars.contains(ch)) {
        final sc = 2.0 + dpScore[i + 1];
        if (sc > dpScore[i]) {
          dpScore[i] = sc;
          dpChoice[i] = LexiconSegment(i, i + 1, ch, 'particle');
        }
      }
      // 5. Symbol or unmatched single char (merged afterwards)
      final fallbackKind = _symbolOnlyRe.hasMatch(ch) ? 'symbol' : 'unmatched';
      final sc = (fallbackKind == 'symbol' ? 1.0 : -5.0) + dpScore[i + 1];
      if (sc > dpScore[i]) {
        dpScore[i] = sc;
        dpChoice[i] = LexiconSegment(i, i + 1, ch, fallbackKind);
      }
    }

    // Reconstruct & merge adjacent unmatched spans
    final raw = <LexiconSegment>[];
    var pos = 0;
    while (pos < n) {
      final seg = dpChoice[pos]!;
      raw.add(seg);
      pos = seg.end;
    }
    final merged = <LexiconSegment>[];
    for (final s in raw) {
      if (merged.isNotEmpty &&
          merged.last.kind == 'unmatched' &&
          s.kind == 'unmatched') {
        final prev = merged.removeLast();
        merged.add(LexiconSegment(prev.start, s.end, '${prev.text}${s.text}', 'unmatched'));
      } else {
        merged.add(s);
      }
    }
    return merged;
  }

  Future<CropTranslation> lookupAndFormat(String decodedOcr) async {
    if (decodedOcr.trim().isEmpty) {
      return const CropTranslation(
        primaryEnglish: 'No readable text',
        reviewedWholePhraseEn: null,
        translationStatus: 'no_readable_text',
        components: [],
        suggestions: [],
      );
    }

    final segs = segment(decodedOcr);
    final exactTerms = {
      for (final s in segs)
        if (s.kind == 'exact') s.text,
    };
    final trMap = await store.lookupTerms(exactTerms);
    final phrasePlan = await store.lookupPhrasePlan(decodedOcr);

    final components = <TranslationComponent>[];
    final compEnParts = <String>[];
    var hasExactEn = false;
    var hasExactAwait = false;
    var hasUnmatched = false;
    var hasProtected = false;

    for (final seg in segs) {
      if (seg.kind == 'whitespace') continue;
      if (seg.kind == 'protected') {
        hasProtected = true;
        components.add(TranslationComponent(
            text: seg.text, kind: 'number_or_price', english: seg.text));
        compEnParts.add(seg.text);
        continue;
      }
      if (seg.kind == 'exact') {
        final tr = trMap[seg.text];
        final en = (tr?.nameEn ?? '').trim();
        if (en.isNotEmpty) {
          hasExactEn = true;
          components.add(TranslationComponent(
            text: seg.text,
            kind: 'exact',
            english: en,
            reading: tr?.readingHiragana ?? '',
          ));
          compEnParts.add(en);
        } else {
          hasExactAwait = true;
          components.add(TranslationComponent(
            text: seg.text,
            kind: 'await_for_translation',
            english: 'Await for translation',
            reading: tr?.readingHiragana ?? '',
          ));
          compEnParts.add('Await for translation');
        }
        continue;
      }
      if (seg.kind == 'particle') {
        components.add(TranslationComponent(text: seg.text, kind: 'particle', english: ''));
        continue;
      }
      if (seg.kind == 'symbol') {
        components.add(TranslationComponent(text: seg.text, kind: 'symbol', english: seg.text));
        compEnParts.add(seg.text);
        continue;
      }
      hasUnmatched = true;
      components.add(TranslationComponent(
          text: seg.text, kind: 'unmatched', english: 'No lexicon match'));
      compEnParts.add('[${seg.text}]');
    }

    String status;
    String fallbackEn;
    if (hasExactEn && !hasExactAwait && !hasUnmatched) {
      status = 'exact_db_english';
      fallbackEn = compEnParts.join(' / ');
    } else if (hasExactAwait && !hasExactEn && !hasUnmatched) {
      status = 'await_for_translation';
      fallbackEn = 'Await for translation';
    } else if (hasExactEn || hasExactAwait) {
      status = 'partial_segment_english';
      fallbackEn = compEnParts.join(' · ');
    } else if (hasUnmatched) {
      status = 'no_lexicon_match';
      fallbackEn = 'No lexicon match';
    } else if (hasProtected) {
      status = 'numeric_or_symbol_only';
      fallbackEn = compEnParts.join(' ');
    } else {
      status = 'no_lexicon_match';
      fallbackEn = 'No lexicon match';
    }

    final reviewedEn = phrasePlan?.phraseEn;
    return CropTranslation(
      primaryEnglish: (reviewedEn != null && reviewedEn.isNotEmpty) ? reviewedEn : fallbackEn,
      reviewedWholePhraseEn: reviewedEn,
      translationStatus: status,
      components: components,
      suggestions: const [],
    );
  }
}
