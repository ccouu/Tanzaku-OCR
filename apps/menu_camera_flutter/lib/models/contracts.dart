/// Shared Stage 4 data contracts used by both DesktopReferenceBackend and future IosCoreMLBackend.
library;

class AppSettings {
  const AppSettings({
    this.yoloConfidence = 0.35,
    this.charLmAlpha = 0.30,
    this.decodeMode = 'charlm_r1',
  });

  static const AppSettings defaults = AppSettings();

  final double yoloConfidence;
  final double charLmAlpha;
  final String decodeMode; // 'charlm_r1' | 'greedy'

  AppSettings copyWith({
    double? yoloConfidence,
    double? charLmAlpha,
    String? decodeMode,
  }) {
    return AppSettings(
      yoloConfidence: yoloConfidence ?? this.yoloConfidence,
      charLmAlpha: charLmAlpha ?? this.charLmAlpha,
      decodeMode: decodeMode ?? this.decodeMode,
    );
  }

  @override
  bool operator ==(Object other) =>
      other is AppSettings &&
      (other.yoloConfidence - yoloConfidence).abs() < 1e-6 &&
      (other.charLmAlpha - charLmAlpha).abs() < 1e-6 &&
      other.decodeMode == decodeMode;

  @override
  int get hashCode => Object.hash(
        (yoloConfidence * 100).round(),
        (charLmAlpha * 100).round(),
        decodeMode,
      );
}

class CapturedPageMeta {
  const CapturedPageMeta({
    required this.captureId,
    required this.filename,
    required this.captureSource,
    required this.streamWidth,
    required this.streamHeight,
    required this.savedWidth,
    required this.savedHeight,
    required this.streamSavedParity,
    required this.imageSha256,
    this.streamRotation = 0,
    this.exifOrientation,
    this.streamSavedParityMode = 'exact',
  });

  final String captureId;
  final String filename;
  final String captureSource;
  final int streamWidth;
  final int streamHeight;
  final int savedWidth;
  final int savedHeight;
  final bool streamSavedParity;
  final String imageSha256;
  final int streamRotation;
  final int? exifOrientation;
  final String streamSavedParityMode;

  factory CapturedPageMeta.fromJson(Map<String, dynamic> j) => CapturedPageMeta(
        captureId: j['capture_id'] as String? ?? '',
        filename: j['filename'] as String? ?? '',
        captureSource: j['capture_source'] as String? ?? 'import',
        streamWidth: (j['stream_width'] as num?)?.toInt() ?? 0,
        streamHeight: (j['stream_height'] as num?)?.toInt() ?? 0,
        savedWidth: (j['saved_width'] as num?)?.toInt() ?? 0,
        savedHeight: (j['saved_height'] as num?)?.toInt() ?? 0,
        streamSavedParity: j['stream_saved_parity'] as bool? ?? false,
        imageSha256: j['image_sha256'] as String? ?? '',
        streamRotation: (j['stream_rotation'] as num?)?.toInt() ?? 0,
        exifOrientation: (j['exif_orientation'] as num?)?.toInt(),
        streamSavedParityMode:
            j['stream_saved_parity_mode'] as String? ?? 'exact',
      );
}

class StageTimings {
  const StageTimings({
    required this.coldLoadMs,
    required this.decodeNormalizeMs,
    required this.persistMs,
    required this.detectMs,
    required this.postprocessMs,
    required this.recognizeMs,
    required this.decodeMs,
    required this.lookupMs,
    required this.firstResultMs,
    required this.totalWallMs,
    required this.summedCropWorkMs,
    this.requestReadMs = 0,
    this.requestIngestMs = 0,
    this.queueWaitMs = 0,
    this.serverPipelineWallMs,
    this.serverRequestTotalMs,
  });

  final double coldLoadMs;
  final double decodeNormalizeMs;
  final double persistMs;
  final double detectMs;
  final double postprocessMs;
  final double recognizeMs;
  final double decodeMs;
  final double lookupMs;
  final double? firstResultMs;
  final double? totalWallMs;
  final double summedCropWorkMs;
  final double requestReadMs;
  final double requestIngestMs;
  final double queueWaitMs;
  final double? serverPipelineWallMs;
  final double? serverRequestTotalMs;

  factory StageTimings.fromJson(Map<String, dynamic> j) => StageTimings(
        coldLoadMs: (j['cold_load_ms'] as num?)?.toDouble() ?? 0,
        decodeNormalizeMs: (j['decode_normalize_ms'] as num?)?.toDouble() ?? 0,
        persistMs: (j['persist_ms'] as num?)?.toDouble() ?? 0,
        detectMs: (j['detect_ms'] as num?)?.toDouble() ?? 0,
        postprocessMs: (j['postprocess_ms'] as num?)?.toDouble() ?? 0,
        recognizeMs: (j['recognize_ms'] as num?)?.toDouble() ?? 0,
        decodeMs: (j['decode_ms'] as num?)?.toDouble() ?? 0,
        lookupMs: (j['lookup_ms'] as num?)?.toDouble() ?? 0,
        firstResultMs: (j['first_result_ms'] as num?)?.toDouble(),
        totalWallMs: (j['total_wall_ms'] as num?)?.toDouble(),
        summedCropWorkMs: (j['summed_crop_work_ms'] as num?)?.toDouble() ?? 0,
        requestReadMs: (j['request_read_ms'] as num?)?.toDouble() ?? 0,
        requestIngestMs: (j['request_ingest_ms'] as num?)?.toDouble() ?? 0,
        queueWaitMs: (j['queue_wait_ms'] as num?)?.toDouble() ?? 0,
        serverPipelineWallMs:
            (j['server_pipeline_wall_ms'] as num?)?.toDouble(),
        serverRequestTotalMs:
            (j['server_request_total_ms'] as num?)?.toDouble() ??
                (j['total_wall_ms'] as num?)?.toDouble(),
      );
}

class CropCounts {
  const CropCounts({
    required this.rawDetections,
    required this.postprocessedLines,
    required this.completedCrops,
    required this.failedCrops,
    required this.totalWindows,
    required this.cropsTruncated,
  });

  final int rawDetections;
  final int postprocessedLines;
  final int completedCrops;
  final int failedCrops;
  final int totalWindows;
  final bool cropsTruncated;

  factory CropCounts.fromJson(Map<String, dynamic> j) => CropCounts(
        rawDetections: (j['raw_detections'] as num?)?.toInt() ?? 0,
        postprocessedLines: (j['postprocessed_lines'] as num?)?.toInt() ?? 0,
        completedCrops: (j['completed_crops'] as num?)?.toInt() ?? 0,
        failedCrops: (j['failed_crops'] as num?)?.toInt() ?? 0,
        totalWindows: (j['total_windows'] as num?)?.toInt() ?? 0,
        cropsTruncated: j['crops_truncated'] as bool? ?? false,
      );
}

class TranslationComponent {
  const TranslationComponent({
    required this.text,
    required this.kind,
    required this.english,
    this.reading = '',
  });

  final String text;
  final String kind;
  final String english;
  final String reading;

  factory TranslationComponent.fromJson(Map<String, dynamic> j) =>
      TranslationComponent(
        text: j['text'] as String? ?? '',
        kind: j['kind'] as String? ?? 'unknown',
        english: j['english'] as String? ?? '',
        reading: j['reading'] as String? ?? '',
      );
}

class FuzzySuggestion {
  const FuzzySuggestion({
    required this.sourceSpan,
    required this.suggestedTerm,
    required this.suggestedEnglish,
    required this.matchType,
  });

  final String sourceSpan;
  final String suggestedTerm;
  final String suggestedEnglish;
  final String matchType;

  factory FuzzySuggestion.fromJson(Map<String, dynamic> j) => FuzzySuggestion(
        sourceSpan: j['source_span'] as String? ?? '',
        suggestedTerm: j['suggested_term'] as String? ?? '',
        suggestedEnglish: j['suggested_english'] as String? ?? '',
        matchType: j['match_type'] as String? ?? '',
      );
}

class CropTranslation {
  const CropTranslation({
    required this.primaryEnglish,
    required this.reviewedWholePhraseEn,
    required this.translationStatus,
    required this.components,
    required this.suggestions,
  });

  final String primaryEnglish;
  final String? reviewedWholePhraseEn;
  final String translationStatus;
  final List<TranslationComponent> components;
  final List<FuzzySuggestion> suggestions;

  factory CropTranslation.fromJson(Map<String, dynamic> j) => CropTranslation(
        primaryEnglish: j['primary_english'] as String? ?? '',
        reviewedWholePhraseEn: j['reviewed_whole_phrase_en'] as String?,
        translationStatus: j['translation_status'] as String? ?? 'no_lexicon_match',
        components: ((j['components'] as List?) ?? const [])
            .map((e) => TranslationComponent.fromJson(e as Map<String, dynamic>))
            .toList(),
        suggestions: ((j['suggestions'] as List?) ?? const [])
            .map((e) => FuzzySuggestion.fromJson(e as Map<String, dynamic>))
            .toList(),
      );
}

class CropResult {
  const CropResult({
    required this.cropId,
    required this.cropIndex,
    required this.status,
    required this.confidence,
    required this.orientation,
    required this.polygon,
    required this.windowCount,
    required this.multiWindowFallback,
    required this.cropPreviewPngB64,
    required this.greedyOcr,
    required this.decodedOcr,
    required this.decoderRoute,
    required this.numericDisagreement,
    required this.translation,
    required this.error,
  });

  final String cropId;
  final int cropIndex;
  final String status;
  final double confidence;
  final String orientation;
  final List<List<double>> polygon;
  final int windowCount;
  final bool multiWindowFallback;
  final String? cropPreviewPngB64;
  final String greedyOcr;
  final String decodedOcr;
  final String decoderRoute;
  final bool numericDisagreement;
  final CropTranslation? translation;
  final String? error;

  factory CropResult.fromJson(Map<String, dynamic> j) => CropResult(
        cropId: j['crop_id'] as String? ?? '',
        cropIndex: (j['crop_index'] as num?)?.toInt() ?? 0,
        status: j['status'] as String? ?? 'failed',
        confidence: (j['confidence'] as num?)?.toDouble() ?? 0,
        orientation: j['orientation'] as String? ?? 'horizontal',
        polygon: ((j['polygon'] as List?) ?? const [])
            .map((pt) => (pt as List).map((v) => (v as num).toDouble()).toList())
            .toList(),
        windowCount: (j['window_count'] as num?)?.toInt() ?? 1,
        multiWindowFallback: j['multi_window_fallback'] as bool? ?? false,
        cropPreviewPngB64: j['crop_preview_png_b64'] as String?,
        greedyOcr: j['greedy_ocr'] as String? ?? '',
        decodedOcr: j['decoded_ocr'] as String? ?? '',
        decoderRoute: j['decoder_route'] as String? ?? '',
        numericDisagreement: j['numeric_disagreement'] as bool? ?? false,
        translation: j['translation'] != null
            ? CropTranslation.fromJson(j['translation'] as Map<String, dynamic>)
            : null,
        error: j['error'] as String?,
      );
}

class PageRun {
  const PageRun({
    required this.runId,
    required this.captureId,
    required this.configId,
    required this.backendId,
    required this.deviceName,
    required this.precision,
    required this.bundleId,
    required this.status,
    required this.stage,
    required this.cacheStatus,
    required this.coldOrWarm,
    required this.isReprocess,
    required this.capture,
    required this.counts,
    required this.timingsMs,
    required this.crops,
    required this.error,
    this.reusedCachedLogits = false,
    this.reusedCachedDetections = false,
  });

  final String runId;
  final String captureId;
  final String configId;
  final String backendId;
  final String deviceName;
  final String precision;
  final String bundleId;
  final String status; // 'running' | 'completed' | 'failed' | 'cancelled'
  final String stage;
  final String cacheStatus;
  final String coldOrWarm;
  final bool isReprocess;
  final CapturedPageMeta capture;
  final CropCounts counts;
  final StageTimings timingsMs;
  final List<CropResult> crops;
  final String? error;
  final bool reusedCachedLogits;
  final bool reusedCachedDetections;

  bool get isTerminal =>
      status == 'completed' || status == 'failed' || status == 'cancelled';

  factory PageRun.fromJson(Map<String, dynamic> j) => PageRun(
        runId: j['run_id'] as String? ?? '',
        captureId: j['capture_id'] as String? ?? '',
        configId: j['config_id'] as String? ?? '',
        backendId: j['backend_id'] as String? ?? '',
        deviceName: j['device_name'] as String? ?? '',
        precision: j['precision'] as String? ?? 'fp32',
        bundleId: j['bundle_id'] as String? ?? '',
        status: j['status'] as String? ?? 'running',
        stage: j['stage'] as String? ?? '',
        cacheStatus: j['cache_status'] as String? ?? '',
        coldOrWarm: j['cold_or_warm'] as String? ?? '',
        isReprocess: j['is_reprocess'] as bool? ?? false,
        reusedCachedLogits: j['reused_cached_logits'] as bool? ??
            (j['cache_status'] == 'reused_cached_logits'),
        reusedCachedDetections: j['reused_cached_detections'] as bool? ??
            (j['cache_status'] == 'reused_cached_logits' ||
                j['cache_status'] == 'reused_raw_detections_fresh_ocr'),
        capture: CapturedPageMeta.fromJson(
            (j['capture'] as Map<String, dynamic>?) ?? const {}),
        counts: CropCounts.fromJson(
            (j['counts'] as Map<String, dynamic>?) ?? const {}),
        timingsMs: StageTimings.fromJson(
            (j['timings_ms'] as Map<String, dynamic>?) ?? const {}),
        crops: ((j['crops'] as List?) ?? const [])
            .map((e) => CropResult.fromJson(e as Map<String, dynamic>))
            .toList(),
        error: j['error'] as String?,
      );
}
