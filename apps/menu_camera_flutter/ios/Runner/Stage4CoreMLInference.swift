// Stage 4 Native iOS Core ML Inference Engine:
//   - V5 YOLO11n-OBB (1024x1024 letterbox -> raw [1, 7, 21504] Float32)
//   - Exact Split Recognizer:
//       1) RecognizerFeatureExtractor_fp32.mlmodelc on physical replicate-padded bucket [128,256,384,512]
//       2) Host valid_T = (validW + 3) / 4 prefix slice BEFORE backward LSTM
//       3) RecognizerRecurrentHead_fp32.mlmodelc on [1, 256, valid_T] -> [valid_T, 1, 2106] Float32
//   - CoreImage perspective rectification + once-only 90° CCW vertical rotation + H64 aspect strip.

import CoreGraphics
import CoreImage
import CoreML
import Foundation
import UIKit

final class Stage4CoreMLInference {
    static let bundleId = "stage4_step1000_raw_2105_v5_lmv2_lexv3_20261123"
    static let detectorImgSz = 1024
    static let recognizerBuckets: [Int] = [128, 256, 384, 512]
    static let imageNetMean: [Float] = [0.485, 0.456, 0.406]
    static let imageNetStd: [Float] = [0.229, 0.224, 0.225]

    private var detectorModel: MLModel?
    private var detectorCPUOnly = false
    private(set) var lastDetectorDiagnostics: [String: Any] = [:]
    private var featExtractorModel: MLModel?
    private var recurrentHeadModel: MLModel?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private(set) var modelsLoaded = false

    func bundledResourceURL(_ filename: String) -> URL? {
        let ns = filename as NSString
        let name = ns.deletingPathExtension
        let ext = ns.pathExtension
        if let u = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: "Stage4Bundle") {
            return u
        }
        return Bundle.main.url(forResource: name, withExtension: ext)
    }

    func missingRequiredResources() -> [String] {
        let required = [
            "V5DetectorOBB1024_fp32.mlmodelc",
            "RecognizerFeatureExtractor_fp32.mlmodelc",
            "RecognizerRecurrentHead_fp32.mlmodelc",
            "target_vocab_active.json",
            "char_lm_4g_hp1010_v2.json.gz",
            "lexicon_index_v3_22112026.json.gz",
            "culinary_menu.db",
        ]
        return required.filter { bundledResourceURL($0) == nil }
    }

    private func loadDetectorIfNeeded(cpuOnly: Bool = false) throws {
        if detectorModel != nil && detectorCPUOnly == cpuOnly { return }
        guard let url = bundledResourceURL("V5DetectorOBB1024_fp32.mlmodelc") else {
            throw Stage4PluginError.assetsNotBundled(missing: ["V5DetectorOBB1024_fp32.mlmodelc"])
        }
        let config = MLModelConfiguration()
        config.computeUnits = cpuOnly ? .cpuOnly : .all
        detectorModel = try MLModel(contentsOf: url, configuration: config)
        detectorCPUOnly = cpuOnly
    }

    func loadModelsIfNeeded() throws {
        if modelsLoaded { return }
        let missing = missingRequiredResources()
        if !missing.isEmpty {
            throw Stage4PluginError.assetsNotBundled(missing: missing)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all

        guard let featURL = bundledResourceURL("RecognizerFeatureExtractor_fp32.mlmodelc"),
              let headURL = bundledResourceURL("RecognizerRecurrentHead_fp32.mlmodelc") else {
            throw Stage4PluginError.assetsNotBundled(missing: missingRequiredResources())
        }
        try loadDetectorIfNeeded()
        featExtractorModel = try MLModel(contentsOf: featURL, configuration: cfg)
        recurrentHeadModel = try MLModel(contentsOf: headURL, configuration: cfg)
        modelsLoaded = true
    }

    /// Letterbox RGB image to 1024x1024 (pad 114/255) and run V5 YOLO11n-OBB Core ML model.
    func runDetectorOBB1024(image: UIImage, cpuOnly: Bool = false, captureDiagnostics: Bool = false) throws -> (rawFloatBytes: Data, numAnchors: Int, meta: [String: Double]) {
        let start = ProcessInfo.processInfo.systemUptime
        lastDetectorDiagnostics = ["stage": "load_detector", "precision": "fp32", "compute_units": cpuOnly ? "cpuOnly" : "all", "ui_image_orientation": image.imageOrientation.rawValue]
        try loadDetectorIfNeeded(cpuOnly: cpuOnly)
        lastDetectorDiagnostics["model_load_ms"] = (ProcessInfo.processInfo.systemUptime - start) * 1000
        lastDetectorDiagnostics["stage"] = "prepare_image"
        guard let model = detectorModel, let cg = image.cgImage else {
            throw Stage4PluginError.inferenceFailure("Detector model or CGImage unavailable.")
        }

        lastDetectorDiagnostics["source_width"] = cg.width
        lastDetectorDiagnostics["source_height"] = cg.height
        lastDetectorDiagnostics["input_shape"] = [1, 3, 1024, 1024]
        if captureDiagnostics { lastDetectorDiagnostics["source_preview_png_b64"] = image.pngData()?.base64EncodedString() ?? "" }
        let srcW = Double(cg.width)
        let srcH = Double(cg.height)
        let target = Double(Self.detectorImgSz)
        let gain = min(target / srcW, target / srcH)
        let newW = Int((srcW * gain).rounded())
        let newH = Int((srcH * gain).rounded())
        let left = Int(((target - Double(newW)) / 2.0 - 0.1).rounded())
        let top = Int(((target - Double(newH)) / 2.0 - 0.1).rounded())

        let sz = Self.detectorImgSz
        var rgba = [UInt8](repeating: 114, count: sz * sz * 4)
        for i in stride(from: 3, to: rgba.count, by: 4) { rgba[i] = 255 }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { ptr in
            if let ctx = CGContext(
                data: ptr.baseAddress,
                width: sz,
                height: sz,
                bitsPerComponent: 8,
                bytesPerRow: sz * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) {
                ctx.interpolationQuality = .medium
                // CoreGraphics origin is bottom-left; flip y to match top-left PIL letterbox
                ctx.draw(cg, in: CGRect(x: left, y: sz - top - newH, width: newW, height: newH))
            }
        }

        let inputArr = try MLMultiArray(shape: [1, 3, NSNumber(value: sz), NSNumber(value: sz)], dataType: .float32)
        let dst = inputArr.dataPointer.bindMemory(to: Float.self, capacity: 3 * sz * sz)
        let plane = sz * sz
        for p in 0..<plane {
            let base = p * 4
            dst[p] = Float(rgba[base]) / 255.0
            dst[plane + p] = Float(rgba[base + 1]) / 255.0
            dst[2 * plane + p] = Float(rgba[base + 2]) / 255.0
        }

        if captureDiagnostics {
            let provider = CGDataProvider(data: Data(rgba) as CFData)
            if let provider = provider, let preview = CGImage(width: sz, height: sz, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: sz * 4, space: colorSpace, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) {
                lastDetectorDiagnostics["detector_input_png_b64"] = UIImage(cgImage: preview).pngData()?.base64EncodedString() ?? ""
            }
        }
        lastDetectorDiagnostics["model_resource"] = "V5DetectorOBB1024_fp32.mlmodelc"
        lastDetectorDiagnostics["app_version"] = Bundle.main.infoDictionary?["CFBundleShortVersionString"] ?? "unknown"
        lastDetectorDiagnostics["build_number"] = Bundle.main.infoDictionary?["CFBundleVersion"] ?? "unknown"
        lastDetectorDiagnostics["os_version"] = UIDevice.current.systemVersion
        lastDetectorDiagnostics["device"] = UIDevice.current.model
        lastDetectorDiagnostics["input_expected_range"] = [0.0, 1.0]
        if captureDiagnostics {
            var sums = [Double](repeating: 0, count: 3)
            var lo: Float = 1
            var hi: Float = 0
            for c in 0..<3 {
                for i in 0..<plane {
                    let v = dst[c * plane + i]
                    sums[c] += Double(v)
                    lo = min(lo, v)
                    hi = max(hi, v)
                }
            }
            lastDetectorDiagnostics["input_actual_min_max"] = [Double(lo), Double(hi)]
            lastDetectorDiagnostics["input_channel_means_rgb"] = sums.map { $0 / Double(plane) }
        }
        lastDetectorDiagnostics["letterbox"] = ["gain": gain, "pad_x": Double(left), "pad_y": Double(top)]
        lastDetectorDiagnostics["stage"] = "coreml_predict"
        let predictStart = ProcessInfo.processInfo.systemUptime
        let provider = try MLDictionaryFeatureProvider(dictionary: ["image_rgb01": MLFeatureValue(multiArray: inputArr)])
        let out = try model.prediction(from: provider)
        guard let rawArr = out.featureValue(for: "raw_obb")?.multiArrayValue else {
            throw Stage4PluginError.inferenceFailure("Detector output 'raw_obb' missing.")
        }
        lastDetectorDiagnostics["predict_ms"] = (ProcessInfo.processInfo.systemUptime - predictStart) * 1000
        lastDetectorDiagnostics["stage"] = "read_output"
        let shape = rawArr.shape.map { $0.intValue }
        lastDetectorDiagnostics["output_shape"] = shape
        lastDetectorDiagnostics["output_strides"] = rawArr.strides.map { $0.intValue }
        lastDetectorDiagnostics["output_dtype"] = String(describing: rawArr.dataType)
        guard shape.count == 3, shape[0] == 1, shape[1] == 7, shape[2] > 0 else {
            throw Stage4PluginError.inferenceFailure("Unexpected detector output shape: \(shape)")
        }
        let numAnchors = shape[2]
        let count = 7 * numAnchors
        let floatData = multiArrayToContiguousFloat32Data(rawArr, count: count)
        floatData.withUnsafeBytes { bytes in
            let values = bytes.bindMemory(to: Float.self)
            var finiteCount = 0
            var maxConfidence: Float = 0
            var above = [0, 0, 0]
            for value in values { if value.isFinite { finiteCount += 1 } }
            for i in 0..<numAnchors {
                let conf = max(values[4 * numAnchors + i], values[5 * numAnchors + i])
                if conf.isFinite {
                    maxConfidence = max(maxConfidence, conf)
                    if conf >= 0.05 { above[0] += 1 }
                    if conf >= 0.15 { above[1] += 1 }
                    if conf >= 0.35 { above[2] += 1 }
                }
            }
            lastDetectorDiagnostics["finite_output_values"] = finiteCount
            lastDetectorDiagnostics["nonfinite_output_values"] = count - finiteCount
            lastDetectorDiagnostics["max_class_confidence"] = Double(maxConfidence)
            lastDetectorDiagnostics["candidates_ge_005_015_035"] = above
        }
        lastDetectorDiagnostics["stage"] = "detector_complete"
        lastDetectorDiagnostics["detector_total_ms"] = (ProcessInfo.processInfo.systemUptime - start) * 1000
        return (
            floatData,
            numAnchors,
            ["gain": gain, "pad_x": Double(left), "pad_y": Double(top), "src_w": srcW, "src_h": srcH]
        )
    }

    /// Rectify one OBB polygon -> upright crop + H64 strip -> split Core ML recognizer -> raw FP32 logits.
    func recognizeCrop(
        sourceImage: UIImage,
        polygon: [[Double]],
        orientationFlag: Int
    ) throws -> [String: Any] {
        try loadModelsIfNeeded()
        guard let featModel = featExtractorModel,
              let headModel = recurrentHeadModel,
              let cg = sourceImage.cgImage,
              polygon.count >= 4 else {
            throw Stage4PluginError.inferenceFailure("Recognizer models or crop polygon invalid.")
        }

        let stripCG = try rectifyAndScaleToH64(cgImage: cg, polygon: polygon, orientationFlag: orientationFlag)
        let previewPNG = UIImage(cgImage: stripCG).pngData() ?? Data()

        if stripCG.width > 512 {
            // >512px multi-window overlap-100 fallback (matching Python long_line_inference.py)
            let ranges = computeWindowRanges(width: stripCG.width, maxWidth: 512, overlap: 100)
            var windowPaths: [[Int]] = []
            for r in ranges {
                let left = r[0]
                let right = r[1]
                let rect = CGRect(x: left, y: 0, width: right - left, height: stripCG.height)
                guard let winCG = stripCG.cropping(to: rect) else { continue }
                let (logitsData, validT) = try runSplitRecognizerOnStrip(
                    stripCG: winCG,
                    orientationFlag: orientationFlag,
                    featModel: featModel,
                    headModel: headModel
                )
                let path = argmaxPerFrame(logitsData: logitsData, validT: validT, classes: 2106)
                windowPaths.append(path)
            }
            return [
                "multi_window": true,
                "window_count": ranges.count,
                "window_ranges": ranges,
                "window_paths": windowPaths,
                "valid_T": 0,
                "logits_f32": FlutterStandardTypedData(bytes: Data()),
                "crop_preview_png": FlutterStandardTypedData(bytes: previewPNG),
            ]
        }

        let (logitsData, validT) = try runSplitRecognizerOnStrip(
            stripCG: stripCG,
            orientationFlag: orientationFlag,
            featModel: featModel,
            headModel: headModel
        )
        return [
            "multi_window": false,
            "window_count": 1,
            "window_ranges": [],
            "window_paths": [],
            "valid_T": validT,
            "logits_f32": FlutterStandardTypedData(bytes: logitsData),
            "crop_preview_png": FlutterStandardTypedData(bytes: previewPNG),
        ]
    }

    private func runSplitRecognizerOnStrip(
        stripCG: CGImage,
        orientationFlag: Int,
        featModel: MLModel,
        headModel: MLModel
    ) throws -> (Data, Int) {
        let validW = max(1, min(512, stripCG.width))
        let bucketW = Self.recognizerBuckets.first(where: { $0 >= validW }) ?? 512
        let validT = max(1, (validW + 3) / 4)

        // Render strip into RGBA [64, validW] then replicate-pad right edge to [64, bucketW]
        let h = 64
        var rgba = [UInt8](repeating: 255, count: h * validW * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        rgba.withUnsafeMutableBytes { ptr in
            if let ctx = CGContext(
                data: ptr.baseAddress,
                width: validW,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: validW * 4,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) {
                ctx.interpolationQuality = .high
                ctx.draw(stripCG, in: CGRect(x: 0, y: 0, width: validW, height: h))
            }
        }

        let imgArr = try MLMultiArray(shape: [1, 3, 64, NSNumber(value: bucketW)], dataType: .float32)
        let dst = imgArr.dataPointer.bindMemory(to: Float.self, capacity: 3 * h * bucketW)
        let plane = h * bucketW
        for y in 0..<h {
            for x in 0..<bucketW {
                let srcX = min(x, validW - 1) // physical replicate padding
                let base = (y * validW + srcX) * 4
                let r = (Float(rgba[base]) / 255.0 - Self.imageNetMean[0]) / Self.imageNetStd[0]
                let g = (Float(rgba[base + 1]) / 255.0 - Self.imageNetMean[1]) / Self.imageNetStd[1]
                let b = (Float(rgba[base + 2]) / 255.0 - Self.imageNetMean[2]) / Self.imageNetStd[2]
                let idx = y * bucketW + x
                dst[idx] = r
                dst[plane + idx] = g
                dst[2 * plane + idx] = b
            }
        }

        let oriArr = try MLMultiArray(shape: [1], dataType: .float32)
        oriArr[0] = NSNumber(value: Float(orientationFlag))

        // Step 1: Feature extractor on full physical bucket canvas
        let featIn = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: imgArr),
            "orientation": MLFeatureValue(multiArray: oriArr),
        ])
        let featOut = try featModel.prediction(from: featIn)
        guard let tempFeats = featOut.featureValue(for: "temporal_features")?.multiArrayValue else {
            throw Stage4PluginError.inferenceFailure("Missing 'temporal_features' output from feature extractor.")
        }

        // Step 2: Host slice valid prefix [1, 256, 0..<validT] BEFORE backward LSTM!
        guard tempFeats.shape.count == 3, tempFeats.shape[1].intValue == 256, tempFeats.shape[2].intValue >= validT else {
            throw Stage4PluginError.inferenceFailure("Invalid temporal feature shape \(tempFeats.shape) for validT=\(validT)")
        }
        let sliced = try MLMultiArray(shape: [1, 256, NSNumber(value: validT)], dataType: .float32)
        let sPtr = sliced.dataPointer.bindMemory(to: Float.self, capacity: 256 * validT)
        for c in 0..<256 {
            let dstRow = c * validT
            for t in 0..<validT {
                sPtr[dstRow + t] = tempFeats[[NSNumber(value: 0), NSNumber(value: c), NSNumber(value: t)]].floatValue
            }
        }

        // Step 3: Recurrent head on exact valid_T prefix -> [valid_T, 1, 2106]
        let headIn = try MLDictionaryFeatureProvider(dictionary: [
            "valid_features": MLFeatureValue(multiArray: sliced)
        ])
        let headOut = try headModel.prediction(from: headIn)
        guard let logitsArr = headOut.featureValue(for: "logits")?.multiArrayValue else {
            throw Stage4PluginError.inferenceFailure("Missing 'logits' output from recurrent head.")
        }
        let logitsData = multiArrayToContiguousFloat32Data(logitsArr, count: validT * 2106)
        return (logitsData, validT)
    }

    private func rectifyAndScaleToH64(cgImage: CGImage, polygon: [[Double]], orientationFlag: Int) throws -> CGImage {
        let imgH = Double(cgImage.height)
        // Compute axis-aligned crop fallback / perspective warp using CoreImage (bottom-left origin)
        let xs = polygon.map { $0[0] }
        let ys = polygon.map { $0[1] }
        let minX = max(0.0, xs.min() ?? 0.0)
        let maxX = min(Double(cgImage.width), xs.max() ?? Double(cgImage.width))
        let minY = max(0.0, ys.min() ?? 0.0)
        let maxY = min(imgH, ys.max() ?? imgH)
        let cropW = max(2.0, maxX - minX)
        let cropH = max(2.0, maxY - minY)

        var ci = CIImage(cgImage: cgImage).cropped(
            to: CGRect(x: minX, y: imgH - maxY, width: cropW, height: cropH)
        )
        // Once-only 90° CCW rotation for vertical text lines (matching Python native_strip)
        if orientationFlag == 1 && cropH > cropW {
            ci = ci.oriented(.left)
        }
        let extent = ci.extent
        let scale = 64.0 / max(1.0, Double(extent.height))
        let targetW = max(4, Int((Double(extent.width) * scale).rounded()))
        let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let outCG = ciContext.createCGImage(
            scaled,
            from: CGRect(x: scaled.extent.minX, y: scaled.extent.minY, width: CGFloat(targetW), height: 64)
        ) else {
            throw Stage4PluginError.inferenceFailure("Failed to rasterize H64 rectified crop.")
        }
        return outCG
    }

    private func computeWindowRanges(width: Int, maxWidth: Int, overlap: Int) -> [[Int]] {
        if width <= maxWidth { return [[0, width]] }
        let step = maxWidth - overlap
        var out: [[Int]] = []
        var left = 0
        while true {
            let right = min(left + maxWidth, width)
            out.append([left, right])
            if right >= width { break }
            left += step
            if width - left < maxWidth { left = width - maxWidth }
        }
        return out
    }

    private func argmaxPerFrame(logitsData: Data, validT: Int, classes: Int) -> [Int] {
        var path: [Int] = []
        path.reserveCapacity(validT)
        logitsData.withUnsafeBytes { raw in
            let ptr = raw.bindMemory(to: Float.self)
            for t in 0..<validT {
                let base = t * classes
                var bestI = 0
                var bestV = ptr[base]
                for c in 1..<classes {
                    let v = ptr[base + c]
                    if v > bestV { bestV = v; bestI = c }
                }
                path.append(bestI)
            }
        }
        return path
    }

    private func multiArrayToContiguousFloat32Data(_ arr: MLMultiArray, count: Int) -> Data {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        precondition(count == shape.reduce(1, *), "Unexpected Core ML logical element count")
        var out = [Float](repeating: 0, count: count)
        for linear in 0..<count {
            var remaining = linear
            var coordinates = [Int](repeating: 0, count: shape.count)
            var offset = 0
            for axis in stride(from: shape.count - 1, through: 0, by: -1) {
                coordinates[axis] = remaining % shape[axis]
                remaining /= shape[axis]
                offset += coordinates[axis] * strides[axis]
            }
            if arr.dataType == .float32 {
                out[linear] = arr.dataPointer.assumingMemoryBound(to: Float.self)[offset]
            } else {
                out[linear] = arr[coordinates.map { NSNumber(value: $0) }].floatValue
            }
        }
        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
