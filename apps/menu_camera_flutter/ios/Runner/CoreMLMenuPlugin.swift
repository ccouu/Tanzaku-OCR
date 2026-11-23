// Stage 4 Native iOS Core ML + AVFoundation + Read-Only SQLite Flutter Plugin.
//
// Exposes typed MethodChannel ("com.menucamera.stage4/coreml_backend") for:
//   - initialize / getCapabilities (manifest & compiled .mlmodelc resource inspection)
//   - startCameraPreview / capturePreviewStreamFrame / stopCameraPreview (AVFoundation exact stream buffer)
//   - runDetectorOBB1024 (V5 YOLO11n-OBB Core ML -> raw [1,7,21504] Float32)
//   - runRecognizerCrop (split RecognizerFeatureExtractor + host valid_T slice + RecognizerRecurrentHead)
//   - lookupSqlTerms / lookupPhrasePlan (read-only SQLite culinary_menu.db)
//   - loadBundledAssetBytes (loads bundled vocab / gzipped CharLM V2 / Lexicon V3)
//
// Zero Python, zero localhost server, zero cloud APIs on iOS.
// Compilation & on-device execution remain unverified until macOS/Xcode/iPhone 15 CI & device testing.

import Flutter
import Foundation
import UIKit

enum Stage4PluginError: Error {
    case assetsNotBundled(missing: [String])
    case invalidArguments(String)
    case inferenceFailure(String)

    var flutterError: FlutterError {
        switch self {
        case .assetsNotBundled(let missing):
            return FlutterError(
                code: "COREML_ASSETS_NOT_BUNDLED",
                message: "Missing compiled Stage 4 iOS bundle resources: \(missing.joined(separator: ", ")).",
                details: ["missing_assets": missing]
            )
        case .invalidArguments(let msg):
            return FlutterError(code: "INVALID_ARGUMENTS", message: msg, details: nil)
        case .inferenceFailure(let msg):
            return FlutterError(code: "INFERENCE_FAILURE", message: msg, details: nil)
        }
    }
}

public final class CoreMLMenuPlugin: NSObject, FlutterPlugin {
    private let inferenceQueue = DispatchQueue(label: "com.menucamera.stage4.inference")
    private let inference = Stage4CoreMLInference()
    private let sqliteStore = Stage4SQLiteStore()
    private let camera: Stage4NativeCamera

    init(textureRegistry: FlutterTextureRegistry?) {
        self.camera = Stage4NativeCamera(textureRegistry: textureRegistry)
        super.init()
    }

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "com.menucamera.stage4/coreml_backend",
            binaryMessenger: registrar.messenger()
        )
        let instance = CoreMLMenuPlugin(textureRegistry: registrar.textures())
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize", "getCapabilities":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                let missing = self.inference.missingRequiredResources()
                var sqliteReady = false
                if missing.isEmpty, let dbURL = self.inference.bundledResourceURL("culinary_menu.db") {
                    do {
                        try self.sqliteStore.open(path: dbURL.path)
                        sqliteReady = true
                    } catch {
                        sqliteReady = false
                    }
                }
                DispatchQueue.main.async {
                    result([
                        "status": missing.isEmpty ? "ready" : "assets_not_bundled",
                        "backend_id": "ios_coreml_native",
                        "device_name": UIDevice.current.model,
                        "os_version": UIDevice.current.systemVersion,
                        "bundle_id": Stage4CoreMLInference.bundleId,
                        "missing_assets": missing,
                        "capabilities": [
                            "avfoundation_stream_camera": true,
                            "coreml_split_recognizer": true,
                            "coreml_obb_1024_detector": true,
                            "readonly_sqlite_store": sqliteReady,
                            "compiled_models_present": missing.isEmpty,
                        ],
                    ])
                }
            }

        case "startCameraPreview":
            camera.startPreview { res in
                DispatchQueue.main.async {
                    switch res {
                    case .success(let payload): result(payload)
                    case .failure(let err):
                        result(FlutterError(code: "CAMERA_START_FAILED", message: err.localizedDescription, details: nil))
                    }
                }
            }

        case "stopCameraPreview":
            camera.stopPreview {
                DispatchQueue.main.async { result(["stopped": true]) }
            }

        case "capturePreviewStreamFrame":
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                do {
                    let frame = try self.camera.captureExactStreamFrame()
                    DispatchQueue.main.async { result(frame) }
                } catch {
                    DispatchQueue.main.async {
                        result(FlutterError(code: "CAMERA_CAPTURE_FAILED", message: error.localizedDescription, details: nil))
                    }
                }
            }

        case "runDetectorOBB1024":
            guard let args = call.arguments as? [String: Any],
                  let typed = args["image_bytes"] as? FlutterStandardTypedData,
                  let uiImg = UIImage(data: typed.data) else {
                result(Stage4PluginError.invalidArguments("Expected 'image_bytes' for runDetectorOBB1024.").flutterError)
                return
            }
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                do {
                    let (rawBytes, numAnchors, meta) = try self.inference.runDetectorOBB1024(image: uiImg)
                    DispatchQueue.main.async {
                        result([
                            "raw_obb_f32": FlutterStandardTypedData(bytes: rawBytes),
                            "num_anchors": numAnchors,
                            "letterbox": meta,
                        ])
                    }
                } catch let err as Stage4PluginError {
                    DispatchQueue.main.async { result(err.flutterError) }
                } catch {
                    DispatchQueue.main.async { result(Stage4PluginError.inferenceFailure(error.localizedDescription).flutterError) }
                }
            }

        case "runRecognizerCrop":
            guard let args = call.arguments as? [String: Any],
                  let typed = args["image_bytes"] as? FlutterStandardTypedData,
                  let uiImg = UIImage(data: typed.data),
                  let poly = args["polygon"] as? [[Double]],
                  let ori = args["orientation_flag"] as? Int else {
                result(Stage4PluginError.invalidArguments("Expected image_bytes, polygon, orientation_flag.").flutterError)
                return
            }
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                do {
                    let out = try self.inference.recognizeCrop(
                        sourceImage: uiImg,
                        polygon: poly,
                        orientationFlag: ori
                    )
                    DispatchQueue.main.async { result(out) }
                } catch let err as Stage4PluginError {
                    DispatchQueue.main.async { result(err.flutterError) }
                } catch {
                    DispatchQueue.main.async { result(Stage4PluginError.inferenceFailure(error.localizedDescription).flutterError) }
                }
            }

        case "lookupSqlTerms":
            guard let args = call.arguments as? [String: Any],
                  let terms = args["terms"] as? [String] else {
                result(Stage4PluginError.invalidArguments("Expected 'terms' list.").flutterError)
                return
            }
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                if let dbURL = self.inference.bundledResourceURL("culinary_menu.db") {
                    try? self.sqliteStore.open(path: dbURL.path)
                }
                let rows = self.sqliteStore.lookupTerms(terms)
                DispatchQueue.main.async { result(rows) }
            }

        case "lookupPhrasePlan":
            guard let args = call.arguments as? [String: Any],
                  let phrase = args["phrase"] as? String else {
                result(Stage4PluginError.invalidArguments("Expected 'phrase' string.").flutterError)
                return
            }
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                if let dbURL = self.inference.bundledResourceURL("culinary_menu.db") {
                    try? self.sqliteStore.open(path: dbURL.path)
                }
                let plan = self.sqliteStore.lookupPhrasePlan(phrase)
                DispatchQueue.main.async { result(plan) }
            }

        case "loadBundledAssetBytes":
            guard let args = call.arguments as? [String: Any],
                  let name = args["filename"] as? String else {
                result(Stage4PluginError.invalidArguments("Expected 'filename' string.").flutterError)
                return
            }
            inferenceQueue.async { [weak self] in
                guard let self = self else { return }
                guard let url = self.inference.bundledResourceURL(name),
                      let data = try? Data(contentsOf: url) else {
                    DispatchQueue.main.async {
                        result(Stage4PluginError.assetsNotBundled(missing: [name]).flutterError)
                    }
                    return
                }
                DispatchQueue.main.async {
                    result(FlutterStandardTypedData(bytes: data))
                }
            }

        default:
            result(FlutterMethodNotImplemented)
        }
    }
}
