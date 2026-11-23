// Stage 4 Native iOS AVFoundation Camera Preview & Exact Preview-Stream Shutter Capture.
//
// Captures the EXACT negotiated AVCaptureVideoDataOutput preview stream pixel buffer
// (modulo justified 90/270 orientation swap) and encodes lossless PNG.
// Zero live YOLO/OCR inference runs on preview frames.

import AVFoundation
import CoreVideo
import Flutter
import Foundation
import UIKit

final class Stage4NativeCamera: NSObject, FlutterTexture, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.menucamera.stage4.camera_session")
    private let bufferLock = NSLock()

    private weak var textureRegistry: FlutterTextureRegistry?
    private var textureId: Int64 = -1
    private var latestPixelBuffer: CVPixelBuffer?
    private var latestWidth: Int = 0
    private var latestHeight: Int = 0
    private var isConfigured = false
    private var isRunning = false

    init(textureRegistry: FlutterTextureRegistry?) {
        self.textureRegistry = textureRegistry
        super.init()
    }

    func startPreview(completion: @escaping (Result<[String: Any], Error>) -> Void) {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            do {
                try self.configureIfNeeded()
                if self.textureId < 0, let reg = self.textureRegistry {
                    self.textureId = reg.register(self)
                }
                if !self.session.isRunning {
                    self.session.startRunning()
                    self.isRunning = true
                }
                let dims = self.currentStreamDimensions()
                completion(.success([
                    "texture_id": self.textureId,
                    "stream_width": dims.width,
                    "stream_height": dims.height,
                    "running": self.isRunning,
                ]))
            } catch {
                completion(.failure(error))
            }
        }
    }

    func stopPreview(completion: @escaping () -> Void) {
        sessionQueue.async { [weak self] in
            guard let self = self else { completion(); return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            self.isRunning = false
            if self.textureId >= 0, let reg = self.textureRegistry {
                reg.unregisterTexture(self.textureId)
                self.textureId = -1
            }
            self.bufferLock.lock()
            self.latestPixelBuffer = nil
            self.bufferLock.unlock()
            completion()
        }
    }

    func captureExactStreamFrame() throws -> [String: Any] {
        bufferLock.lock()
        let pb = latestPixelBuffer
        let w = latestWidth
        let h = latestHeight
        bufferLock.unlock()

        guard let pixelBuffer = pb, w > 0, h > 0 else {
            throw NSError(
                domain: "Stage4NativeCamera",
                code: 1001,
                userInfo: [NSLocalizedDescriptionKey: "No preview stream frame available yet."]
            )
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let ctx = CIContext(options: [.useSoftwareRenderer: false])
        guard let cgImage = ctx.createCGImage(ciImage, from: CGRect(x: 0, y: 0, width: w, height: h)) else {
            throw NSError(
                domain: "Stage4NativeCamera",
                code: 1002,
                userInfo: [NSLocalizedDescriptionKey: "Failed to rasterize CVPixelBuffer to CGImage."]
            )
        }
        let uiImage = UIImage(cgImage: cgImage)
        guard let pngData = uiImage.pngData() else {
            throw NSError(
                domain: "Stage4NativeCamera",
                code: 1003,
                userInfo: [NSLocalizedDescriptionKey: "Failed to encode stream buffer to lossless PNG."]
            )
        }
        return [
            "png_bytes": FlutterStandardTypedData(bytes: pngData),
            "stream_width": w,
            "stream_height": h,
            "stream_rotation": 0,
            "capture_source": "ios_avfoundation_stream_buffer",
        ]
    }

    private func configureIfNeeded() throws {
        if isConfigured { return }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .hd1920x1080
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            throw NSError(
                domain: "Stage4NativeCamera",
                code: 1000,
                userInfo: [NSLocalizedDescriptionKey: "Back wide-angle camera unavailable on this device."]
            )
        }
        let input = try AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        if let conn = videoOutput.connection(with: .video), conn.isVideoOrientationSupported {
            conn.videoOrientation = .portrait
        }
        isConfigured = true
    }

    private func currentStreamDimensions() -> (width: Int, height: Int) {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        return (latestWidth, latestHeight)
    }

    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        bufferLock.lock()
        latestPixelBuffer = pb
        latestWidth = CVPixelBufferGetWidth(pb)
        latestHeight = CVPixelBufferGetHeight(pb)
        let tid = textureId
        bufferLock.unlock()

        if tid >= 0 {
            textureRegistry?.textureFrameAvailable(tid)
        }
    }

    // MARK: - FlutterTexture
    func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        guard let pb = latestPixelBuffer else { return nil }
        return Unmanaged.passRetained(pb)
    }
}
