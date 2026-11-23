import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:menu_camera_flutter/services/ios_coreml_backend.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('com.menucamera.stage4/coreml_backend');

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test(
    'Float32 decoder respects unaligned subviews and rejects partial floats',
    () {
      final bytes = Uint8List(15);
      final view = Uint8List.sublistView(bytes, 3, 15);
      final data = ByteData.sublistView(view);
      data.setFloat32(0, 0.95, Endian.little);
      data.setFloat32(4, -2.5, Endian.little);
      data.setFloat32(8, 512.0, Endian.little);
      final decoded = decodeNativeFloat32Bytes(view);
      expect(decoded[0], closeTo(0.95, 1e-6));
      expect(decoded.sublist(1), [-2.5, 512.0]);
      expect(
        () => decodeNativeFloat32Bytes(Uint8List(3)),
        throwsFormatException,
      );
    },
  );

  test(
    'diagnostic runs only native detector and NMS, without text assets',
    () async {
      final calls = <String>[];
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
            calls.add(call.method);
            expect(call.method, 'runDetectorDiagnostic');
            expect(call.arguments['cpu_only'], true);
            final f = Float32List.fromList([50, 50, 40, 12, 0.95, 0.01, 0]);
            return {
              'status': 'passed',
              'native': {
                'stage': 'detector_complete',
                'max_class_confidence': 0.95,
              },
              'raw_obb_f32': f.buffer.asUint8List(),
              'num_anchors': 1,
              'letterbox': {
                'gain': 1.0,
                'pad_x': 0.0,
                'pad_y': 0.0,
                'src_w': 100.0,
                'src_h': 100.0,
              },
            };
          });
      final backend = IosCoreMLBackend();
      final report = await backend.runDetectorDiagnostic(
        null,
        confidence: 0.35,
        cpuOnly: true,
      );
      expect(calls, ['runDetectorDiagnostic']);
      expect(report['status'], 'passed');
      expect(report['post_nms_count'], 1);
      expect(report['test'], 'bundled_menu101');
    },
  );

  test('native detector failure retains completed-stage evidence', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          channel,
          (call) async => {
            'status': 'failed',
            'error': 'model unavailable',
            'native': {'stage': 'load_detector', 'precision': 'fp32'},
          },
        );
    final report = await IosCoreMLBackend().runDetectorDiagnostic(
      Uint8List(2),
      confidence: 0.35,
    );
    expect(report['status'], 'failed');
    expect(report['stage'], 'load_detector');
    expect(report['error'], contains('model unavailable'));
    expect((report['native'] as Map)['precision'], 'fp32');
  });
}
