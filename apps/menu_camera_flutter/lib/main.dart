/// Entrypoint for Stage 4 Capture-First Menu Camera Flutter App.
/// Uses DesktopReferenceBackend (localhost Python) on Windows/Web and IosCoreMLBackend scaffold on iOS.
library;

import 'package:material_ui/material_ui.dart';

import 'screens/capture_home_screen.dart';
import 'services/menu_backend.dart';
import 'services/settings_store.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MenuCameraApp());
}

class MenuCameraApp extends StatelessWidget {
  const MenuCameraApp({super.key, this.backend, this.settingsStore});

  final MenuBackend? backend;
  final SettingsStore? settingsStore;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Menu Camera MVP (Stage 4)',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF0F4C81),
        useMaterial3: true,
      ),
      home: CaptureHomeScreen(
        backend: backend ?? createPlatformMenuBackend(),
        settingsStore: settingsStore ?? SettingsStore(),
      ),
    );
  }
}
