/// Local persistence for Stage 4 settings (SharedPreferences) with explicit defaults & reset.
library;

import 'package:shared_preferences/shared_preferences.dart';
import '../models/contracts.dart';

class SettingsStore {
  static const _kConf = 'stage4_yolo_confidence';
  static const _kAlpha = 'stage4_charlm_alpha';
  static const _kMode = 'stage4_decode_mode';

  Future<AppSettings> load() async {
    final prefs = await SharedPreferences.getInstance();
    final conf = prefs.getDouble(_kConf) ?? AppSettings.defaults.yoloConfidence;
    final alpha = prefs.getDouble(_kAlpha) ?? AppSettings.defaults.charLmAlpha;
    final mode = prefs.getString(_kMode) ?? AppSettings.defaults.decodeMode;
    return AppSettings(
      yoloConfidence: conf.clamp(0.10, 0.90),
      charLmAlpha: alpha.clamp(0.00, 0.60),
      decodeMode: (mode == 'greedy' || mode == 'charlm_r1') ? mode : 'charlm_r1',
    );
  }

  Future<void> save(AppSettings settings) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_kConf, settings.yoloConfidence);
    await prefs.setDouble(_kAlpha, settings.charLmAlpha);
    await prefs.setString(_kMode, settings.decodeMode);
  }

  Future<AppSettings> reset() async {
    await save(AppSettings.defaults);
    return AppSettings.defaults;
  }
}
