"""Generate 50 diverse prompt specs and JSON sidecars for mobile menu training images."""
import json
from pathlib import Path

OUT_DIR = Path(r"D:\profolio\japanese_translation\data\raw\gemini_banana_mobile_menus_train_50")

CATEGORIES = [
    ("izakaya_specials", "Izakaya Handwritten Daily Specials", [
        "本日のおすすめ", "鶏軟骨唐揚げ 520円", "生雲丹刺し 1,480円", "熊本産馬刺 1,280円",
        "炙り〆鯖 780円", "だし巻玉子 580円", "エイヒレ炙り 480円", "生ビール 中 580円"
    ]),
    ("sushi_nigiri", "Edomae Sushi Counter Menu", [
        "特選にぎり", "生海胆 一貫 650円", "大とろ 720円", "中とろ 550円",
        "こはだ 350円", "煮穴子 480円", "赤貝 580円", "玉子 220円"
    ]),
    ("yakitori_skewer", "Charcoal Yakitori Skewer Board", [
        "炭火焼鳥", "軟骨 一本 220円", "ねぎま 200円", "つくね 260円",
        "白レバー 280円", "砂肝 190円", "手羽先 250円", "うずら玉子 180円"
    ]),
    ("ramen_tsukemen", "Tonkotsu & Niboshi Ramen Shop", [
        "麺類お品書き", "濃厚煮干そば 950円", "特製つけ麺 1,150円", "醤油らぁ麺 880円",
        "味玉トッピング 150円", "チャーシュー増し 320円", "替玉 150円", "焼き餃子 420円"
    ]),
    ("kyushu_izakaya", "Kyushu & Kumamoto Regional Menu", [
        "九州うまかもん", "特上馬刺 1,580円", "馬刺三種盛 2,100円", "博多もつ鍋 1,480円",
        "明太玉子焼 680円", "辛子蓮根 550円", "ごまカンパチ 880円", "芋焼酎 ロック 550円"
    ]),
    ("seafood_kappo", "Coastal Seafood Kappo Menu", [
        "本日の鮮魚", "北海道産雲丹 1,680円", "活帆立バター 780円", "金目鯛煮付 1,380円",
        "本鮪赤身 980円", "牡蠣ポン酢 850円", "鯵なめろう 680円", "純米吟醸 一合 820円"
    ]),
    ("soba_udon", "Handmade Stone-Milled Soba & Udon", [
        "手打ち蕎麦・うどん", "鴨せいろ 1,350円", "海老天ざる 1,480円", "山菜とろろ 980円",
        "きつねうどん 780円", "カレー南蛮 1,050円", "板わさ 480円", "そばがき 650円"
    ]),
    ("yakiniku_horumon", "Wagyu Yakiniku & Horumon Grill", [
        "焼肉・ホルモン", "上タン塩 1,580円", "黒毛和牛カルビ 1,380円", "ミノサンド 880円",
        "豚軟骨塩焼 620円", "ハラミ 1,180円", "キムチ盛合せ 680円", "冷麺 920円"
    ]),
    ("okonomiyaki_teppan", "Osaka Teppanyaki & Okonomiyaki", [
        "鉄板焼お品書", "豚玉お好み焼 920円", "海鮮ミックス玉 1,280円", "ねぎ焼すじコン 1,080円",
        "ソース焼そば 850円", "とん平焼 680円", "いか姿焼 780円", "ラムネ 280円"
    ]),
    ("cafe_kissaten", "Showa Kissaten & Japanese Cafe", [
        "喫茶メニュー", "ブレンド珈琲 520円", "宇治抹茶ラテ 620円", "厚切り小倉トースト 580円",
        "昔ながらのプリン 550円", "ナポリタン 880円", "玉子サンド 720円", "クリームソーダ 650円"
    ]),
    ("oden_winter", "Simmered Oden Counter Sheet", [
        "おでん各種", "大根 200円", "玉子 180円", "牛すじ 320円",
        "はんぺん 200円", "ちくわぶ 180円", "しらたき 180円", "熱燗 一合 550円"
    ]),
    ("tonkatsu_teishoku", "Teishoku Set Meal Restaurant", [
        "定食おしながき", "特選ロースかつ定食 1,480円", "ひれかつ定食 1,580円", "さば味噌煮定食 1,050円",
        "鶏唐揚定食 980円", "生姜焼定食 1,020円", "ごはん大盛 無料", "豚汁変更 180円"
    ]),
    ("unagi_washoku", "Charcoal-Grilled Unagi Specialist", [
        "鰻のお品書", "うな重 特上 3,800円", "うな重 上 2,900円", "ひつまぶし 3,400円",
        "白焼 2,700円", "肝焼 二本 780円", "うざく 880円", "肝吸 320円"
    ]),
    ("tempura_tendon", "Edomae Tempura Counter Menu", [
        "揚げたて天ぷら", "大海老天丼 1,450円", "穴子一本天 680円", "海胆大葉包み 880円",
        "きす天ぷら 420円", "舞茸天 350円", "かき揚げ 580円", "赤出汁 250円"
    ]),
    ("kushikatsu_bar", "Shinsekai Kushikatsu Standing Bar", [
        "串カツ定番", "牛串 180円", "紅生姜 150円", "れんこん 150円",
        "うずら 160円", "チーズ 180円", "アスパラ一本 240円", "どて焼き 420円"
    ]),
    ("japanese_sake_list", "Craft Sake & Shochu List", [
        "厳選地酒・焼酎", "純米大吟醸 980円", "特別純米 750円", "生原酒 820円",
        "麦焼酎 水割り 520円", "黒糖焼酎 ロック 580円", "梅酒ソーダ 540円", "和らぎ水 無料"
    ]),
    ("wagashi_tea", "Kyoto Matcha & Wagashi Parlor", [
        "甘味処", "宇治金時かき氷 980円", "白玉クリームあんみつ 850円", "抹茶わらび餅 720円",
        "みたらし団子 480円", "ほうじ茶パフェ 920円", "お抹茶セット 780円", "ぜんざい 680円"
    ]),
    ("chuka_gyoza", "Machi-Chuka Gyoza & Fried Rice", [
        "町中華メニュー", "特製焼餃子 450円", "水餃子 480円", "五目炒飯 820円",
        "麻婆豆腐 880円", "回鍋肉 920円", "酢豚 950円", "瓶ビール 620円"
    ]),
    ("curry_yoshoku", "Japanese Yoshoku & Curry House", [
        "洋食・カレー", "欧風ビーフカレー 1,050円", "カツカレー 1,280円", "オムライス 980円",
        "デミグラスハンバーグ 1,350円", "エビフライ 1,250円", "カニクリームコロッケ 950円", "コーンスープ 420円"
    ]),
    ("robata_robatayaki", "Hokkaido Robatayaki Hearth Menu", [
        "炉端焼き", "特大ほっけ開き 1,380円", "雲丹乗せ焼きおにぎり 780円", "じゃがバター塩辛 580円",
        "いか一夜干し 820円", "ししゃも三本 620円", "北あかり揚げ 520円", "サッポロ生 600円"
    ]),
    ("motsuyaki_tachinomi", "Shitamachi Tachinomi Motsuyaki", [
        "もつ焼・煮込み", "もつ煮込み 450円", "シロ 160円", "カシラ 170円",
        "ハツ 160円", "軟骨つくね 220円", "ガツ刺し 420円", "ホッピーセット 480円"
    ]),
    ("kaisendon_market", "Morning Fish Market Kaisendon", [
        "朝市海鮮丼", "贅沢雲丹いくら丼 2,680円", "本鮪三色丼 1,980円", "ねぎとろ月見丼 1,380円",
        "海胆小鉢 980円", "あら汁 350円", "サーモン親子丼 1,680円", "大盛 +150円"
    ]),
    ("udon_sanuki", "Sanuki Self-Serve Udon Counter", [
        "讃岐うどん", "ぶっかけ 並 480円", "釜玉うどん 550円", "肉うどん 720円",
        "ちくわ天 150円", "かしわ天 180円", "半熟玉子天 150円", "いなり 120円"
    ]),
    ("bento_deli", "Station Bento & Side Dish Menu", [
        "お持ち帰り弁当", "幕の内弁当 980円", "牛すき焼き弁当 1,280円", "銀鮭西京焼弁当 1,080円",
        "鶏照り焼き丼 820円", "だし巻折詰 620円", "ポテトサラダ 320円", "お茶 150円"
    ]),
    ("izakaya_horse_uni", "Seasonal Delicacy Tasting Sheet", [
        "今宵の逸品", "馬刺赤身 1,180円", "霜降り馬刺 1,580円", "生海胆お造り 1,650円",
        "生雲丹軍艦 二貫 980円", "鶏軟骨塩焼 480円", "ヤゲン軟骨梅和え 520円", "お通し 350円"
    ]),
]

TEXTURES = [
    "handmade fibrous ivory Echizen washi paper with subtle deckled edges",
    "weathered dark cedar wood board with natural wood grain",
    "matte kraft brown recycled paper clipped onto a dark walnut clipboard",
    "matte black chalkboard menu with faint chalk dust smudges",
    "glossy laminated A4 menu sheet with soft overhead bulb reflection",
    "open linen-bound menu book page on thick cream cotton paper",
    "aged yellowed Mino paper with tiny natural fibers and slight corner crease",
    "clear acrylic tabletop tent holder with printed off-white card insert",
    "dark navy textured cloth-bound menu cover opened to a cream inner sheet",
    "hanging hinoki cypress wooden slats lined side-by-side on a counter rail",
    "thick cork-backed bistro menu board with pinned cream paper sheet",
    "sleek brushed aluminum tabletop menu frame with matte white insert",
    "folded tri-fold glossy bistro flyer lying flat on a cafe table",
    "plastic-sleeved clear book menu page with mild indoor window sheen",
    "charred yakisugi wooden menu plaque with recessed cream paper",
    "pale bamboo-fiber menu sheet with woven edge border",
    "heavy stone-gray matte cardstock menu with embossed border",
    "warm beige rice-paper placemat menu lying beside chopsticks",
    "vintage brass-cornered dark leatherette menu folder opened flat",
    "smooth white coated menu card in a wooden groove stand",
]

TYPOGRAPHIES = [
    "bold expressive Japanese sumi brush calligraphy (vertical top-to-bottom right-to-left columns)",
    "crisp upright Japanese Mincho serif print with clean horizontal rows",
    "heavy rounded Maru-Gothic Japanese menu lettering with bold prices",
    "handwritten fine black felt-tip pen Japanese characters on horizontal lines",
    "hand-lettered white chalk script on dark slate surface",
    "energetic Showa-era sign-painter brush lettering in black and vermilion ink",
    "neat vertical brush-pen handwriting with slightly varied stroke weights",
    "compact clean Gothic sans-serif print with vertical dish names and horizontal prices",
    "elegant thin Mincho vertical columns with traditional Japanese numerals",
    "casual thick black permanent marker handwriting on light paper",
    "woodblock-style carved Japanese print typography in dark charcoal ink",
    "crisp white and cream paint-marker lettering on dark wood background",
    "warm dark-brown ink semi-cursive handwriting on textured cream paper",
    "modern clean sans-serif menu typography with boxed section headers",
]

INTERIORS = [
    "warm wooden izakaya counter with blurred ceramic sake cup at the outer edge",
    "natural daylight cafe table with blurred ceramic coffee saucer in the corner",
    "dimly lit yakitori bar table with soft amber lantern bokeh around the border",
    "clean pale cypress sushi counter with blurred green tea cup at the outer border",
    "busy ramen shop red-laminate counter with blurred chili oil jar at the edge",
    "modern Japanese dining table with blurred wooden chopstick rest at the corner",
    "traditional tatami room low table with warm paper-lantern ambient glow",
    "bright neighborhood teishoku diner table with blurred soy sauce bottle at border",
    "cozy sake bar slate table with soft overhead pendant lamp lighting",
    "bright morning fish-market wooden counter with natural window light",
]

ANGLES = [
    "nearly perpendicular top-down mobile shot with 3-degree roll and 5-degree pitch (<= 10 deg total)",
    "straight-on seated mobile capture with 6-degree downward pitch and 2-degree roll (<= 10 deg total)",
    "handheld smartphone shot at 4-degree left yaw and 7-degree downward pitch (<= 10 deg total)",
    "flat-lay mobile photo with 8-degree roll and 3-degree pitch (<= 10 deg total)",
    "direct smartphone shot with 5-degree right yaw and 5-degree downward pitch (<= 10 deg total)",
    "near-perpendicular mobile capture with 2-degree tilt and 9-degree roll (<= 10 deg total)",
    "seated diner angle with 7-degree pitch and 4-degree roll (<= 10 deg total)",
    "overhead phone camera with 1-degree roll and 8-degree pitch (<= 10 deg total)",
    "slightly angled mobile capture at 6-degree roll and 6-degree pitch (<= 10 deg total)",
    "clean tabletop mobile shot with 9-degree pitch and 2-degree roll (<= 10 deg total)",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for idx in range(1, 51):
        cat_idx = (idx - 1) % len(CATEGORIES)
        cat_slug, cat_title, lines = CATEGORIES[cat_idx]
        # Second pass over categories modifies prices/order slightly for uniqueness
        if idx > len(CATEGORIES):
            variant_tag = " (Evening Edition)"
            mod_lines = [lines[0]] + [
                l.replace("580円", "600円").replace("480円", "500円").replace("880円", "900円")
                for l in lines[1:]
            ]
        else:
            variant_tag = ""
            mod_lines = list(lines)

        texture = TEXTURES[(idx * 3 + 1) % len(TEXTURES)]
        typography = TYPOGRAPHIES[(idx * 5 + 2) % len(TYPOGRAPHIES)]
        interior = INTERIORS[(idx * 7 + 3) % len(INTERIORS)]
        angle = ANGLES[(idx - 1) % len(ANGLES)]

        # Chalkboard / white-ink / paper consistency guard
        if "chalkboard" in texture and "chalk" not in typography and "paint-marker" not in typography:
            typography = "hand-lettered white chalk script on dark slate surface"
        elif "chalk" in typography and "chalkboard" not in texture and "yakisugi" not in texture:
            texture = "matte black chalkboard menu with faint chalk dust smudges"
        elif "paint-marker" in typography and "wood" not in texture and "yakisugi" not in texture:
            texture = "weathered dark cedar wood board with natural wood grain"
        elif "cream paper" in typography and "wood" in texture:
            typography = "bold dark sumi brush lettering carved/painted cleanly onto the wooden slats"

        intended_text = " / ".join(mod_lines)
        quoted_lines = ", ".join(f'"{l}"' for l in mod_lines)
        prompt = (
            "Use case: photorealistic-natural. Asset: realistic mobile phone photograph of a Japanese restaurant menu for OCR training. "
            "Simulate an unedited smartphone photo (26mm focal length, f/1.8, natural indoor mobile camera processing) taken by a diner "
            "so the menu fills 80% to 90% of the frame, with 10% to 20% surrounding restaurant context "
            f"({interior}) softly visible around the outer border. "
            f"Camera angle: {angle}. Camera tilt and roll must NOT exceed 10 degrees. "
            f"Menu material and texture: {texture}. "
            f"Typography: {typography}. "
            "Render ONLY genuine, sharp, orthographically accurate Japanese characters and numerals "
            "(no fake pseudo-kanji strokes, no English translations, no romanization, no watermarks, no collage). "
            f"Japanese menu text to render faithfully on the menu: {quoted_lines}. "
            "Keep all text inside the menu boundaries with no clipped characters. High resolution."
        )

        item = {
            "id": idx,
            "category": cat_slug,
            "title": f"{cat_title}{variant_tag}",
            "status": "pending_image",
            "provider": "gemini_banana",
            "image": f"menu-{idx:03d}.jpg",
            "angle": angle,
            "max_camera_tilt_deg": 10,
            "interior": interior,
            "texture": texture,
            "typography": typography,
            "intended_text": intended_text,
            "gt_lines": mod_lines,
            "prompt": prompt,
            "reviewed": False,
            "source": "genai_synthetic_mobile_menu_train_50",
            "transcription_verified": False,
        }
        items.append(item)
        (OUT_DIR / f"menu-{idx:03d}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (OUT_DIR / "prompts.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "collection_name": "gemini_banana_mobile_menus_train_50",
        "total_prompts": 50,
        "completed": 0,
        "max_camera_tilt_deg": 10,
        "prompt_sidecars": "menu-001.json through menu-050.json",
        "jsonl_pack": "prompts.jsonl",
        "items": items,
    }
    (OUT_DIR / "collection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote 50 prompt specs to {OUT_DIR}")


if __name__ == "__main__":
    main()
