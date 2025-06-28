import pyautogui
import pytesseract
import time
import cv2
import numpy as np
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyscreeze
import os

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
RIGHT_TEXT_REGION = (1352, 140, 468, 510)

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_FOLDER = os.path.join(SCRIPT_PATH, 'scans_templates')

TEMPLATE_FILE_NAMES = [
    'icon_item.png', 'icon_item1.png', 'icon_item2.png',
    'icon_item3.png', 'icon_item4.png', 'icon_item5.png',
    'icon_item6.png', 'icon_item7.png'
]
TEMPLATE_IMAGES = [os.path.join(TEMPLATES_FOLDER, name) for name in TEMPLATE_FILE_NAMES]

SYMBOL_MAP = {
    "keeneye": "symbol_keeneye.png",
    "swiftrush": "symbol_swiftrush.png",
    "battlewill": "symbol_battlewill.png",
    "bloodbath" : "symbol_bloodbath.png",
    "bramble" : "symbol_bramble.png",
    "bulwark" : "symbol_bulwark.png",
    "cure" : "symbol_cure.png",
    "fury" : "symbol_fury.png",
    "harvest" : "symbol_harvest.png",
    "momentum" : "symbol_momentum.png",
    "onslaught" : "symbol_onslaught.png",
    "strive" : "symbol_strive.png",
    "timewave" : "symbol_timewave.png",
    "unbreakable" : "symbol_unbreakable.png",
    "wellspring" : "symbol_wellspring.png"
}

ITEMS_REGION = (169, 127, 1146, 700)

pyautogui.PAUSE = 0
ITEM_CONFIDENCE_LEVEL = 0.85
SYMBOL_CONFIDENCE_LEVEL = 0.85
MINIMUM_GROUPING_DISTANCE = 30
PAUSE_AFTER_CLICK = 0.001
SCROLL_AMOUNT = -3000
OUTPUT_FILE = 'gear.json'

def group_close_positions(position_list):
    if not position_list: return []
    unique_positions = []
    position_list.sort(key=lambda p: (p.top, p.left))
    for pos in position_list:
        center_pos = pyautogui.center(pos)
        if not any(math.dist(pyautogui.center(unique), center_pos) < MINIMUM_GROUPING_DISTANCE for unique in unique_positions):
            unique_positions.append(pos)
    return unique_positions

def normalize_text(text):
    return "".join(text.lower().split())

def count_symbols(template_image, search_region):
    try:
        full_path = os.path.join(TEMPLATES_FOLDER, template_image)
        occurrences = list(pyautogui.locateAllOnScreen(
            full_path,
            region=search_region,
            confidence=SYMBOL_CONFIDENCE_LEVEL
        ))
        return len(occurrences)
    except (pyscreeze.PyScreezeException, FileNotFoundError):
        return 0

def ocr_worker(screenshot):
    try:
        cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.medianBlur(gray_image, 3)
        resize_factor = 2.5
        large_image = cv2.resize(denoised_image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        _, processed_image = cv2.threshold(large_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        tesseract_config = '--psm 3 --oem 1'
        extracted_text = pytesseract.image_to_string(processed_image, lang='eng', config=tesseract_config)
        lines = [line for line in extracted_text.split('\n') if line.strip() != '']
        return "\n".join(lines)
    except Exception as e:
        return f"OCR_ERROR: {e}"

def parse_stats_from_text(raw_text):
    known_stats = [
        "DEF", "CRIT Rate", "Effect RES", "ATK", "SPD",
        "HP", "CRIT DMG", "Effect ACC"
    ]
    extracted_data = {"stats": {}, "equipped_by": "Nobody"}
    for stat in known_stats:
        pattern = re.compile(rf"{re.escape(stat)}\s*\+?(\d+)\%?")
        match = pattern.search(raw_text)
        if match:
            extracted_data["stats"][stat] = int(match.group(1))
    equipped_pattern = re.compile(r"(.+?)\s+Equipped")
    equipped_match = equipped_pattern.search(raw_text)
    if equipped_match:
        extracted_data["equipped_by"] = equipped_match.group(1).strip()
    return extracted_data

def main():
    print("Gear Scanner will start in 3 seconds...")
    time.sleep(3)
    start_time = time.time()

    seen_text_identifiers = set()
    final_results = []
    global_item_counter = 0
    consecutive_failures = 0

    while True:
        print("\n--- Searching for icons...")
        found_positions_on_screen = []
        for template in TEMPLATE_IMAGES:
            try:
                found = list(pyautogui.locateAllOnScreen(
                    template, confidence=ITEM_CONFIDENCE_LEVEL, region=ITEMS_REGION, grayscale=True
                ))
                found_positions_on_screen.extend(found)
            except pyscreeze.PyScreezeException: continue

        positions_to_click = group_close_positions(found_positions_on_screen)

        if not positions_to_click:
            print("No item icons found in this cycle.")
            consecutive_failures += 1
        else:
            print(f"Found {len(positions_to_click)} icons. Processing one by one...")
            truly_new_items_this_round = 0

            for position in positions_to_click:
                pyautogui.click(pyautogui.center(position))
                time.sleep(PAUSE_AFTER_CLICK)

                text_screenshot = pyautogui.screenshot(region=RIGHT_TEXT_REGION)
                original_text = ocr_worker(text_screenshot)

                if not original_text or "OCR_ERROR" in original_text:
                    continue

                text_id = normalize_text(original_text)
                if text_id not in seen_text_identifiers:
                    truly_new_items_this_round += 1
                    global_item_counter += 1
                    seen_text_identifiers.add(text_id)

                    parsed_data = parse_stats_from_text(original_text)

                    found_symbol_type = "None"
                    symbol_count = 0

                    text_lower = original_text.lower()
                    for keyword, symbol_file in SYMBOL_MAP.items():
                        if keyword in text_lower:
                            found_symbol_type = keyword
                            symbol_count = count_symbols(symbol_file, ITEMS_REGION)
                            break

                    formatted_item = {
                        'item_number': global_item_counter,
                        'set_name': found_symbol_type,
                        'symbol_count': symbol_count,
                        'stats': parsed_data['stats'],
                        'equipped_by': parsed_data['equipped_by'],
                        'raw_text': original_text
                    }
                    final_results.append(formatted_item)

            print(f"{truly_new_items_this_round} new items found.")

            if truly_new_items_this_round == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                print("Scrolling down...")
                pyautogui.scroll(SCROLL_AMOUNT)

        if consecutive_failures >= 2:
            print("\nEnd of inventory.")
            break

    print("\n" + "="*50 + "\nPROCESS COMPLETED!")
    print(f"A total of {len(final_results)} unique items were read and saved.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to '{OUTPUT_FILE}'.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()