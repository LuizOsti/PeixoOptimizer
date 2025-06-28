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
from Levenshtein import distance as levenshtein_distance

pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'
RIGHT_TEXT_REGION = (1352, 140, 468, 510)

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_FOLDER = os.path.join(SCRIPT_PATH, 'scans_templates')

TEMPLATE_FILE_NAMES = [
    'icone_item.png', 'icone_item1.png', 'icone_item2.png',
    'icone_item3.png', 'icone_item4.png', 'icone_item5.png',
    'icone_item6.png', 'icone_item7.png'
]
TEMPLATE_IMAGES = [os.path.join(TEMPLATES_FOLDER, name) for name in TEMPLATE_FILE_NAMES]

SET_KEYWORDS = [
    "keeneye", "swiftrush", "battlewill", "bloodbath", "bramble", 
    "bulwark", "cure", "fury", "harvest", "momentum", "onslaught", 
    "strive", "timewave", "unbreakable", "wellspring"
]

# Sets que usarão reconhecimento de formato em vez de cor
ACHROMATIC_SETS = ["unbreakable", "bulwark"]

# Base de dados de cores para os sets coloridos
SYMBOL_COLOR_RANGES = {
    "fury":        [[np.array([0, 150, 150]), np.array([10, 255, 255])], [np.array([170, 150, 150]), np.array([180, 255, 255])]],
    "momentum":    [[np.array([0, 120, 120]), np.array([10, 255, 255])], [np.array([170, 120, 120]), np.array([180, 255, 255])]],
    "keeneye":     [[np.array([0, 120, 120]), np.array([10, 255, 255])], [np.array([170, 120, 120]), np.array([180, 255, 255])]],
    "onslaught":   [[np.array([0, 120, 120]), np.array([10, 255, 255])], [np.array([170, 120, 120]), np.array([180, 255, 255])]],
    "strive":      [[np.array([0, 120, 120]), np.array([10, 255, 255])], [np.array([170, 120, 120]), np.array([180, 255, 255])]],
    "battlewill":  [[np.array([130, 80, 80]), np.array([165, 255, 255])]],
    "bloodbath":   [[np.array([20, 100, 100]), np.array([35, 255, 255])]],
    "timewave":    [[np.array([85, 80, 80]), np.array([130, 255, 255])]],
    "swiftrush":   [[np.array([85, 80, 80]), np.array([130, 255, 255])]],
    "harvest":     [[np.array([35, 80, 80]), np.array([85, 255, 255])]],
    "wellspring":  [[np.array([35, 80, 80]), np.array([85, 255, 255])]],
    "cure":        [[np.array([35, 80, 80]), np.array([85, 255, 255])]],
    "bramble":     [[np.array([130, 80, 80]), np.array([165, 255, 255])]]
}


KNOWN_STATS = [ "ATK", "HP", "DEF", "SPD", "CRIT Rate", "CRIT DMG", "Effect ACC", "Effect RES", "ATK Bonus", "DEF Bonus" ]
ITEMS_REGION = (169, 127, 1146, 700)
pyautogui.PAUSE = 0
ITEM_CONFIDENCE_LEVEL = 0.85
MINIMUM_GROUPING_DISTANCE = 30
PAUSE_AFTER_CLICK = 0.001
SCROLL_AMOUNT = -3000
OUTPUT_FILE = 'gear.json'

def get_closest_match(s, options):
    if not s or not s.strip(): return None
    s_lower = s.lower()
    for opt in options:
        if opt.lower() == s_lower: return opt
    distances = {opt: levenshtein_distance(s_lower, opt.lower()) for opt in options}
    min_dist = min(distances.values())
    if min_dist > 3: return None
    return min(distances, key=distances.get)

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

def count_symbols_by_template(region_screenshot, set_name):
    try:
        template_filename = f"simbolo_{set_name}.png"
        template_path = os.path.join(TEMPLATES_FOLDER, template_filename)
        template = cv2.imread(template_path, 0)
        if template is None: return 0

        _, template_bin = cv2.threshold(template, 50, 255, cv2.THRESH_BINARY)
        w, h = template.shape[::-1]

        region_gray = cv2.cvtColor(np.array(region_screenshot), cv2.COLOR_RGB2GRAY)
        res = cv2.matchTemplate(region_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.8)

        rectangles = [ [int(pt[0]), int(pt[1]), int(w), int(h)] for pt in zip(*loc[::-1]) ]
        grouped_rects, _ = cv2.groupRectangles(rectangles, 1, 0.5)
        return len(grouped_rects)
    except Exception as e:
        print(f"Erro na contagem por template para {set_name}: {e}")
        return 0

def count_symbols_by_color(region_screenshot, set_name):
    if set_name not in SYMBOL_COLOR_RANGES: return 0
    try:
        image_cv = cv2.cvtColor(np.array(region_screenshot), cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
        color_ranges = SYMBOL_COLOR_RANGES[set_name]
        combined_mask = cv2.inRange(image_hsv, color_ranges[0][0], color_ranges[0][1])

        if len(color_ranges) > 1:
            for i in range(1, len(color_ranges)):
                mask = cv2.inRange(image_hsv, color_ranges[i][0], color_ranges[i][1])
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        kernel = np.ones((3,3), np.uint8)
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area, max_area = 35, 500
        valid_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.7 < aspect_ratio < 1.4:
                    valid_contours += 1
        return valid_contours
    except Exception as e:
        print(f"Erro na contagem por cor para {set_name}: {e}")
        return 0

def ocr_worker(screenshot):
    # A abordagem de múltiplos pipelines é a mais robusta.
    cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Pipeline 1: Simples Ampliado
    scale_factor = 4
    p1_img = cv2.resize(gray_image.copy(), (gray_image.shape[1] * scale_factor, gray_image.shape[0] * scale_factor), interpolation=cv2.INTER_LANCZOS4)
    
    # Pipeline 2: Adaptativo
    p2_img = cv2.adaptiveThreshold(gray_image.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    p2_img = cv2.resize(p2_img, (p2_img.shape[1] * 2, p2_img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    pipelines = [p1_img, p2_img]
    results = []
    tesseract_config = '--psm 6 --oem 1'
    
    for p_img in pipelines:
        try:
            results.append(pytesseract.image_to_string(p_img, lang='eng', config=tesseract_config))
        except Exception:
            results.append("")

    best_score, best_text = -1, ""
    keywords = KNOWN_STATS + SET_KEYWORDS
    for text in results:
        score = 0
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 5
        score += len(re.findall(r'\d', text)) + text.count('+')
        if score > best_score:
            best_score, best_text = score, text
            
    return "\n".join([line for line in best_text.split('\n') if line.strip() != ''])

def parse_stats_from_text(raw_text):
    raw_text = raw_text.replace("+A", "+4")
    extracted_data = {"stats": {}, "equipped_by": "Nobody"}

    if "not equipped by" in raw_text.lower():
        extracted_data["equipped_by"] = "Nobody"
        raw_text = re.sub(r'.*not equipped by.*', '', raw_text, flags=re.IGNORECASE)

    lines = raw_text.split('\n')
    stat_pattern = re.compile(r"([a-zA-Z\s]+?)\s*\+\s*([\d,]+)%?")
    
    for line in lines:
        clean_line = line.strip()
        stat_match = stat_pattern.search(clean_line)
        equipped_match = re.search(r'(.+?)\s*Equipped', clean_line, re.IGNORECASE)

        if stat_match:
            stat_name_raw, stat_value_str = stat_match.group(1).strip(), stat_match.group(2).replace(',', '')
            corrected_stat_name = get_closest_match(stat_name_raw, KNOWN_STATS)
            if corrected_stat_name and stat_value_str.isdigit():
                if '%' in clean_line:
                    extracted_data["stats"][corrected_stat_name] = f"{stat_value_str}%"
                else:
                    extracted_data["stats"][corrected_stat_name] = int(stat_value_str)
        elif equipped_match and extracted_data["equipped_by"] == "Nobody":
            char_name = equipped_match.group(1).strip()
            if "Current" in char_name:
                char_name = char_name.replace("Current", "").strip()
            parts = char_name.split()
            extracted_data["equipped_by"] = parts[-1] if parts else "Nobody"
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
                found = list(pyautogui.locateAllOnScreen(template, confidence=ITEM_CONFIDENCE_LEVEL, region=ITEMS_REGION, grayscale=True))
                found_positions_on_screen.extend(found)
            except pyscreeze.PyScreezeException: continue

        positions_to_click = group_close_positions(found_positions_on_screen)

        if not positions_to_click:
            print("No item icons found in this cycle.")
            consecutive_failures += 1
        else:
            print(f"Found {len(positions_to_click)} unique icons. Processing...")
            truly_new_items_this_round = 0

            for position in positions_to_click:
                pyautogui.click(pyautogui.center(position))
                time.sleep(PAUSE_AFTER_CLICK)
                text_screenshot = pyautogui.screenshot(region=RIGHT_TEXT_REGION)
                original_text = ocr_worker(text_screenshot)

                if not original_text or "OCR_ERROR" in original_text: continue
                text_id = normalize_text(original_text)

                if text_id not in seen_text_identifiers:
                    truly_new_items_this_round += 1
                    global_item_counter += 1
                    seen_text_identifiers.add(text_id)
                    print(f"--> Processing item {global_item_counter}...")

                    parsed_data = parse_stats_from_text(original_text)
                    found_set_name, symbol_count = "None", 0
                    text_lower = original_text.lower()

                    for keyword in SET_KEYWORDS:
                        if keyword in text_lower:
                            found_set_name = keyword
                            width, _ = text_screenshot.size
                            symbol_search_area = text_screenshot.crop((0, 0, width, 120))
                            
                            if found_set_name in ACHROMATIC_SETS:
                                symbol_count = count_symbols_by_template(symbol_search_area, found_set_name)
                            else:
                                symbol_count = count_symbols_by_color(symbol_search_area, found_set_name)
                            break

                    final_results.append({
                        'item_number': global_item_counter,
                        'set_name': found_set_name,
                        'symbol_count': symbol_count,
                        'stats': parsed_data['stats'],
                        'equipped_by': parsed_data['equipped_by'],
                        'raw_text': original_text
                    })

            print(f"{truly_new_items_this_round} new items were processed and saved.")
            if truly_new_items_this_round == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            print("Scrolling down to find more items...")
            pyautogui.scroll(SCROLL_AMOUNT)
            time.sleep(0.5)

        if consecutive_failures >= 2:
            print("\nNo new items found after scrolling twice. Assuming end of inventory.")
            break

    print("\n" + "="*50 + "\nPROCESS COMPLETED!")
    print(f"A total of {len(final_results)} unique items were read and saved.")
    end_time = time.time()
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to '{OUTPUT_FILE}'.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()