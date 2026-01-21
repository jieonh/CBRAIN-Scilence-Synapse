import cv2
import mediapipe as mp
import time
import numpy as np
import queue
import threading
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from llm_pipeline import start_llm_worker
import math

# --- 1. 모델 및 설정 ---
model_path = '/Users/ryu/Documents/Dev/2026CNWS/hand-tracking-using-mediapipe/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# --- 2. 데이터 매핑 ---
CATEGORIES = ["감정 및 상태", "행동 및 동사", "동물", "스포츠", "날씨 및 자연"]

CONSONANT_MAP = {
    1: "ㄱ", 2: "ㄴ", 3: "ㄷ", 4: "ㄹ", 5: "ㅁ",
    6: "ㅂ", 7: "ㅅ", 8: "ㅇ", 9: "ㅈ", 10: "ㅊ",
    11: "ㅋ", 12: "ㅌ", 13: "ㅍ", 14: "ㅎ",
    15: "ㄲ", 16: "ㄸ", 17: "ㅃ", 18: "ㅆ", 19: "ㅉ"
}

VOWEL_MAP = {
    1: "ㅏ", 2: "ㅑ", 3: "ㅓ", 4: "ㅕ", 5: "ㅗ",
    6: "ㅛ", 7: "ㅜ", 8: "ㅠ", 9: "ㅡ", 10: "ㅣ",
    11: "ㅐ", 12: "ㅒ", 13: "ㅔ", 14: "ㅖ",
    15: "ㅘ", 16: "ㅙ", 17: "ㅚ", 18: "ㅝ", 19: "ㅞ", 20: "ㅟ", 21: "ㅢ"
}

def load_korean_font(font_size=40):
    try:
        return ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttc", font_size)
    except:
        return ImageFont.load_default()

KOREAN_FONT = load_korean_font(40)
BIG_FONT = load_korean_font(60)
LLM_FONT = load_korean_font(35)

def draw_korean_text(img_bgr, text, position, color=(255, 255, 255), font=KOREAN_FONT):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_multiline_text(img_bgr, lines, start_pos, line_gap, color=(255, 255, 255), font=KOREAN_FONT):
    x, y = start_pos
    for line in lines:
        img_bgr = draw_korean_text(img_bgr, line, (x, y), color=color, font=font)
        y += line_gap
    return img_bgr

def wrap_lines(text, max_len):
    if not text: return ["-"]
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        while len(line) > max_len:
            lines.append(line[:max_len])
            line = line[max_len:]
        lines.append(line)
    return lines

# --- 3. 전역 상태 변수 ---
detection_result = None
llm_queue = queue.Queue()
llm_state = {"last_response": "대기 중...", "last_error": "", "last_input": "-"}

PHASE_CATEGORY = 0
PHASE_SPLIT_INPUT = 1 

current_phase = PHASE_CATEGORY
category_idx = None

collected_consonants = []
collected_vowels = []

# --- 독립된 상태 관리 변수 ---
cons_stable_start = None
cons_last_val = -1
cons_locked = False

vowel_stable_start = None
vowel_last_val = -1
vowel_locked = False

transition_start_time = None

# [추가됨] 삭제 기능용 타이머 변수
undo_start_time = None 
undo_locked = False # 한 번 삭제 후 잠금

def print_result(result, output_image, timestamp_ms):
    global detection_result
    detection_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=4, 
    result_callback=print_result
)

cap = cv2.VideoCapture(0)
worker_thread = start_llm_worker(llm_queue, llm_state)

def count_fingers(landmarks):
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    pinky_mcp = landmarks[17]
    
    dist_thumb_pinky = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y)
    ref_len = math.hypot(wrist.x - pinky_mcp.x, wrist.y - pinky_mcp.y)
    thumb_is_open = dist_thumb_pinky > (ref_len * 0.9)

    fingers_open_count = 0
    for t_idx, p_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        d_tip = math.hypot(landmarks[t_idx].x - wrist.x, landmarks[t_idx].y - wrist.y)
        d_pip = math.hypot(landmarks[p_idx].x - wrist.x, landmarks[p_idx].y - wrist.y)
        if d_tip > d_pip * 1.3:
            fingers_open_count += 1
            
    return (1 if thumb_is_open else 0) + fingers_open_count

cat_stable_start = 0
cat_last_val = -1

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success: break

        img = cv2.flip(img, 1) 
        h, w, _ = img.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        hands_list = []
        if detection_result and detection_result.hand_landmarks:
            for lm in detection_result.hand_landmarks:
                val = count_fingers(lm)
                hands_list.append({"lm": lm, "x": lm[0].x, "val": val})
            hands_list.sort(key=lambda item: item['x'])

        # 뼈대 그리기
        for hand in hands_list:
            lm = hand["lm"]
            for start, end in HAND_CONNECTIONS:
                p1 = (int(lm[start].x * w), int(lm[start].y * h))
                p2 = (int(lm[end].x * w), int(lm[end].y * h))
                cv2.line(img, p1, p2, (200, 200, 200), 2)

        hand_count = len(hands_list)
        
        # -----------------------------------------------------------------
        # Phase 0: 카테고리 선택
        # -----------------------------------------------------------------
        if current_phase == PHASE_CATEGORY:
            if hand_count >= 1:
                val = hands_list[0]["val"] 
                
                if val == 0:
                    cat_stable_start = None
                else:
                    if val == cat_last_val:
                        if cat_stable_start is None: cat_stable_start = time.time()
                        elapsed = time.time() - cat_stable_start
                        
                        prog = min(1.0, elapsed / 2.0)
                        cv2.circle(img, (w//2, h//2), 50, (0, 255, 255), 2)
                        cv2.circle(img, (w//2, h//2), int(50*prog), (0, 255, 255), -1)
                        
                        if elapsed >= 2.0:
                            if 1 <= val <= 5:
                                category_idx = val - 1
                                current_phase = PHASE_SPLIT_INPUT
                                collected_consonants = []
                                collected_vowels = []
                                cons_stable_start = None
                                vowel_stable_start = None
                                cons_locked = False
                                vowel_locked = False
                                # [추가됨] 삭제 상태 초기화
                                undo_start_time = None
                                undo_locked = False
                    else:
                        cat_stable_start = time.time()
                        cat_last_val = val
            
            img = draw_korean_text(img, "STEP 1: 주제 선택 (대표자 1인)", (20, 30), (255, 255, 0), BIG_FONT)
            for i, cat in enumerate(CATEGORIES):
                color = (0, 255, 0) if category_idx == i else (150, 150, 150)
                img = draw_korean_text(img, f"{i+1}. {cat}", (50, 100 + i * 50), color)

        # -----------------------------------------------------------------
        # Phase 1: 분할 입력 모드
        # -----------------------------------------------------------------
        elif current_phase == PHASE_SPLIT_INPUT:
            cv2.line(img, (w//2, 0), (w//2, h-150), (255, 255, 255), 2)
            img = draw_korean_text(img, "자음 (왼쪽 유저)", (20, 20), (100, 255, 100), BIG_FONT)
            img = draw_korean_text(img, "모음 (오른쪽 유저)", (w//2 + 20, 20), (255, 100, 255), BIG_FONT)
            
            if hand_count == 4:
                # [좌측 유저]
                cons_hand_10 = hands_list[0]
                cons_hand_1 = hands_list[1]
                cons_val = (cons_hand_10["val"] * 5) + cons_hand_1["val"]
                
                # [우측 유저]
                vowel_hand_10 = hands_list[2]
                vowel_hand_1 = hands_list[3]
                vowel_val = (vowel_hand_10["val"] * 5) + vowel_hand_1["val"]

                c_pos = (int(cons_hand_10["x"]*w), int(cons_hand_10["lm"][0].y*h)-20)
                v_pos = (int(vowel_hand_10["x"]*w), int(vowel_hand_10["lm"][0].y*h)-20)
                cv2.putText(img, f"V:{cons_val}", c_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
                cv2.putText(img, f"V:{vowel_val}", v_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,255), 2)

                # =========================================================
                # 1. 전송 로직 (양쪽 모두 30)
                # =========================================================
                if cons_val == 30 and vowel_val == 30:
                    undo_start_time = None # 삭제 타이머 리셋
                    cons_stable_start = None
                    vowel_stable_start = None
                    cons_locked = False
                    vowel_locked = False
                    
                    if transition_start_time is None: transition_start_time = time.time()
                    elapsed = time.time() - transition_start_time
                    prog = min(1.0, elapsed / 2.0)
                    
                    bar_w = 400
                    cv2.rectangle(img, (w//2 - bar_w//2, h//2 - 40), (w//2 + bar_w//2, h//2 + 40), (50, 50, 50), -1)
                    cv2.rectangle(img, (w//2 - bar_w//2, h//2 - 40), (w//2 - bar_w//2 + int(bar_w*prog), h//2 + 40), (0, 100, 255), -1)
                    img = draw_korean_text(img, "전송 중...", (w//2 - 60, h//2 - 15), (255, 255, 255), BIG_FONT)
                    
                    if elapsed >= 2.0:
                        c_str = "".join(collected_consonants)
                        v_str = "".join(collected_vowels)
                        cat_txt = CATEGORIES[category_idx]
                        
                        prompt = f"자음목록: {c_str}, 모음목록: {v_str}"
                        llm_state["last_input"] = f"[전송완료] {prompt}"
                        llm_queue.put({"category": cat_txt, "text": prompt})
                        
                        collected_consonants = []
                        collected_vowels = []
                        current_phase = PHASE_CATEGORY
                        category_idx = None
                        transition_start_time = None

                # =========================================================
                # [추가됨] 2. 왼쪽 유저 '삭제' 기능 (왼쪽 30 / 오른쪽 0)
                # =========================================================
                elif cons_val == 30 and vowel_val == 0:
                    transition_start_time = None
                    cons_stable_start = None # 입력 타이머 중지
                    
                    if undo_start_time is None: undo_start_time = time.time()
                    elapsed = time.time() - undo_start_time
                    
                    if not undo_locked:
                        prog = min(1.0, elapsed / 2.0)
                        # 빨간색 삭제 게이지
                        cv2.rectangle(img, (w//4 - 100, h//2 - 30), (w//4 + 100, h//2 + 30), (50, 50, 50), -1)
                        cv2.rectangle(img, (w//4 - 100, h//2 - 30), (w//4 - 100 + int(200*prog), h//2 + 30), (0, 0, 255), -1)
                        img = draw_korean_text(img, "자음 삭제 중...", (w//4 - 80, h//2 - 10), (255, 255, 255))
                        
                        if elapsed >= 2.0:
                            if collected_consonants:
                                collected_consonants.pop() # 마지막 글자 삭제
                            undo_locked = True # 삭제 완료 (반복 방지)
                    else:
                        img = draw_korean_text(img, "삭제 완료", (w//4 - 60, h//2), (0, 0, 255))

                # =========================================================
                # [추가됨] 3. 오른쪽 유저 '삭제' 기능 (오른쪽 30 / 왼쪽 0)
                # =========================================================
                elif vowel_val == 30 and cons_val == 0:
                    transition_start_time = None
                    vowel_stable_start = None # 입력 타이머 중지
                    
                    if undo_start_time is None: undo_start_time = time.time()
                    elapsed = time.time() - undo_start_time
                    
                    if not undo_locked:
                        prog = min(1.0, elapsed / 2.0)
                        # 빨간색 삭제 게이지
                        cv2.rectangle(img, (3*w//4 - 100, h//2 - 30), (3*w//4 + 100, h//2 + 30), (50, 50, 50), -1)
                        cv2.rectangle(img, (3*w//4 - 100, h//2 - 30), (3*w//4 - 100 + int(200*prog), h//2 + 30), (0, 0, 255), -1)
                        img = draw_korean_text(img, "모음 삭제 중...", (3*w//4 - 80, h//2 - 10), (255, 255, 255))
                        
                        if elapsed >= 2.0:
                            if collected_vowels:
                                collected_vowels.pop() # 마지막 글자 삭제
                            undo_locked = True
                    else:
                        img = draw_korean_text(img, "삭제 완료", (3*w//4 - 60, h//2), (0, 0, 255))

                # =========================================================
                # 4. 일반 입력 모드 (기존 로직)
                # =========================================================
                else:
                    transition_start_time = None
                    undo_start_time = None # [추가됨] 삭제 타이머 초기화
                    undo_locked = False
                    
                    # --- 왼쪽(자음) 로직 ---
                    if cons_val == 0: 
                        cons_stable_start = None
                        cons_locked = False
                        img = draw_korean_text(img, "대기", (w//4, h//2 + 50), (150, 150, 150))
                    elif cons_val in CONSONANT_MAP:
                        char = CONSONANT_MAP[cons_val]
                        if cons_val == cons_last_val:
                            if cons_stable_start is None: cons_stable_start = time.time()
                            c_elapsed = time.time() - cons_stable_start
                            
                            if not cons_locked:
                                prog = min(1.0, c_elapsed / 2.0)
                                cx, cy = w//4, h//2
                                cv2.circle(img, (cx, cy), 40, (0, 255, 0), 2)
                                cv2.circle(img, (cx, cy), int(40*prog), (0, 255, 0), -1)
                                img = draw_korean_text(img, char, (cx-10, cy-20), (0,0,0), BIG_FONT)
                                
                                if c_elapsed >= 2.0:
                                    collected_consonants.append(char)
                                    cons_locked = True
                            else:
                                img = draw_korean_text(img, "입력완료", (w//4 - 40, h//2), (0, 255, 0))
                        else:
                            cons_stable_start = time.time()
                            cons_last_val = cons_val
                            cons_locked = False
                    else:
                        cons_stable_start = None
                        cons_locked = False
                    cons_last_val = cons_val

                    # --- 오른쪽(모음) 로직 ---
                    if vowel_val == 0:
                        vowel_stable_start = None
                        vowel_locked = False
                        img = draw_korean_text(img, "대기", (3*w//4, h//2 + 50), (150, 150, 150))
                    elif vowel_val in VOWEL_MAP:
                        char = VOWEL_MAP[vowel_val]
                        if vowel_val == vowel_last_val:
                            if vowel_stable_start is None: vowel_stable_start = time.time()
                            v_elapsed = time.time() - vowel_stable_start
                            
                            if not vowel_locked:
                                prog = min(1.0, v_elapsed / 2.0)
                                cx, cy = 3*w//4, h//2
                                cv2.circle(img, (cx, cy), 40, (255, 0, 255), 2)
                                cv2.circle(img, (cx, cy), int(40*prog), (255, 0, 255), -1)
                                img = draw_korean_text(img, char, (cx-10, cy-20), (0,0,0), BIG_FONT)
                                
                                if v_elapsed >= 2.0:
                                    collected_vowels.append(char)
                                    vowel_locked = True
                            else:
                                img = draw_korean_text(img, "입력완료", (3*w//4 - 40, h//2), (255, 0, 255))
                        else:
                            vowel_stable_start = time.time()
                            vowel_last_val = vowel_val
                            vowel_locked = False
                    else:
                        vowel_stable_start = None
                        vowel_locked = False
                    vowel_last_val = vowel_val

            else:
                msg = f"인식된 손: {hand_count} / 4"
                img = draw_korean_text(img, msg, (w//2 - 100, h//2), (0, 0, 255), BIG_FONT)

        # -----------------------------------------------------------------
        # 하단 정보창 (입력 화면용)
        # -----------------------------------------------------------------
        box_y = h - 150
        cv2.rectangle(img, (0, box_y), (w, h), (0, 0, 0), -1)
        cat_disp = CATEGORIES[category_idx] if category_idx is not None else "선택안됨"
        
        cons_str = "".join(collected_consonants)
        vowel_str = "".join(collected_vowels)
        
        img = draw_korean_text(img, f"주제: {cat_disp}", (20, box_y + 20), (255, 255, 0))
        img = draw_korean_text(img, f"자음: {cons_str}", (20, box_y + 80), (150, 255, 150))
        img = draw_korean_text(img, f"모음: {vowel_str}", (w//2 + 20, box_y + 80), (255, 150, 255))
        
        # -----------------------------------------------------------------
        # LLM 출력 화면
        # -----------------------------------------------------------------
        llm_canvas = np.zeros((480, 860, 3), dtype=np.uint8)
        llm_canvas[:] = (25, 25, 25) 
        cv2.rectangle(llm_canvas, (10, 10), (850, 470), (60, 60, 60), 2)
        
        llm_canvas = draw_korean_text(llm_canvas, f"주제: {cat_disp}", (30, 30), (255, 255, 0), LLM_FONT)
        
        if current_phase == PHASE_SPLIT_INPUT:
            live_input_status = f"작성 중... [ 자음: {cons_str} ]  [ 모음: {vowel_str} ]"
            color_status = (0, 255, 255) 
        else:
            live_input_status = "대기 중 (주제 선택 필요)"
            color_status = (150, 150, 150)
            
        llm_canvas = draw_korean_text(llm_canvas, live_input_status, (30, 80), color_status, LLM_FONT)
        
        cv2.line(llm_canvas, (30, 140), (830, 140), (100, 100, 100), 1)
        
        resp_lines = wrap_lines(llm_state["last_response"], 35)
        llm_canvas = draw_multiline_text(llm_canvas, resp_lines, (30, 160), 45, (255, 255, 255), LLM_FONT)

        cv2.imshow("Dual User 5-Base Input", img)
        cv2.imshow("LLM Output", llm_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()