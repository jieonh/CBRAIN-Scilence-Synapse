import cv2
import time
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import ImageFont, ImageDraw, Image

# ------------------- 한글 오토마타 -------------------
class HangulAssembler:
    def __init__(self):
        self.CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
        self.JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
        self.JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
        self.DOUBLE_JUNG = {'ㅗㅏ':'ㅘ', 'ㅗㅐ':'ㅙ', 'ㅗㅣ':'ㅚ', 'ㅜㅓ':'ㅝ', 'ㅜㅔ':'ㅞ', 'ㅜㅣ':'ㅟ', 'ㅡㅣ':'ㅢ'}

    def compose_text(self, input_jamos):
        res = ""
        cho, jung, jong = None, None, None
        
        for jamo in input_jamos:
            if jamo in self.CHO:
                if cho is not None:
                    if jung is None: 
                        res += cho
                        cho = jamo
                    elif jong is not None: 
                         res += self.combine(cho, jung, jong)
                         cho, jung, jong = jamo, None, None
                    else: 
                        if jamo in self.JONG:
                            jong = jamo
                        else: 
                            res += self.combine(cho, jung, None)
                            cho, jung, jong = jamo, None, None
                else:
                    cho = jamo
            elif jamo in self.JUNG:
                if cho is None: res += jamo
                elif jung is None: jung = jamo
                elif jong is None:
                    combined_jung = self.DOUBLE_JUNG.get(jung + jamo)
                    if combined_jung: jung = combined_jung
                    else:
                        res += self.combine(cho, jung, None)
                        cho, jung, jong = None, jamo, None
                else:
                    res += self.combine(cho, jung, None)
                    cho = jong
                    jung = jamo
                    jong = None
            
        if cho is not None: res += self.combine(cho, jung, jong)
        elif jung is not None: res += jung
        return res

    def combine(self, c, j, k):
        if c is None: return ""
        if j is None: return c
        try:
            c_idx = self.CHO.index(c)
            j_idx = self.JUNG.index(j)
            k_idx = self.JONG.index(k) if k else 0
            code = 0xAC00 + (c_idx * 588) + (j_idx * 28) + k_idx
            return chr(code)
        except: return c + (j if j else "") + (k if k else "")

# ------------------- 설정 변수 -------------------
MODEL_PATH = 'hand_landmarker.task'
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc" # 폰트 경로 확인 필요
CLICK_THRESHOLD = 0.5
CURSOR_MEMORY_TIME = 0.15 

KEY_W, KEY_H = 80, 75
KEY_PADDING = 8
START_X, START_Y = 100, 250 # 숫자줄이 없어져서 조금 아래로 내림 (원하면 조절 가능)

COLOR_BG = (50, 50, 50) 
COLOR_NORMAL = (255, 255, 255) 
COLOR_DISABLED = (80, 80, 80) 
COLOR_HOVER = (255, 0, 255)    
COLOR_CLICK = (0, 255, 0)      
COLOR_TEXT = (255, 255, 255)   
COLOR_SHIFT = (255, 215, 0) 
COLOR_CURSOR_OUTER = (0, 255, 255)

# [수정됨] 숫자열 삭제됨
keys_layout = [
    ['ㅂ', 'ㅈ', 'ㄷ', 'ㄱ', 'ㅅ', 'ㅛ', 'ㅕ', 'ㅑ', 'ㅐ', 'ㅔ'],
    ['ㅁ', 'ㄴ', 'ㅇ', 'ㄹ', 'ㅎ', 'ㅗ', 'ㅓ', 'ㅏ', 'ㅣ'],
    ['Shift', 'ㅋ', 'ㅌ', 'ㅊ', 'ㅍ', 'ㅠ', 'ㅜ', 'ㅡ', '지움']
]
SHIFT_MAP = {'ㅂ':'ㅃ', 'ㅈ':'ㅉ', 'ㄷ':'ㄸ', 'ㄱ':'ㄲ', 'ㅅ':'ㅆ', 'ㅐ':'ㅒ', 'ㅔ':'ㅖ'}
HOMING_KEYS = ['ㄹ', 'ㅓ'] 

# ------------------- MediaPipe 설정 -------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3, 
    min_hand_presence_confidence=0.3, 
    min_tracking_confidence=0.3
)

# ------------------- 클래스 -------------------
class Button:
    def __init__(self, pos, text, size):
        self.pos = pos
        self.size = size
        self.base_text = text 
        self.current_text = text 
        self.hover_start_time = 0
        self.is_triggered = False
        self.is_hovered = False
        self.enabled = True
    
    def get_rect(self):
        return int(self.pos[0]), int(self.pos[1]), int(self.pos[0] + self.size[0]), int(self.pos[1] + self.size[1])
    
    def update_state(self, pointers, current_time, is_shift_on):
        if is_shift_on and self.base_text in SHIFT_MAP:
            self.current_text = SHIFT_MAP[self.base_text]
        else:
            self.current_text = self.base_text

        if not self.enabled: return None

        x1, y1, x2, y2 = self.get_rect()
        self.is_hovered = False
        
        for (px, py) in pointers:
            if x1 < px < x2 and y1 < py < y2:
                self.is_hovered = True
                break
        
        if self.is_hovered:
            if self.hover_start_time == 0:
                self.hover_start_time = current_time
            
            elapsed = current_time - self.hover_start_time
            if elapsed > CLICK_THRESHOLD and not self.is_triggered:
                self.is_triggered = True
                return self.current_text 
        else:
            self.hover_start_time = 0
            self.is_triggered = False
        return None

def draw_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# --- 초기화 ---
assembler = HangulAssembler()
buttonList = []
current_y = START_Y

for i, row in enumerate(keys_layout):
    row_offset = 0
    # [수정됨] 인덱스 변경 (0: ㅂ라인, 1: ㅁ라인, 2: Shift라인)
    if i == 1: 
        row_offset = KEY_W // 3 # 두번째 줄(ㅁㄴㅇㄹ...) 들여쓰기
    
    current_x = START_X + row_offset
    for j, key in enumerate(row):
        current_key_w = KEY_W
        if key == '지움': current_key_w = int(KEY_W * 1.8) 
        elif key == 'Shift': current_key_w = int(KEY_W * 1.5)
        buttonList.append(Button([current_x, current_y], key, [current_key_w, KEY_H]))
        current_x += current_key_w + KEY_PADDING
    current_y += KEY_H + KEY_PADDING

reset_btn = Button([1100, 30], "초기화", [150, 60])

# ------------------- 메인 실행 -------------------
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    input_jamos = [] 
    display_text = "" 
    is_shift_on = False 
    
    last_known_pointers = []
    last_pointer_update_time = 0

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1) 
        ui_img = np.full(img.shape, COLOR_BG, dtype=np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        curr_time = time.time()
        frame_timestamp_ms = int(curr_time * 1000)
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        # 1. 포인터 (손바닥 중심)
        active_pointers = []
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                palm_center = hand_landmarks[9] 
                cx, cy = int(palm_center.x * img.shape[1]), int(palm_center.y * img.shape[0])
                active_pointers.append((cx, cy))
            
            last_known_pointers = active_pointers
            last_pointer_update_time = curr_time
        else:
            if curr_time - last_pointer_update_time < CURSOR_MEMORY_TIME:
                active_pointers = last_known_pointers
            else:
                active_pointers = []

        # 커서 그리기
        for (cx, cy) in active_pointers:
            cv2.circle(ui_img, (cx, cy), 6, COLOR_HOVER, cv2.FILLED)
            cv2.circle(ui_img, (cx, cy), 15, COLOR_CURSOR_OUTER, 2) 
            cv2.line(ui_img, (cx-20, cy), (cx+20, cy), COLOR_CURSOR_OUTER, 1)
            cv2.line(ui_img, (cx, cy-20), (cx, cy+20), COLOR_CURSOR_OUTER, 1)

        # 2. 정보 표시
        cv2.rectangle(ui_img, (45, 80), (800, 150), COLOR_NORMAL, 2)
        display_text = assembler.compose_text(input_jamos)
        ui_img = draw_text(ui_img, display_text, (55, 90), 50, COLOR_TEXT)
        
        # 3. 초기화 버튼
        if reset_btn.update_state(active_pointers, curr_time, False):
            input_jamos = [] 
            is_shift_on = False
            print("Text Cleared")
        
        rx1, ry1, rx2, ry2 = reset_btn.get_rect()
        r_color = COLOR_CLICK if reset_btn.is_triggered else (COLOR_HOVER if reset_btn.is_hovered else COLOR_NORMAL)
        cv2.rectangle(ui_img, (rx1, ry1), (rx2, ry2), r_color, 2)
        ui_img = draw_text(ui_img, "", (rx1+20, ry1+15), 30, COLOR_TEXT)
        
        if reset_btn.is_hovered:
             elapsed = curr_time - reset_btn.hover_start_time
             bw = int((rx2-rx1) * (elapsed / CLICK_THRESHOLD))
             cv2.rectangle(ui_img, (rx1, ry2-5), (rx1+bw, ry2), COLOR_CLICK, cv2.FILLED)

        # 4. 키보드 버튼 로직
        for button in buttonList:
            if button.base_text == 'Shift': button.enabled = True

            clicked_char = button.update_state(active_pointers, curr_time, is_shift_on)
            
            if clicked_char:
                if clicked_char == 'Shift': 
                    is_shift_on = not is_shift_on
                elif clicked_char == '지움':
                    if input_jamos: input_jamos.pop()
                else:
                    input_jamos.append(clicked_char)
                    if is_shift_on: is_shift_on = False 

            # 버튼 그리기
            x1, y1, x2, y2 = button.get_rect()
            
            border_color = COLOR_NORMAL
            border_thick = 2
            if button.base_text == 'Shift' and is_shift_on:
                border_color = COLOR_SHIFT 
                border_thick = 4

            if button.is_triggered:
                cv2.rectangle(ui_img, (x1, y1), (x2, y2), COLOR_CLICK, 8)
            elif button.is_hovered:
                cv2.rectangle(ui_img, (x1, y1), (x2, y2), COLOR_HOVER, 4)
                elapsed = curr_time - button.hover_start_time
                bw = int((x2 - x1 - 10) * (elapsed / CLICK_THRESHOLD))
                if bw > 0:
                    cv2.rectangle(ui_img, (x1+5, y2-10), (x1+5+bw, y2-5), COLOR_CLICK, cv2.FILLED)
            else:
                cv2.rectangle(ui_img, (x1, y1), (x2, y2), border_color, border_thick)
            
            # 버튼 텍스트
            # t_size = 35
            # text_x = x1 + (KEY_W // 2) - (t_size // 2) - 5
            # text_y = y1 + (KEY_H // 2) - (t_size // 2) - 5
            # ui_img = draw_text(ui_img, button.current_text, (text_x, text_y), t_size, COLOR_TEXT)
            
            # 기준 돌기 표시 (ㄹ, ㅓ)
            if button.base_text in HOMING_KEYS:
                cx_key = x1 + int(button.size[0] / 2)
                cy_key = y1 + int(button.size[1] / 2) + 20
                cv2.circle(ui_img, (cx_key, cy_key), 4, (200, 200, 200), cv2.FILLED)

        cv2.imshow("Simple Hangul Keyboard", ui_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()