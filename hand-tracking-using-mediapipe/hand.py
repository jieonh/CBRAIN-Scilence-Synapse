import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont

# --- 1. 모델 파일 경로 설정 (다운로드 받은 파일) ---
model_path = '/Users/hwangjieon/Desktop/2026CNWS/hand-tracking-using-mediapipe/hand_landmarker.task'

# --- 2. 주요 클래스 별칭 설정 ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

# --- 3. 손가락 연결 정보 직접 정의 (mp.solutions 제거용) ---
# 손목(0)과 각 손가락 마디를 연결하는 정의입니다.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),   # 중지 (손바닥 연결 포함)
    (9, 13), (13, 14), (14, 15), (15, 16), # 약지 (손바닥 연결 포함)
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # 소지 (손바닥 연결 포함)
]

# --- 3-1. 손가락 개수 -> 한글 초성/카테고리 매핑 ---
INITIAL_CONSONANTS = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"
]

CATEGORIES = [
    "감정 및 상태",
    "행동 및 동사",
    "동물",
    "스포츠",
    "날씨 및 자연"
]

def load_korean_font(font_size=48):
    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf"
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()

KOREAN_FONT = load_korean_font(48)

def draw_korean_text(img_bgr, text, position, color=(255, 255, 255), font=KOREAN_FONT):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# --- 4. 콜백 및 변수 설정 ---
detection_result = None

# --- 4-1. 초성 입력 상태 관리 ---
current_initials = []
last_right_count = None
stable_since = None
last_appended_count = None
zero_since = None
left_zero_since = None
last_completed_word = ""
last_completed_time = 0.0
last_left_count = None
left_stable_since = None
current_category = None
category_history = []
category_zero_since = None
last_category_label = "카테고리: -"

# 입력 안정화/구분 시간 (초)
LETTER_STABLE_TIME = 0.25
ZERO_GAP_TIME = 0.2
COMPLETED_DISPLAY_TIME = 1.5
LEFT_WORD_END_TIME = 0.4
CATEGORY_STABLE_TIME = 0.25
CATEGORY_DONE_TIME = 1.0

def print_result(result, output_image, timestamp_ms):
    global detection_result
    detection_result = result

# --- 5. Landmarker 옵션 설정 ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=4,
    result_callback=print_result
)

# --- 6. 메인 실행 루프 ---
cap = cv2.VideoCapture(0)

# 'with' 문을 사용하여 리소스를 안전하게 관리합니다.
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # mp.Image 변환
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        
        # 타임스탬프 (ms)
        frame_timestamp_ms = int(time.time() * 1000)
        
        # 감지 실행 (비동기)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # 결과 그리기
        total_left_fingers = 0
        total_right_fingers = 0
        left_hand_count = 0
        right_hand_count = 0
        if detection_result and detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                h, w, c = img.shape
                split_x = w // 2
                
                # 왼손/오른손 구분
                hand_type = "Unknown"
                if detection_result.handedness and idx < len(detection_result.handedness):
                    if detection_result.handedness[idx]:
                        hand_type = detection_result.handedness[idx][0].category_name
                
                # 손가락 개수 계산
                fingers_up = 0
                # 엄지
                if hand_type == "Right":
                    fingers_up += 1 if hand_landmarks[4].x > hand_landmarks[3].x else 0
                else:
                    fingers_up += 1 if hand_landmarks[4].x < hand_landmarks[3].x else 0
                # 나머지 손가락들
                for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    fingers_up += 1 if hand_landmarks[tip_id].y < hand_landmarks[pip_id].y else 0
                
                # 화면 좌/우 분할 기준으로 카운트
                wrist_x = int(hand_landmarks[0].x * w)
                is_left_side = wrist_x < split_x
                if is_left_side:
                    total_left_fingers += fingers_up
                    left_hand_count += 1
                else:
                    total_right_fingers += fingers_up
                    right_hand_count += 1
                
                # (A) 뼈대 그리기 (정의한 HAND_CONNECTIONS 사용)
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    # 좌표 가져오기
                    start_lm = hand_landmarks[start_idx]
                    end_lm = hand_landmarks[end_idx]
                    
                    # 픽셀 좌표 변환
                    x1, y1 = int(start_lm.x * w), int(start_lm.y * h)
                    x2, y2 = int(end_lm.x * w), int(end_lm.y * h)
                    
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # (B) 손가락 끝점 그리기
                finger_tips = [4, 8, 12, 16, 20]
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # 손가락 끝 강조
                    if id in finger_tips:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                
                # 손가락 개수 표시 (화면 좌/우 영역에 표시)
                info_text = f"{hand_type}: {fingers_up}"
                if is_left_side:
                    cv2.putText(img, info_text, (10, 40 + (left_hand_count - 1) * 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(img, info_text, (split_x + 10, 40 + (right_hand_count - 1) * 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        # 분할선 표시
        h, w, _ = img.shape
        split_x = w // 2
        cv2.line(img, (split_x, 0), (split_x, h), (0, 255, 255), 2)

        # 좌/우 총합 표시
        cv2.putText(img, f"Left Total: {total_left_fingers}", (10, 40 + left_hand_count * 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        cv2.putText(img, f"Right Total: {total_right_fingers}", (split_x + 10, 40 + right_hand_count * 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        now = time.time()

        # 초성 입력 상태 업데이트 (오른쪽 손가락 개수 기준)
        if total_right_fingers != last_right_count:
            last_right_count = total_right_fingers
            stable_since = now

        if total_right_fingers == 0:
            if zero_since is None:
                zero_since = now
        else:
            zero_since = None

        if 1 <= total_right_fingers <= len(INITIAL_CONSONANTS):
            if stable_since is not None and (now - stable_since) >= LETTER_STABLE_TIME:
                if last_appended_count != total_right_fingers:
                    current_initials.append(INITIAL_CONSONANTS[total_right_fingers - 1])
                    last_appended_count = total_right_fingers

        if zero_since is not None and (now - zero_since) >= ZERO_GAP_TIME:
            last_appended_count = None

        # 왼손 주먹(0)으로 단어 종료 (왼손이 실제로 잡힐 때만)
        if left_hand_count > 0 and total_left_fingers == 0:
            if left_zero_since is None:
                left_zero_since = now
        else:
            left_zero_since = None

        if left_zero_since is not None and (now - left_zero_since) >= LEFT_WORD_END_TIME and current_initials:
            last_completed_word = " ".join(current_initials)
            last_completed_time = now
            current_initials = []
            last_appended_count = None
            stable_since = None

        # 왼쪽 화면: 카테고리 매핑 (1~5)
        if total_left_fingers != last_left_count:
            last_left_count = total_left_fingers
            left_stable_since = now

        if left_hand_count > 0 and 1 <= total_left_fingers <= len(CATEGORIES):
            if left_stable_since is not None and (now - left_stable_since) >= CATEGORY_STABLE_TIME:
                current_category = CATEGORIES[total_left_fingers - 1]
                last_category_label = f"카테고리({total_left_fingers}): {current_category}"

        if left_hand_count > 0 and total_left_fingers == 0:
            if category_zero_since is None:
                category_zero_since = now
        else:
            category_zero_since = None

        if category_zero_since is not None and (now - category_zero_since) >= CATEGORY_DONE_TIME and current_category:
            if not category_history or category_history[-1] != current_category:
                category_history.append(current_category)
            category_zero_since = None

        if 1 <= total_left_fingers <= len(CATEGORIES):
            left_category = CATEGORIES[total_left_fingers - 1]
        else:
            left_category = "?"
        img = draw_korean_text(img, last_category_label, (10, h - 60), color=(0, 255, 255))

        category_history_text = "카테고리 기록: " + (", ".join(category_history) if category_history else "-")
        img = draw_korean_text(img, category_history_text, (10, h - 110), color=(0, 255, 255))

        # 오른쪽 화면: 초성 매핑 (1~19)
        if 1 <= total_right_fingers <= len(INITIAL_CONSONANTS):
            right_initial = INITIAL_CONSONANTS[total_right_fingers - 1]
        else:
            right_initial = "?"
        right_label = f"초성({total_right_fingers}): {right_initial}"
        img = draw_korean_text(img, right_label, (split_x + 10, h - 60), color=(0, 255, 255))

        # 입력 중인 초성 시퀀스 표시 (오른쪽 화면)
        input_text = "입력: " + (", ".join(current_initials) if current_initials else "-")
        img = draw_korean_text(img, input_text, (split_x + 10, h - 110), color=(0, 255, 255))

        # 완료된 단어 표시 (짧게)
        if last_completed_word and (now - last_completed_time) <= COMPLETED_DISPLAY_TIME:
            completed_text = f"완료: {last_completed_word}"
            img = draw_korean_text(img, completed_text, (split_x + 10, h - 160), color=(255, 200, 0))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()