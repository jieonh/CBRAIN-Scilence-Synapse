import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. 모델 파일 경로 설정 (다운로드 받은 파일) ---
model_path = '/Users/ryu/Documents/Dev/2026CNWS/hand-tracking-using-mediapipe/hand_landmarker.task'

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

# --- 4. 콜백 및 변수 설정 ---
detection_result = None

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
pTime = 0

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
        if detection_result and detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, c = img.shape
                
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
                    
                    # 모든 관절에 작은 점 찍기 (선택사항)
                    # cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

                    # 손가락 끝 강조
                    if id in finger_tips:
                        print(f"ID: {id}, X: {cx}, Y: {cy}")
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # FPS 표시
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()