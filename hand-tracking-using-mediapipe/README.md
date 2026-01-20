# 손 제스처 기반 스피드퀴즈 정답 추론기 (MediaPipe + LLM)

이 프로젝트는 웹캠에서 손을 실시간으로 추적하고, 손가락 개수 조합으로 **카테고리와 초성**을 입력해 **스피드퀴즈 정답 후보 TOP 5**를 추론하는 인터랙티브 시스템입니다.  
MediaPipe 손 랜드마커로 손을 인식하고, 입력 안정화 로직을 통해 신뢰도 있는 입력을 만든 뒤, OpenAI 또는 Ollama 기반 LLM으로 결과를 생성합니다.

---
## 주요 기능

- **실시간 손 추적**: MediaPipe Hand Landmarker를 사용해 21개 랜드마크 추출
- **손가락 개수 인식**: 엄지/나머지 손가락 판별 로직으로 0~5 카운트
- **좌/우 영역 분리 입력**
  - 왼쪽 화면: 카테고리 선택(1~5)
  - 오른쪽 화면: 초성 입력(1~19)
- **입력 안정화 로직**
  - 일정 시간 유지 시 입력 확정
  - 0 손가락 상태로 입력 구분
- **퀴즈 타이머**: 제한 시간 내 입력 수행(기본 5분)
- **LLM 추론 결과 표시**: 별도 창에서 후보 TOP 5 출력
- **한글 UI 렌더링**: PIL + OpenCV로 한글 텍스트 출력

---
## 프로젝트 구조

```
hand-tracking-using-mediapipe/
├─ hand.py                 # 메인 실행: 손 인식 + 입력 처리 + UI + LLM 호출
├─ llm_pipeline.py         # LLM 요청/응답 파이프라인
├─ prompt.txt              # LLM 프롬프트 템플릿
├─ hand_landmarker.task    # MediaPipe 손 랜드마커 모델
├─ requirements.txt        # Python 의존성 목록
├─ Screenshot.png          # 실행 결과 캡처
└─ hand-landmarks.png      # 손 랜드마크 예시 이미지
```

---
## 동작 개념 (요약)

1. **MediaPipe**가 손 랜드마크를 실시간으로 감지  
2. 각 손가락의 상태로 **손가락 개수** 계산  
3. 화면 좌/우 위치를 기준으로 **카테고리/초성** 입력  
4. 입력이 일정 시간 유지되면 확정  
5. LLM에 `카테고리 + 초성` 조합 전달  
6. 정답 후보 TOP 5를 별도 창에 출력

---
## 카테고리/초성 매핑

### 카테고리 (왼쪽 화면, 1~5)
1) 감정 및 상태  
2) 행동 및 동사  
3) 동물  
4) 스포츠  
5) 날씨 및 자연  

### 초성 (오른쪽 화면, 1~19)
1) ㄱ  2) ㄴ  3) ㄷ  4) ㄹ  5) ㅁ  
6) ㅂ  7) ㅅ  8) ㅇ  9) ㅈ  10) ㅊ  
11) ㅋ 12) ㅌ 13) ㅍ 14) ㅎ 15) ㄲ  
16) ㄸ 17) ㅃ 18) ㅆ 19) ㅉ  

---
## 실행 방법

### 1) 환경 준비
Python 3.7 이상 설치 후 의존성 설치:
```
pip install -r requirements.txt
```

### 2) 실행
```
python hand.py
```

### 3) 종료
`q` 키 입력

---
## LLM 설정

### OpenAI 사용
- 환경변수 `OPENAI_API_KEY` 설정 **또는**
- `openai_api_key.txt` 파일에 키 저장

### Ollama 사용
- 환경변수 `LLM_PROVIDER=ollama`
- `OLLAMA_URL`, `OLLAMA_MODEL` 설정 가능

---
## 설정 가능한 주요 값 (hand.py)

- `QUIZ_DURATION`: 퀴즈 제한 시간(초)
- `LETTER_STABLE_TIME`: 초성 입력 안정화 시간
- `ZERO_GAP_TIME`: 입력 분리용 0손가락 유지 시간
- `CATEGORY_STABLE_TIME`: 카테고리 확정 시간

