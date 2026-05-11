# 🐾SafeSight
> "놓치지 않는 시선, 연결되는 안전"
> 텍스트 기반 멀티모달 AI 반려동물 실종 탐색 시스템

## 📌 프로젝트 소개
사용자가 자연어로 인상착의를 입력하면 CLIP 기반 시맨틱 검색으로
유기동물 보호소 공고에서 유사한 반려동물을 탐색하는 AI 시스템입니다.

## 🛠 기술 스택
- **Detection**: YOLOv8
- **Embedding/Search**: CLIP (ViT-B/32) + FAISS
- **Web UI**: Streamlit
- **학습 프레임워크**: PyTorch

## 👥 팀원
| 이름 | 역할 |
|---|---|
| 신연주 | CLIP 임베딩 / 시맨틱 검색 |
| 민지영 | 데이터 수집 / YOLO Detection |

## 📁 프로젝트 구조
SafeSight/
  📁 notebooks/
    📓 00_setup.ipynb            # 환경 세팅
    📓 01_data_collection.ipynb  # 데이터 수집 (민지영)
    📓 02_data_preprocess.ipynb  # 데이터 전처리 (민지영)
    📓 03_yolo_train.ipynb       # YOLO 학습 (민지영)
    📓 04_clip_embedding.ipynb   # CLIP 임베딩 DB 구축 (연주)
    📓 05_clip_search.ipynb      # 텍스트 검색 (연주)
    📓 06_evaluation.ipynb       # 성능 평가 (공통)
  📁 data/
    📁 raw/                      # 원본 데이터
    📁 yolo/                     # YOLO 학습용
      📁 images/train val test
      📁 labels/train val test
      📄 dataset.yaml
    📁 embeddings/               # CLIP 벡터 저장
  📁 models/                     # 학습된 가중치
  📁 app/                        # Streamlit 웹앱
    📄 app.py
  📄 requirements.txt

## 🚀 파이프라인

사용자 텍스트 입력 ("흰색 말티즈 빨간 목줄")
        ↓
CLIP 텍스트 인코더 → 512차원 벡터
        ↓
FAISS DB에서 유사도 검색
        ↓
YOLO로 바운딩 박스 위치 표시
        ↓
유사도 순 상위 5개 결과 출력


## 📊 데이터셋
- 국가동물보호시스템 API (유기동물 공고 이미지 + 특징 텍스트)
- Oxford-IIIT Pet Dataset (YOLO Baseline)
- AI Hub 반려동물 이미지 데이터

## 📅 개발 일정
| 주차 | 내용 |
|---|---|
| 10주 | 데이터 수집 및 Baseline 구축 |
| 11주 | CLIP 임베딩 + YOLO 학습 |
| 12주 | 웹앱 통합 및 중간 데모 |
| 13주 | 예외처리 및 시스템 안정화 |
| 14주 | 최종 데모 리허설 |
| 16주 | 최종 발표 및 제출 |
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✅ README.md 작성 완료!")
