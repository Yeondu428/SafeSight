import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from deep_translator import GoogleTranslator
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


# =========================
# 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "raw" / "images"
EMBED_DIR = DATA_DIR / "embeddings"

LORA_DIR = MODEL_DIR / "clip_lora_augmentation"
FAISS_PATH = EMBED_DIR / "image_index.faiss"
INDEX_MAP_PATH = EMBED_DIR / "index_map.csv"
METADATA_PATH = DATA_DIR / "raw" / "metadata.csv"

TEMP_DIR = BASE_DIR / "app" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def check_required_files():
    required_paths = [
        LORA_DIR,
        FAISS_PATH,
        INDEX_MAP_PATH,
        METADATA_PATH,
        IMG_DIR,
    ]

    missing = [str(path.relative_to(BASE_DIR)) for path in required_paths if not path.exists()]

    if missing:
        st.error("필수 파일 또는 폴더가 없습니다.")
        st.code("\n".join(missing))
        st.info(
            "GitHub에서 받은 뒤 models/, data/ 폴더가 제대로 들어있는지 확인해주세요.\n\n"
            "필요한 구조:\n"
            "models/clip_lora_augmentation/\n"
            "data/embeddings/image_index.faiss\n"
            "data/embeddings/index_map.csv\n"
            "data/raw/metadata.csv\n"
            "data/raw/images/"
        )
        st.stop()


# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    lora_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    lora_model.to(device)
    lora_model.eval()

    yolo_model = YOLO("yolov8n.pt")

    return lora_model, processor, yolo_model, device


# =========================
# DB 로드
# =========================
@st.cache_resource
def load_db():
    index = faiss.read_index(str(FAISS_PATH))
    index_map = pd.read_csv(INDEX_MAP_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    return index, index_map, metadata


check_required_files()

lora_model, processor, yolo_model, device = load_model()
index, index_map, metadata = load_db()


CAT_KEYWORDS = [
    "고양이",
    "묘",
    "코리안숏헤어",
    "페르시안",
    "러시안블루",
    "스핑크스",
    "랙돌",
    "샴",
]


# =========================
# 검색 함수
# =========================
def search(query=None, pil_img=None, k=5, animal_type=None):
    with torch.no_grad():
        if pil_img is not None:
            inputs = processor(images=pil_img, return_tensors="pt").to(device)

            img_out = lora_model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            emb = lora_model.visual_projection(img_out.pooler_output)
            emb_np = F.normalize(emb, dim=-1).cpu().numpy().astype("float32")

        else:
            try:
                query_en = GoogleTranslator(source="ko", target="en").translate(query)
            except Exception:
                query_en = query

            inputs = processor(
                text=[query_en],
                return_tensors="pt",
                padding=True
            ).to(device)

            txt_out = lora_model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            emb = lora_model.text_projection(txt_out.pooler_output)
            emb_np = F.normalize(emb, dim=-1).cpu().numpy().astype("float32")

    similarities, indices = index.search(emb_np, index.ntotal)

    results = []

    for sim, idx in zip(similarities[0], indices[0]):
        desertion_no = index_map.iloc[idx]["desertionNo"]

        meta_row = metadata[
            metadata["desertionNo"].astype(str) == str(desertion_no)
        ]

        if meta_row.empty:
            continue

        meta = meta_row.iloc[0]
        kind_name = str(meta.get("kindNm", ""))

        if animal_type == "🐶 강아지":
            if any(cat in kind_name for cat in CAT_KEYWORDS):
                continue

        if animal_type == "🐱 고양이":
            if not any(cat in kind_name for cat in CAT_KEYWORDS):
                continue

        img_path = IMG_DIR / f"{desertion_no}.jpg"

        results.append(
            {
                "id": str(desertion_no),
                "similarity": round(float(sim) * 100, 1),
                "kindNm": meta.get("kindNm", ""),
                "colorCd": meta.get("colorCd", ""),
                "specialMark": meta.get("specialMark", ""),
                "happenPlace": meta.get("happenPlace", ""),
                "orgNm": meta.get("orgNm", ""),
                "img_path": str(img_path),
            }
        )

        if len(results) >= k:
            break

    return results


# =========================
# Session State
# =========================
if "page" not in st.session_state:
    st.session_state.page = "main"

if "query" not in st.session_state:
    st.session_state.query = ""

if "animal_type" not in st.session_state:
    st.session_state.animal_type = ""

if "results" not in st.session_state:
    st.session_state.results = []

if "cropped_img_path" not in st.session_state:
    st.session_state.cropped_img_path = None


# =========================
# 메인 페이지
# =========================
def main_page():
    st.title("🐾 SafeSight")
    st.caption("놓치지 않는 시선, 연결되는 안전")
    st.divider()

    animal_type = st.radio(
        "동물 종류",
        ["🐶 강아지", "🐱 고양이"],
        horizontal=True,
    )

    st.divider()

    st.subheader("🔍 인상착의 입력")
    query = st.text_input(
        "찾는 반려동물 특징을 입력하세요",
        placeholder="예: 흰색 말티즈 빨간 목줄 착용",
    )

    st.subheader("📸 사진 업로드")
    image = st.file_uploader(
        "또는 사진으로 검색",
        type=["jpg", "jpeg", "png"],
    )

    if image:
        st.image(image, caption="업로드된 사진", width=200)

    st.divider()

    if st.button("🔍 탐색 시작", use_container_width=True):
        if not query and not image:
            st.warning("⚠️ 특징을 입력하거나 사진을 업로드해주세요!")
            return

        with st.spinner("🔍 탐색 중..."):
            if image is not None:
                input_image = Image.open(image).convert("RGB")
                yolo_results = yolo_model(input_image)

                cropped_image = None

                for res in yolo_results:
                    for box in res.boxes:
                        cls_id = int(box.cls[0])

                        # COCO 기준: 15 = cat, 16 = dog
                        if cls_id in [15, 16]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cropped_image = input_image.crop((x1, y1, x2, y2))
                            break

                    if cropped_image is not None:
                        break

                if cropped_image is None:
                    cropped_image = input_image

                results = search(
                    pil_img=cropped_image,
                    animal_type=animal_type,
                )

                cropped_path = TEMP_DIR / "temp_crop.jpg"
                cropped_image.save(cropped_path)
                st.session_state.cropped_img_path = str(cropped_path)

            else:
                results = search(
                    query=query,
                    animal_type=animal_type,
                )

                st.session_state.cropped_img_path = None

        st.session_state.page = "result"
        st.session_state.query = query if query else "업로드된 이미지 기준 검색"
        st.session_state.animal_type = animal_type
        st.session_state.results = results

        st.rerun()


# =========================
# 결과 페이지
# =========================
def result_page():
    if st.button("← 뒤로가기"):
        st.session_state.page = "main"
        st.rerun()

    st.title("검색 결과")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.caption(f'🔍 "{st.session_state.query}"')
        st.caption(f"🐾 {st.session_state.animal_type}")

    with col2:
        st.metric("검색 건수", f"{len(st.session_state.results)}건")

    if (
        st.session_state.cropped_img_path
        and Path(st.session_state.cropped_img_path).exists()
    ):
        st.image(
            st.session_state.cropped_img_path,
            caption="YOLOv8 탐지 및 크롭 영역",
            width=150,
        )

    st.divider()

    if not st.session_state.results:
        st.warning("검색 결과가 없습니다.")
        return

    for result in st.session_state.results:
        with st.container(border=True):
            col1, col2 = st.columns([3, 1])

            with col1:
                img_path = Path(result["img_path"])

                if img_path.exists():
                    st.image(str(img_path), width=100)
                else:
                    st.write("🐾 이미지 없음")

                st.markdown(f"**공고 #{result['id']}**")
                st.caption(f"📍 {result['happenPlace']} ({result['orgNm']})")

                tag_html = " ".join(
                    [
                        f'<span style="background:#e6f1fb;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:4px">{result["kindNm"]}</span>',
                        f'<span style="background:#eaf3de;padding:2px 8px;border-radius:10px;font-size:12px;margin-right:4px">{result["colorCd"]}</span>',
                    ]
                )

                st.markdown(tag_html, unsafe_allow_html=True)
                st.caption(f"특징: {result['specialMark']}")

            with col2:
                score = result["similarity"]
                color = "#185fa5" if score >= 50 else "#854f0b"

                st.markdown(
                    f'<div style="text-align:right;font-size:24px;font-weight:bold;color:{color}">{score}%</div>',
                    unsafe_allow_html=True,
                )

                st.progress(min(max(score / 100, 0), 1))


# =========================
# 페이지 라우팅
# =========================
if st.session_state.page == "main":
    main_page()

elif st.session_state.page == "result":
    result_page()
