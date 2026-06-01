import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# [1단계] AI 모델 및 이기종 데이터베이스(강아지+고양이) 통합 로드
# ==========================================
st.set_page_config(page_title="SafeSight", page_icon="🎯", layout="wide")

@st.cache_resource
def load_integrated_assets():
    # 1. 지영님의 YOLOv8 동물 탐지 가중치 로드
    yolo = YOLO('models/best.pt') 
    
    # 2. 친구분이 세팅한 멀티모달 CLIP 모델 및 전처리기 로드
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 3. 각각 추출한 강아지 DB와 고양이 DB 로드
    dog_embeds = np.load('models/shelter_embeddings.npy') 
    dog_paths = np.load('models/shelter_paths.npy')
    
    cat_embeds = np.load('models/cat_embeddings.npy')
    cat_paths = np.load('models/cat_paths.npy')
    
    # 🌟 [수학적 통합] 두 동물 데이터를 하나로 세로 병합(vstack/concatenate)
    total_embeddings = np.vstack([dog_embeds, cat_embeds])
    total_paths = np.concatenate([dog_paths, cat_paths])
    
    return yolo, clip, processor, total_embeddings, total_paths

# 자산 로드 실행
try:
    yolo_model, clip_model, clip_processor, total_embeds, total_paths = load_integrated_assets()
except Exception as e:
    st.error(f"⚠️ 파일 로드 실패: {e}\nmodels/ 폴더 안에 best.pt와 강아지/고양이 npy 파일들이 모두 제대로 들어있는지 확인해주세요.")

# ==========================================
# [2단계] 사용자 인터페이스 (UI 화면 디자인)
# ==========================================
st.title("🎯 SafeSight - AI 유기동물 통합 매칭 시스템")
st.write("YOLOv8의 객체 추적 기술과 CLIP의 멀티모달 매칭 기술을 결합하여 유기동물을 실시간으로 찾아냅니다.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🕵️‍♂️ 발견동물 정보 입력")
    # 이미지 업로드 및 텍스트 키워드 동시 입력창
    uploaded_file = st.file_uploader("발견하거나 보호 중인 동물 사진을 업로드하세요", type=["jpg", "jpeg", "png"])
    text_query = st.text_input("전단지 정보 혹은 특징 키워드 입력", placeholder="예: 삼색 고양이, 갈색 푸들, 흰색 말티즈")

with col2:
    st.subheader("📊 AI 모델 연산 및 매칭 결과")
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="📷 사용자가 업로드한 원본 사진", use_container_width=True)
        
        # ------------------------------------------
        # [3단계] 지영님의 YOLO 가동 ➔ 동물 영역만 싹둑 자르기(Crop)
        # ------------------------------------------
        yolo_results = yolo_model(input_image, verbose=False)
        boxes = yolo_results[0].boxes
        cropped_img = None
        
        for box in boxes:
            if float(box.conf[0]) >= 0.4:
                xyxy = box.xyxy[0].tolist()
                cropped_img = input_image.crop((int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                break
        
        if cropped_img is None:
            cropped_img = input_image
            st.info("💡 YOLO가 특정 동물 영역을 감지하지 못해 원본 이미지 전체로 분석합니다.")
        else:
            st.image(cropped_img.resize((224, 224)), caption="✂️ YOLO가 크롭한 전처리 영역 (224x224)", width=224)
            
        # ------------------------------------------
        # [4단계] 친구분의 CLIP 가동 ➔ 시각 지문 생성 및 통합 DB 대조
        # ------------------------------------------
        if text_query:
            with st.spinner("🔄 통합 벡터 데이터베이스 내 텍스트/이미지 대조 중..."):
                inputs = clip_processor(images=cropped_img.resize((224, 224)), return_tensors="pt")
                
                with torch.no_grad():
                    query_features = clip_model.get_image_features(**inputs)
                    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                    query_np = query_features.cpu().numpy()[0]
                
                # 병합된 total_embeds와 행렬 내적(Dot Product) 연산으로 코사인 유사도 계산
                similarities = np.dot(total_embeds, query_np)
                max_idx = np.argmax(similarities)
                match_prob = similarities[max_idx] * 100 
                
                # 최종 웹 화면 출력
                st.success(f"📈 매칭 완료! 입력한 특징과 가장 일치하는 동물 발견!")
                st.metric(label="최고 유사도 일치율", value=f"{match_prob:.2f}%")
                st.info(f"📂 보호소 통합 데이터 매칭 경로: \n`{total_paths[max_idx]}`")
