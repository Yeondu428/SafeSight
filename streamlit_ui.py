import streamlit as st
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

# 1. 페이지 설정
st.set_page_config(page_title="SafeSight", page_icon="🎯", layout="wide")

# 2. AI 모델 및 통합 벡터 데이터베이스 로드
@st.cache_resource
def load_assets():
    # 강아지를 학습했던 YOLO 모델 로드
    yolo = YOLO('models/best.pt') 
    
    # 멀티모달 CLIP 모델 로드
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # [💡 핵심] 강아지와 고양이 데이터베이스를 각각 로드하여 하나로 합치기!
    # 기존에 가지고 계시던 강아지 npy 파일명에 맞게 shelter 부분을 수정하셔도 됩니다.
    dog_embeds = np.load('models/shelter_embeddings.npy') 
    dog_paths = np.load('models/shelter_paths.npy')
    
    cat_embeds = np.load('models/cat_embeddings.npy')
    cat_paths = np.load('models/cat_paths.npy')
    
    # 수학적으로 두 행렬을 하나로 병합 (통합 벡터 DB 구축)
    total_embeddings = np.vstack([dog_embeds, cat_embeds])
    total_paths = np.concatenate([dog_paths, cat_paths])
    
    return yolo, clip, processor, total_embeddings, total_paths

try:
    yolo_model, clip_model, clip_processor, total_embeds, total_paths = load_assets()
except Exception as e:
    st.error(f"⚠️ 에러 발생: {e}. 깃허브 models/ 폴더에 npy 파일들과 best.pt가 모두 있는지 확인해주세요.")

# 3. UI 화면 구성
st.title("🎯 SafeSight - AI 유기동물 매칭 시스템")
st.write("강아지 2만 장의 지식과 새로운 고양이 데이터베이스가 통합된 버전입니다.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🕵️‍♂️ 발견한 동물 이미지 업로드")
    uploaded_file = st.file_uploader("보호소나 길가에서 발견한 동물 사진을 올려주세요", type=["jpg", "jpeg", "png"])
    text_query = st.text_input("찾고자 하는 전단지 속 특징 키워드 입력", placeholder="예: 삼색 고양이, 갈색 푸들, 빨간 목줄")

with col2:
    st.subheader("📊 AI 객체 크롭 및 멀티모달 분석 결과")
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="원본 발견 이미지", use_container_width=True)
        
        # YOLO 실시간 전처리 크롭
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
            st.info("💡 특정 구역이 크롭되지 않아 원본 전체 구역을 분석합니다.")
        else:
            st.image(cropped_img.resize((224, 224)), caption="YOLO가 전처리한 영역", width=224)
            
        # 특징 매칭 연산
        if text_query:
            with st.spinner("통합 벡터 데이터베이스 대조 중..."):
                inputs = clip_processor(images=cropped_img.resize((224, 224)), return_tensors="pt")
                with torch.no_grad():
                    query_features = clip_model.get_image_features(**inputs)
                    query_features = query_features / query_features.norm(dim=-1, keepdim=True)
                    query_np = query_features.cpu().numpy()[0]
                
                # 코사인 유사도 연산으로 통합 DB에서 매칭 확률 계산
                similarities = np.dot(total_embeds, query_np)
                max_idx = np.argmax(similarities)
                match_prob = similarities[max_idx] * 100
                
                st.success(f"📈 매칭 완료! 입력하신 특징과 가장 유사한 동물의 일치율: **{match_prob:.2f}%**")
                st.write(f"📂 매칭된 보호소 파일 경로: `{total_paths[max_idx]}`")
