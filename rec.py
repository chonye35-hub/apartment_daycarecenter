# refactored_app.py

import requests
import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import torch
from PIL import Image
import os
import json
import re

# --- 1. 라이브러리 임포트 ---
from mobile_sam import sam_model_registry, SamPredictor
from streamlit_drawable_canvas import st_canvas

# --- 2. 상수 정의 ---
# 파일 경로들을 상단에 상수로 정의하여 관리 용이성을 높입니다.
IMAGE_PATH = "input.jpg"
MODEL_PATH = "mobile_sam.pt"
JSON_PATH = "colors_lab.json"

# ===================================================================
# 헬퍼 함수 (Helper Functions)
# ===================================================================

def normalize_brightness(image_bgr: np.ndarray) -> np.ndarray:
    """HSV 색상 공간을 사용하여 이미지의 밝기를 평탄화합니다."""
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    v_eq = cv2.equalizeHist(v)
    image_hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(image_hsv_eq, cv2.COLOR_HSV2BGR)

def extract_lab_palette(image_bgr: np.ndarray, mask: np.ndarray, n_colors: int = 5) -> np.ndarray:
    """마스크 영역에서 LAB 색상 팔레트를 추출합니다."""
    pixels_in_mask = image_bgr[mask]
    if len(pixels_in_mask) == 0:
        raise ValueError("마스크 영역에 픽셀이 없습니다.")
    masked_lab = cv2.cvtColor(pixels_in_mask.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(masked_lab)
    return kmeans.cluster_centers_

def create_palette_image(lab_colors: np.ndarray, block_size: tuple = (50, 50)) -> np.ndarray:
    """LAB 색상 배열로 팔레트 이미지를 생성합니다."""
    bgr_colors = []
    for c in lab_colors:
        L, a, b = np.clip(c, 0, 255)
        lab_pixel = np.uint8([[[L, a, b]]])
        bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)[0, 0]
        bgr_colors.append(bgr)
    h, w = block_size[1], block_size[0] * len(bgr_colors)
    palette_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(bgr_colors):
        palette_img[:, i * block_size[0]:(i + 1) * 100] = color
    return palette_img

def calc_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """두 LAB 팔레트 간의 평균 유클리드 거리를 계산합니다."""
    return np.mean(np.linalg.norm(np.array(lab1) - np.array(lab2), axis=1))

def get_top_similar_images(ref_lab: np.ndarray, all_data: list, top_n: int = 3, exclude: list = []) -> list:
    """참조 팔레트와 가장 유사한 이미지들을 찾습니다."""
    candidates = [e for e in all_data if re.search(r'A \(\d+\)_2.png$', e['name']) and e['raw'] not in exclude]
    dists = [(calc_distance(ref_lab, e['lab']), e) for e in candidates]
    dists.sort(key=lambda x: x[0])
    return dists[:top_n]


# ===================================================================
# 데이터 및 모델 로딩 함수
# ===================================================================

@st.cache_resource
def load_sam_predictor(model_path: str) -> SamPredictor:
    """MobileSAM 모델을 로드하고 Predictor를 반환합니다. (클라우드 배포를 위해 다운로드 로직 유지)"""
    if not os.path.exists(model_path):
        with st.spinner("☁️ AI 모델을 다운로드 중입니다... (최초 1회)"):
            url = "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                st.error(f"🚨 모델 다운로드 중 오류 발생: {e}")
                st.stop()
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry[model_type](checkpoint=model_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    return SamPredictor(mobile_sam)

@st.cache_data
def load_all_json_data(json_path: str) -> dict:
    """추천을 위한 JSON 데이터를 로드합니다."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ===================================================================
# UI 컴포넌트 함수
# ===================================================================

def display_area_selection(predictor, image, image_np):
    """영역 선택 UI를 표시하고, 분석 결과를 반환하는 함수"""
    st.info("👇 아래 이미지에서 색상을 추출할 영역을 클릭하여 선택하세요.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data and canvas_result.json_data["objects"]:
        points = [(obj["left"], obj["top"]) for obj in canvas_result.json_data["objects"]]
        if points:
            masks, _, _ = predictor.predict(np.array(points), np.ones(len(points)), multimask_output=False)
            st.session_state.mask = masks[0]

    if st.button("✅ 이 영역으로 분석 및 추천 실행", type="primary"):
        if "mask" in st.session_state and st.session_state.mask is not None:
            with st.spinner("색상 분석 및 유사 이미지 검색 중..."):
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                norm_image = normalize_brightness(image_bgr)
                input_lab_palette = extract_lab_palette(norm_image, st.session_state.mask)
                
                # 분석 결과를 session_state에 저장하고, 앱을 재실행하여 결과 화면으로 전환
                st.session_state.results = {
                    "input_palette_img": create_palette_image(input_lab_palette),
                    "recommendations": get_top_similar_images(input_lab_palette, load_all_json_data(JSON_PATH))
                }
                st.rerun()
        else:
            st.warning("먼저 위 이미지에서 영역을 클릭하여 선택해주세요!")

def display_results():
    """분석 및 추천 결과를 표시하는 함수"""
    st.write("---")
    st.header("✨ 분석 결과")

    results = st.session_state.results
    st.subheader("추출된 색상 팔레트")
    st.image(results["input_palette_img"], channels="BGR")
    
    st.subheader("유사 이미지 추천")
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            if i < len(results["recommendations"]):
                dist, entry = results["recommendations"][i]
                raw_name = entry['raw']
                folder = 'data/raw_apt' if raw_name.startswith("A") else 'data/raw_child'
                img_path = os.path.join(folder, raw_name)
                
                if os.path.exists(img_path):
                    rec_image = Image.open(img_path)
                    st.image(rec_image, caption=f"유사도 점수: {dist:.2f}")
                else:
                    st.warning(f"이미지 없음: {img_path}")
    
    if st.button("다시 분석하기"):
        # 상태를 초기화하고 앱을 재실행하여 처음 화면으로 돌아감
        st.session_state.results = None
        st.session_state.mask = None
        st.rerun()

# ===================================================================
# 메인 실행 함수
# ===================================================================

def main():
    """메인 애플리케이션 실행 로직"""
    st.set_page_config(layout="wide")
    st.title("🎨 단일 이미지 컬러 분석 시스템 (리팩토링 버전)")

    # 모델/데이터 로딩은 한번만 실행됨
    predictor = load_sam_predictor(MODEL_PATH)
    
    # 상태 초기화
    if "results" not in st.session_state:
        st.session_state.results = None

    if st.session_state.results:
        # 결과가 있으면 결과 표시 UI 호출
        display_results()
    else:
        # 결과가 없으면 영역 선택 UI 호출
        st.header(f"분석 대상 이미지: `{IMAGE_PATH}`")
        if not os.path.exists(IMAGE_PATH):
            st.error(f"'{IMAGE_PATH}' 파일을 찾을 수 없습니다!")
            return
        
        image = Image.open(IMAGE_PATH).convert("RGB")
        image_np = np.array(image)
        predictor.set_image(image_np)
        display_area_selection(predictor, image, image_np)

if __name__ == "__main__":
    main()