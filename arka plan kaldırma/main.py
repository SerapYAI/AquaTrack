# type 'streamlit run main.py' on the terminal to run the code
import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # GrabCut ile kaba ön plan maskesini elde etme
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (1, 1, w - 2, h - 2)
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(
        "uint8"
    )

    # Morfolojik kapanışla deliği kapatma
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Bağlı bileşenleri bulma, sadece geniş yatay bileşenleri seçme
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask2, connectivity=8)
    clean = np.zeros_like(mask2)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 500:
            continue
        clean[labels == i] = 1

    if np.sum(clean) == 0:
        return None

    # Kenarları yumuşatıp alfa kanalı oluşturma
    clean = cv2.GaussianBlur(clean.astype("float32"), (7, 7), 0)
    alpha = (clean > 0.5).astype("uint8") * 255

    # RGBA’ye dönüştürme ve return etme
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, alpha])

    return rgba


# Streamlit Arayüzü
st.title("Arkaplan Kaldırma Uygulaması")

uploaded_file = st.file_uploader("Resim Yükleyim", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Orijinal Resim", use_container_width=True)
    result = process_image(uploaded_file)

    # Sonucu Yazdır
    st.image(
        result,
        caption="İşlenmiş Resim",
        channels="BGRA",
        use_container_width=True,
    )

    # Sonucu İndir
    result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        "Sonucu İndir",
        data=byte_im,
        file_name="islenmis_resim.png",
        mime="image/png",
    )
