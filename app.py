import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO


model = YOLO("best.pt")  

st.title("تشخیص آفات گیاهی با YOLOv11")

uploaded_file = st.file_uploader("یک تصویر آپلود کنید", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="تصویر ورودی", use_column_width=True)

    if st.button("تشخیص آفات"):
        results = model.predict(np.array(image), imgsz=640, conf=0.25)
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="تصویر با تشخیص آفات", use_column_width=True)

        names = model.names
        detected_classes = [names[int(box.cls)] for box in results[0].boxes]
        if detected_classes:
            st.success("آفات شناسایی‌شده:")
            st.write(", ".join(set(detected_classes)))
        else:
            st.warning("هیچ آفتی شناسایی نشد.")
