import streamlit as st
from PIL import Image
import numpy as np
import easyocr

st.set_page_config(page_title="Automatic Grading App", layout="wide")
st.title("📘 Automatic Grading App")

st.write("Upload an answer sheet image to scan:")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ---- ANSWER KEY (bạn thay theo đề của bạn) ----
answer_key = {
    1: "shark",
    2: "stairs",
    3: "duck",
    4: "backpack",
    5: "clay"
}

reader = easyocr.Reader(["en"])

def extract_text(image):
    result = reader.readtext(np.array(image), detail=0)
    return result

def grade_student(text_list, answer_key):
    score = 0
    results = []

    for qnum, ans in answer_key.items():
        found = any(ans.lower() in t.lower() for t in text_list)
        if found:
            score += 1
            results.append(f"Q{qnum}: ✔ Correct")
        else:
            results.append(f"Q{qnum}: ✘ Incorrect (expected: {ans})")

    return score, results

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Answer Sheet", width=400)

    with st.spinner("🔍 Scanning image..."):
        text = extract_text(img)

    st.write("### OCR Extracted Text:")
    st.write(text)

    score, details = grade_student(text, answer_key)

    st.write("### 📝 Grading Result:")
    st.write("Score:", score, "/", len(answer_key))

    for d in details:
        st.write("-", d)
