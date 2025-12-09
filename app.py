import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import re
import os
import json
import pandas as pd
import datetime
from io import BytesIO
import cv2

# =====================
#  LEVENSHTEIN FALLBACK
# =====================
try:
    import Levenshtein as _lev
    def levenshtein_ratio(a, b):
        return _lev.ratio(a, b)
except Exception:
    from difflib import SequenceMatcher
    def levenshtein_ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()


# =====================
#   CONSTANTS
# =====================
ANSWER_DIR = "answer_keys"
RESULTS_DIR = "results"

os.makedirs(ANSWER_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)


# =====================
#  OCR FUNCTION
# =====================
def ocr_image_pil(image: Image.Image):
    """Run EasyOCR and return list of cleaned text lines."""
    img_arr = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    try:
        result = reader.readtext(gray, detail=0)
    except Exception as e:
        st.error(f"OCR error: {e}")
        result = []

    cleaned = [r.strip() for r in result if isinstance(r, str) and r.strip()]
    return cleaned


# ================
# PARSING HELPERS
# ================
def _to_int_safe(x):
    try:
        return int(x)
    except:
        return None


def parse_mcq_from_lines(lines):
    answers = {}
    for line in lines:
        m = re.match(r'^\s*(\d{1,3})[\.\)\-\: ]+\s*([A-Da-d])\b', line)
        if m:
            q = _to_int_safe(m.group(1))
            if q:
                answers[q] = m.group(2).upper()
            continue

        m2 = re.match(r'^\s*(\d{1,3})\s+([A-Da-d])\b', line)
        if m2:
            q = _to_int_safe(m2.group(1))
            if q:
                answers[q] = m2.group(2).upper()
            continue

        m3 = re.findall(r'(\d{1,3})\s*[:\-\)\.]\s*([A-Da-d])', line)
        for q, a in m3:
            qn = _to_int_safe(q)
            if qn:
                answers[qn] = a.upper()

    return answers


def parse_tf_from_lines(lines):
    answers = {}
    for line in lines:
        m = re.findall(r'(\d{1,3})\s*[:\-\)\.]\s*(True|False|T|F)', line, flags=re.I)
        for q, a in m:
            qn = _to_int_safe(q)
            if not qn:
                continue
            answers[qn] = "True" if a.strip()[0].upper() == "T" else "False"
    return answers


def parse_fill_from_lines(lines):
    answers = {}
    for line in lines:
        m = re.match(r'^\s*(\d{1,3})[\.\)\-\: ]+\s*(.+)', line)
        if m:
            q = _to_int_safe(m.group(1))
            if q:
                answers[q] = m.group(2).strip()
    return answers


def parse_match_from_lines(lines):
    answers = {}
    for line in lines:
        m = re.findall(r'(\d{1,3})\s*[-\:\)]\s*([A-Za-z])', line)
        for q, a in m:
            qn = _to_int_safe(q)
            if qn:
                answers[qn] = a.upper()

        m2 = re.findall(r'(\d{1,3})\s+([A-Za-z])\b', line)
        for q, a in m2:
            qn = _to_int_safe(q)
            if qn:
                answers[qn] = a.upper()
    return answers


# =====================
#   GRADING FUNCTIONS
# =====================
def grade_mcq(student, key):
    ak = {int(k): str(v).strip().upper() for k, v in key.items()}
    sa = {int(k): str(v).strip().upper() for k, v in student.items()}

    total = correct = 0
    details = []

    for q, ans in ak.items():
        total += 1
        stu = sa.get(q)

        if not stu:
            details.append(f"Q{q}: No answer (expected {ans})")
        elif stu == ans:
            correct += 1
            details.append(f"Q{q}: ✔ {stu}")
        else:
            details.append(f"Q{q}: ✘ {stu} (expected {ans})")

    return correct, total, details


def grade_tf(student, key):
    ak = {int(k): ("True" if str(v).lower().startswith("t") else "False") for k, v in key.items()}
    sa = {}

    for k, v in student.items():
        try:
            k2 = int(k)
        except:
            continue

        sa[k2] = "True" if str(v).lower().startswith("t") else "False"

    return grade_mcq(sa, ak)


def grade_fill(student, key, threshold=0.75):
    ak = {int(k): str(v).strip() for k, v in key.items()}
    sa = {int(k): str(v).strip() for k, v in student.items()}

    total = correct = 0
    details = []

    for q, ans in ak.items():
        total += 1
        stu = sa.get(q)

        if not stu:
            details.append(f"Q{q}: No answer (expected '{ans}')")
            continue

        stu_norm = re.sub(r'[^A-Za-z0-9 ]', '', stu.lower()).strip()
        ans_norm = re.sub(r'[^A-Za-z0-9 ]', '', ans.lower()).strip()

        ratio = levenshtein_ratio(stu_norm, ans_norm)

        if ratio >= threshold:
            correct += 1
            details.append(f"Q{q}: ✔ ({stu} ≈ {ans})")
        else:
            details.append(f"Q{q}: ✘ ({stu}) expected ({ans}) | sim={ratio:.2f}")

    return correct, total, details


def grade_match(student, key):
    ak = {int(k): str(v).strip().upper() for k, v in key.items()}
    sa = {int(k): str(v).strip().upper() for k, v in student.items()}

    total = correct = 0
    details = []

    for q, ans in ak.items():
        total += 1
        stu = sa.get(q)

        if stu == ans:
            correct += 1
            details.append(f"{q}-{stu}: ✔")
        else:
            details.append(f"{q}-{stu if stu else 'None'}: ✘ expected {ans}")

    return correct, total, details


# ======================
#     STREAMLIT UI
# ======================
st.title("📘 Automatic Exam Grader (OCR + Answer Keys)")

tab1, tab2 = st.tabs(["📤 Upload Answer Key", "📝 Grade Student Answers"])


# ======================
#   TAB 1 — ANSWER KEY
# ======================
with tab1:
    st.header("Tạo Answer Key")
    ak_type = st.selectbox("Loại Answer Key", ["MCQ", "True/False", "Fill-in", "Matching"])

    file = st.file_uploader("Upload ảnh Answer Key", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Image preview", width=400)

        lines = ocr_image_pil(img)
        st.write("OCR result:")
        st.text("\n".join(lines))

        if st.button("Parse Answer Key"):
            if ak_type == "MCQ":
                ans = parse_mcq_from_lines(lines)
            elif ak_type == "True/False":
                ans = parse_tf_from_lines(lines)
            elif ak_type == "Fill-in":
                ans = parse_fill_from_lines(lines)
            else:
                ans = parse_match_from_lines(lines)

            st.success("Parsed successfully!")
            st.json(ans)

            key_name = st.text_input("Đặt tên Answer Key (VD: english_test_1)")

            if key_name and st.button("Lưu Answer Key"):
                with open(f"{ANSWER_DIR}/{key_name}.json", "w", encoding="utf-8") as f:
                    json.dump({"type": ak_type, "key": ans}, f, indent=2)
                st.success("Saved!")


# ==============================
#   TAB 2 — GRADING STUDENTS
# ==============================
with tab2:
    st.header("Chấm bài học sinh")

    # Load answer key list
    keys = [f for f in os.listdir(ANSWER_DIR) if f.endswith(".json")]

    if not keys:
        st.warning("Chưa có answer key!")
    else:
        chosen = st.selectbox("Chọn answer key", keys)

        file = st.file_uploader("Upload bài làm học sinh", type=["png", "jpg", "jpeg"])

        if file:
            img = Image.open(file)
            st.image(img, caption="Student Image", width=400)

            lines = ocr_image_pil(img)
            st.write("OCR result:")
            st.text("\n".join(lines))

            if st.button("Parse Student Answers"):
                with open(f"{ANSWER_DIR}/{chosen}", "r", encoding="utf-8") as f:
                    data = json.load(f)

                ak_type = data["type"]
                key = data["key"]

                if ak_type == "MCQ":
                    stu = parse_mcq_from_lines(lines)
                    c, t, det = grade_mcq(stu, key)

                elif ak_type == "True/False":
                    stu = parse_tf_from_lines(lines)
                    c, t, det = grade_tf(stu, key)

                elif ak_type == "Fill-in":
                    stu = parse_fill_from_lines(lines)
                    c, t, det = grade_fill(stu, key)

                else:
                    stu = parse_match_from_lines(lines)
                    c, t, det = grade_match(stu, key)

                st.subheader("Kết quả:")
                st.write(f"**Score: {c} / {t}**")
                st.text("\n".join(det))
