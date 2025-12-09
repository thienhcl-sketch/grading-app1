# app.py
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
import tempfile
import Levenshtein

st.set_page_config(page_title="Auto Grading System", layout="wide")
st.title("📘 Auto Grading — Multi-format (MCQ, Fill, Match, TF, Writing, Essay, Vocab, Listening)")

# -----------------------------
# Utils / Init
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True nếu có GPU
ANSWER_DIR = "answer_keys"
RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def save_results_excel(df, path):
    df.to_excel(path, index=False)

def ocr_image_pil(image: Image.Image):
    """Run EasyOCR and return list of lines (lowercased)."""
    img_arr = np.array(image.convert('RGB'))
    # Optional preproc: grayscale & adaptive threshold
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    # simple denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # run OCR
    result = reader.readtext(gray, detail=0)
    cleaned = [r.strip() for r in result if r.strip()]
    return cleaned

# -----------------------------
# Parsers: try to extract answers from OCR text
# -----------------------------
def parse_mcq_from_lines(lines):
    """
    Try to find patterns like:
    1 A
    1. A
    1) A
    1:A
    1-A
    or lines that are just "A B C D"
    Returns dict: {question_number: answer_letter}
    """
    answers = {}
    for line in lines:
        # common patterns
        m = re.match(r'^\s*(\d{1,3})[\.\)\-\: ]+\s*([A-Da-d])\b', line)
        if m:
            q = int(m.group(1))
            a = m.group(2).upper()
            answers[q] = a
            continue
        # pattern "1 A" with space
        m2 = re.match(r'^\s*(\d{1,3})\s+([A-Da-d])\b', line)
        if m2:
            answers[int(m2.group(1))] = m2.group(2).upper()
            continue
        # pattern "1: A" anywhere
        m3 = re.findall(r'(\d{1,3})\s*[:\-\)\.]\s*([A-Da-d])', line)
        for tup in m3:
            answers[int(tup[0])] = tup[1].upper()
    return answers

def parse_tf_from_lines(lines):
    """
    Look for patterns like: 1 True / False or 1 T / F
    """
    answers = {}
    for line in lines:
        m = re.findall(r'(\d{1,3})\s*[:\-\)\.]\s*(True|False|T|F|TRUE|FALSE)', line)
        for q,a in m:
            val = 'True' if a[0].upper()=='T' else 'False' if a[0].upper()=='F' else a.title()
            answers[int(q)] = val
    return answers

def parse_fill_from_lines(lines):
    """
    For fill-in blanks, often OCR returns "1. elephant" etc.
    Return dict {q: text}
    """
    answers = {}
    for line in lines:
        m = re.match(r'^\s*(\d{1,3})[\.\)\-\: ]+\s*(.+)', line)
        if m:
            q = int(m.group(1))
            ans = m.group(2).strip()
            answers[q] = ans
    return answers

def parse_match_from_lines(lines):
    """
    Look for patterns like "1-C", "1 - C", "1 C"
    Return dict {left_index: right_label}
    """
    answers = {}
    for line in lines:
        m = re.findall(r'(\d{1,3})\s*[-\:\)]\s*([A-Za-z])', line)
        for q,a in m:
            answers[int(q)] = a.upper()
        m2 = re.findall(r'(\d{1,3})\s+([A-Za-z])\b', line)
        for q,a in m2:
            # only accept single-letter right side
            if len(a)==1 and a.isalpha():
                answers[int(q)] = a.upper()
    return answers

# -----------------------------
# Graders for each type
# -----------------------------
def grade_mcq(student_answers: dict, answer_key: dict):
    total = 0
    correct = 0
    details = []
    for q, ans in answer_key.items():
        total += 1
        sa = student_answers.get(int(q))
        if sa is None:
            details.append(f"Q{q}: No answer (expected {ans})")
        elif sa.upper() == ans.upper():
            correct += 1
            details.append(f"Q{q}: ✔ {sa}")
        else:
            details.append(f"Q{q}: ✘ {sa} (expected {ans})")
    return correct, total, details

def grade_tf(student_answers: dict, answer_key: dict):
    return grade_mcq(student_answers, answer_key)  # same logic but keys values True/False

def grade_fill(student_answers: dict, answer_key: dict, lev_threshold=0.75):
    total = correct = 0
    details = []
    for q, ans in answer_key.items():
        total += 1
        sa = student_answers.get(int(q))
        if not sa:
            details.append(f"Q{q}: No answer (expected '{ans}')")
            continue
        # normalize
        sa_n = re.sub(r'[^A-Za-z0-9 ]','', sa.lower()).strip()
        ans_n = re.sub(r'[^A-Za-z0-9 ]','', ans.lower()).strip()
        if not ans_n:
            details.append(f"Q{q}: No key")
            continue
        # use Levenshtein ratio
        ratio = Levenshtein.ratio(sa_n, ans_n)
        if ratio >= lev_threshold:
            correct += 1
            details.append(f"Q{q}: ✔ ({sa} ≈ {ans})")
        else:
            details.append(f"Q{q}: ✘ ({sa}) expected ({ans}), similarity {ratio:.2f}")
    return correct, total, details

def grade_match(student_answers: dict, answer_key: dict):
    total = correct = 0
    details = []
    for left, right in answer_key.items():
        total += 1
        sa = student_answers.get(int(left))
        if sa == right:
            correct += 1
            details.append(f"{left}-{sa}: ✔")
        else:
            details.append(f"{left}-{sa if sa else 'None'}: ✘ expected {right}")
    return correct, total, details

def analyze_writing_text(student_text: str, rubric: dict):
    """
    Simple rubric scoring:
    rubric = {"length": {"min": 20}, "keywords": ["because", "but"], "max_score": 10}
    We'll score by presence of keywords, length, basic punctuation and capitalization.
    Return score_out_of_max, feedback_list
    """
    max_score = rubric.get("max_score",10)
    score = max_score
    feedback = []

    s = student_text.strip()
    if len(s.split()) < rubric.get("length", {}).get("min", 5):
        feedback.append("Too short.")
        score -= 2

    # keywords
    for kw in rubric.get("keywords", []):
        if kw.lower() not in s.lower():
            feedback.append(f"Missing keyword: {kw}")
            score -= 1

    # simple spelling heuristics: count tokens with many non-alpha chars
    tokens = re.findall(r"[A-Za-z']+", s)
    miss = 0
    for t in tokens:
        if len(t)>1 and not t.isalpha():
            miss += 1
    if miss>0:
        feedback.append(f"Possible errors detected: {miss} tokens.")
        score -= min(2, miss)

    # punctuation & capitalization
    if not s or not s[0].isupper():
        feedback.append("Start with a capital letter.")
        score -= 1
    if s and s[-1] not in ".!?":
        feedback.append("Missing ending punctuation.")
        score -= 1

    score = max(0, score)
    return score, max_score, feedback

def grade_vocab(student_answers, answer_key):
    return grade_fill(student_answers, answer_key)

def grade_listening(uploaded_audio_answer, answer_key):
    # This app assumes teacher provides a text-based answer key for listening, e.g. "1: apple"
    # If student submit answers via typed text area, we compare similarly to fill
    # Here we just stub: compare dicts
    return grade_fill(uploaded_audio_answer, answer_key)

# -----------------------------
# Streamlit UI
# -----------------------------
st.sidebar.header("Settings / Files")
st.sidebar.write("Drop your answer_key JSON files into `answer_keys/` folder.")
st.sidebar.write("Expected keys: mcq_part*.json, fill_part*.json, match_part*.json, tf_part*.json, writing_rubric.json, vocab_part*.json, listening_part*.json")

student_name = st.text_input("Student name")
student_id = st.text_input("Student ID (optional)")

uploaded = st.file_uploader("Upload student answer sheet image", type=["png","jpg","jpeg"])

# load available answer files
available_keys = {}
if os.path.exists(ANSWER_DIR):
    for fn in os.listdir(ANSWER_DIR):
        if fn.endswith(".json"):
            try:
                available_keys[fn] = load_json(os.path.join(ANSWER_DIR, fn))
            except Exception as e:
                st.sidebar.error(f"Cannot load {fn}: {e}")
else:
    st.sidebar.info("No answer_keys folder found. Create one with required JSON files.")

st.markdown("---")
st.header("Auto-grading panel")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=False, width=600)

    with st.spinner("Running OCR..."):
        lines = ocr_image_pil(image)
    st.write("OCR output (lines):")
    st.write(lines)

    # Parse answers automatically
    mcq_parsed = parse_mcq_from_lines(lines)
    tf_parsed = parse_tf_from_lines(lines)
    fill_parsed = parse_fill_from_lines(lines)
    match_parsed = parse_match_from_lines(lines)

    st.subheader("Detected answers (automated parse)")
    st.write("MCQ parsed:", mcq_parsed)
    st.write("TF parsed:", tf_parsed)
    st.write("Fill parsed:", fill_parsed)
    st.write("Match parsed:", match_parsed)

    total_score = 0
    total_max = 0
    result_details = {}

    # 1) MCQ grading
    for key_fname, key_content in available_keys.items():
        if key_fname.startswith("mcq"):
            st.subheader(f"Grading {key_fname}")
            correct, total, details = grade_mcq(mcq_parsed, key_content)
            st.write(f"Score: {correct}/{total}")
            for d in details:
                st.write("-", d)
            total_score += correct
            total_max += total
            result_details[key_fname] = {"score": correct, "max": total, "details": details}

    # 2) TF grading
    for key_fname, key_content in available_keys.items():
        if key_fname.startswith("tf"):
            st.subheader(f"Grading {key_fname}")
            correct, total, details = grade_tf(tf_parsed, key_content)
            st.write(f"Score: {correct}/{total}")
            for d in details:
                st.write("-", d)
            total_score += correct
            total_max += total
            result_details[key_fname] = {"score": correct, "max": total, "details": details}

    # 3) Fill-in grading
    for key_fname, key_content in available_keys.items():
        if key_fname.startswith("fill"):
            st.subheader(f"Grading {key_fname}")
            correct, total, details = grade_fill(fill_parsed, key_content)
            st.write(f"Score: {correct}/{total}")
            for d in details:
                st.write("-", d)
            total_score += correct
            total_max += total
            result_details[key_fname] = {"score": correct, "max": total, "details": details}

    # 4) Match grading
    for key_fname, key_content in available_keys.items():
        if key_fname.startswith("match"):
            st.subheader(f"Grading {key_fname}")
            correct, total, details = grade_match(match_parsed, key_content)
            st.write(f"Score: {correct}/{total}")
            for d in details:
                st.write("-", d)
            total_score += correct
            total_max += total
            result_details[key_fname] = {"score": correct, "max": total, "details": details}

    # 5) Vocab (treated as fill)
    for key_fname, key_content in available_keys.items():
        if key_fname.startswith("vocab"):
            st.subheader(f"Grading {key_fname}")
            correct, total, details = grade_vocab(fill_parsed, key_content)
            st.write(f"Score: {correct}/{total}")
            for d in details:
                st.write("-", d)
            total_score += correct
            total_max += total
            result_details[key_fname] = {"score": correct, "max": total, "details": details}

    # 6) Listening (assume teacher provides a typed area of student's answers for listening)
    if "listening_part1.json" in available_keys:
        st.subheader("Listening answers (manual input from student teacher or typed by human)")
        listen_input = st.text_area("Paste student's listening answers (one per line like '1 apple'):")
        listen_parsed = {}
        for line in listen_input.splitlines():
            m = re.match(r'^\s*(\d{1,3})[\.\)\:\-\s]+\s*(.+)$', line.strip())
            if m:
                listen_parsed[int(m.group(1))] = m.group(2).strip()
        correct, total, details = grade_listening(listen_parsed, available_keys["listening_part1.json"])
        st.write(f"Listening: {correct}/{total}")
        for d in details:
            st.write("-", d)
        total_score += correct
        total_max += total
        result_details["listening_part1.json"] = {"score": correct, "max": total, "details": details}

    # 7) Writing & Essay: teacher may paste recognized writing or type it manually
    if "writing_rubric.json" in available_keys:
        st.subheader("Writing / Essay grading")
        st.write("If student's writing is printed/handwritten, paste OCR text for the writing or edit it below.")
        default_write = "\n".join([l for l in lines])  # suggest OCR output
        student_writing = st.text_area("Student writing (edit if OCR wrong):", value=default_write, height=200)
        rubric = available_keys["writing_rubric.json"]
        w_score, w_max, w_feedback = analyze_writing_text(student_writing, rubric)
        st.write(f"Writing: {w_score}/{w_max}")
        for f in w_feedback:
            st.write("-", f)
        total_score += w_score
        total_max += w_max
        result_details["writing"] = {"score": w_score, "max": w_max, "feedback": w_feedback}

    st.markdown("---")
    st.success(f"TOTAL SCORE: {total_score} / {total_max}")

    # Save result button
    if st.button("Save result to Excel"):
        file_path = os.path.join(RESULTS_DIR, "student_results.xlsx")
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
        else:
            df = pd.DataFrame(columns=["StudentID","StudentName","Date","TotalScore","MaxScore","DetailsFile"])

        date = datetime.date.today().isoformat()
        # Save details JSON for this student's result (one file per save)
        details_folder = os.path.join(RESULTS_DIR, "details")
        if not os.path.exists(details_folder):
            os.makedirs(details_folder)
        details_fname = f"{student_name or 'unknown'}_{student_id or 'id'}_{date}.json"
        with open(os.path.join(details_folder, details_fname), "w", encoding="utf8") as f:
            json.dump(result_details, f, ensure_ascii=False, indent=2)

        new_row = {
            "StudentID": student_id,
            "StudentName": student_name,
            "Date": date,
            "TotalScore": total_score,
            "MaxScore": total_max,
            "DetailsFile": details_fname
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(file_path, index=False)
        st.success(f"Saved results and details to {file_path} and {details_fname}")

    # Option to download excel
    if os.path.exists(os.path.join(RESULTS_DIR, "student_results.xlsx")):
        with open(os.path.join(RESULTS_DIR, "student_results.xlsx"), "rb") as f:
            st.download_button("Download all results (Excel)", data=f, file_name="student_results.xlsx")

else:
    st.info("Upload an image to begin.")
