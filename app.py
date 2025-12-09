import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import easyocr
import json
import datetime

st.set_page_config(page_title="EduScan – Auto Grading", layout="wide")

# -------------------------
# LOAD OCR
# -------------------------
reader = easyocr.Reader(["en"])

# -------------------------
# FUNCTIONS
# -------------------------
def load_answer_key(file):
    with open(file, "r", encoding="utf8") as f:
        return json.load(f)

def extract_text(image):
    result = reader.readtext(np.array(image), detail=0)
    return [t.lower().strip() for t in result]

def grade_mcq(student_answers, answer_key):
    score = 0
    details = []

    for q, correct in answer_key.items():
        student_ans = None
        for t in student_answers:
            if f"{q}".lower() in t or t in ["a", "b", "c", "d"]:
                student_ans = t.upper()
                break

        if student_ans == correct.upper():
            score += 1
            details.append(f"Q{q}: ✔ Correct ({student_ans})")
        else:
            details.append(f"Q{q}: ✘ Incorrect — Your answer: {student_ans}, Expected: {correct}")

    return score, details


def analyze_writing(text):
    """Simple writing feedback"""
    corrections = []
    score = 10

    common_errors = {
        "i": "I",
        "dont": "don't",
        "writting": "writing",
        "becaus": "because",
        "teh": "the"
    }

    for wrong, right in common_errors.items():
        if wrong in text.lower():
            corrections.append(f"Correction: '{wrong}' → '{right}'")
            score -= 1

    return score, corrections


def save_results(name, part1, part2, writing_score, total):
    file_path = "results/student_results.xlsx"

    if not os.path.exists("results"):
        os.makedirs("results")

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=[
            "Name", "Part 1", "Part 2", "Writing", "Total", "Date"
        ])

    new_row = {
        "Name": name,
        "Part 1": part1,
        "Part 2": part2,
        "Writing": writing_score,
        "Total": total,
        "Date": datetime.date.today()
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(file_path, index=False)


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📘 EduScan – Automatic Test Grading")

student_name = st.text_input("Student Name")

uploaded = st.file_uploader("Upload Student Answer Sheet", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Answer Sheet", width=500)

    st.subheader("🔍 OCR – Extracting Text...")
    text = extract_text(img)
    st.write(text)

    st.subheader("📌 Part 1 – MCQ")
    part1_key = load_answer_key("answer_keys/part1.json")
    part1_score, part1_details = grade_mcq(text, part1_key)
    st.write(part1_details)

    st.subheader("📌 Part 2 – MCQ")
    part2_key = load_answer_key("answer_keys/part2.json")
    part2_score, part2_details = grade_mcq(text, part2_key)
    st.write(part2_details)

    st.subheader("📝 Part 3 – Writing")
    writing_input = st.text_area("Enter student's writing:")
    writing_score, writing_feedback = analyze_writing(writing_input)

    st.write("Feedback:")
    for f in writing_feedback:
        st.write("-", f)

    # Total score
    total_score = part1_score + part2_score + writing_score
    st.success(f"🎯 Total Score: {total_score}")

    if st.button("💾 Save Result"):
        save_results(student_name, part1_score, part2_score, writing_score, total_score)
        st.success("Saved to Excel successfully!")
