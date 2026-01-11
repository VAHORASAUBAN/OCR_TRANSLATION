# import streamlit as st
# import cv2, json, pytesseract, unicodedata, torch
# import numpy as np
# from pdf2image import convert_from_path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from tempfile import NamedTemporaryFile

# st.set_page_config(page_title="Multilingual OCR + Translation", layout="wide")

# # ---------------- CONFIG ----------------
# LANG_SCRIPT_MAP = {
#     "Devanagari":"hin_Deva","Tamil":"tam_Taml","Telugu":"tel_Telu",
#     "Malayalam":"mal_Mlym","Kannada":"kan_Knda","Gujarati":"guj_Gujr",
#     "Bengali":"ben_Beng","Gurmukhi":"pan_Guru","Arabic":"urd_Arab","Latin":"eng_Latn"
# }

# MODEL="facebook/nllb-200-3.3B"
# DEVICE="cuda" if torch.cuda.is_available() else "cpu"
# MAX_TOKENS = 1024

# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE)
#     model.eval()
#     return tokenizer, model

# tokenizer, model = load_model()
# cache = {}

# # ---------------- OCR HELPERS ----------------
# def preprocess(img):
#     img = np.array(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray = cv2.bilateralFilter(gray,11,75,75)
#     clahe = cv2.createCLAHE(2.0,(8,8))
#     return clahe.apply(gray)

# def get_script(c):
#     o = ord(c)
#     if 0x0900 <= o <= 0x097F: return "Devanagari"
#     if 0x0B80 <= o <= 0x0BFF: return "Tamil"
#     if 0x0C00 <= o <= 0x0C7F: return "Telugu"
#     if 0x0D00 <= o <= 0x0D7F: return "Malayalam"
#     if 0x0C80 <= o <= 0x0CFF: return "Kannada"
#     if 0x0A80 <= o <= 0x0AFF: return "Gujarati"
#     if 0x0980 <= o <= 0x09FF: return "Bengali"
#     if 0x0A00 <= o <= 0x0A7F: return "Gurmukhi"
#     if 0x0600 <= o <= 0x06FF: return "Arabic"
#     if 0x0041 <= o <= 0x007A: return "Latin"
#     return None

# def split_lines(text):
#     return [x.strip() for x in text.split("\n") if x.strip()]

# # ---------------- TRANSLATION ----------------
# def tokenize_chunks(text):
#     ids = tokenizer(text).input_ids
#     chunks, cur = [], []
#     for i in ids:
#         cur.append(i)
#         if len(cur) >= MAX_TOKENS:
#             chunks.append(cur)
#             cur = []
#     if cur:
#         chunks.append(cur)
#     return chunks

# def translate(txt, lang):
#     key = (txt, lang)
#     if key in cache:
#         return cache[key]

#     tokenizer.src_lang = lang
#     forced = tokenizer.convert_tokens_to_ids("eng_Latn")

#     final = ""
#     for c in tokenize_chunks(txt):
#         inp = torch.tensor([c]).to(DEVICE)
#         with torch.no_grad():
#             out = model.generate(inp, forced_bos_token_id=forced)
#         final += tokenizer.decode(out[0], skip_special_tokens=True) + " "

#     cache[key] = final.strip()
#     return cache[key]

# # ---------------- STREAMLIT UI ----------------
# st.title("📄 Multilingual OCR + AI Translation Engine")
# st.markdown("Upload a scanned PDF and get **script-aware OCR + English translation**")

# uploaded = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded:
#     with NamedTemporaryFile(delete=False, suffix=".pdf") as f:
#         f.write(uploaded.read())
#         pdf_path = f.name

#     st.info("Running OCR… this is heavy stuff, give it a sec 💪")

#     pages = convert_from_path(pdf_path, dpi=600)
#     raw_lines = []

#     for i, p in enumerate(pages):
#         img = preprocess(p)
#         text = pytesseract.image_to_string(
#             img,
#             lang="eng+tam+hin+urd+tel+kan+mal+ben+guj+pan",
#             config="--psm 6 --oem 3"
#         )
#         text = unicodedata.normalize("NFKC", text)
#         for line in split_lines(text):
#             raw_lines.append((i+1, line))

#     # --- detect active scripts ---
#     char_count = {}
#     total = 0
#     for _, line in raw_lines:
#         for c in line:
#             s = get_script(c)
#             if s:
#                 char_count[s] = char_count.get(s,0)+1
#                 total += 1

#     script_ratio = {s:char_count[s]/total for s in char_count}
#     valid_scripts = {s for s,r in script_ratio.items() if r >= 0.10}

#     st.subheader("🧠 Detected Scripts")
#     for s in valid_scripts:
#         st.write(f"**{s}** → {script_ratio[s]*100:.2f}%")

#     # --- filter + translate ---
#     results = []
#     for page, line in raw_lines:
#         counts = {}
#         for c in line:
#             s = get_script(c)
#             if s:
#                 counts[s] = counts.get(s,0)+1
#         if not counts:
#             continue

#         dominant = max(counts, key=counts.get)
#         if dominant not in valid_scripts:
#             continue

#         src = LANG_SCRIPT_MAP.get(dominant,"eng_Latn")
#         if src == "eng_Latn":
#             eng = line
#         else:
#             eng = translate(line, src)

#         results.append((page, dominant, line, eng))

#     # --- Display ---
#     st.subheader("📜 Page-wise OCR + Translation")
#     for page, script, ocr, eng in results:
#         st.markdown(f"### Page {page} — {script}")
#         st.code(ocr)
#         st.success(eng)


import streamlit as st
import cv2, json, pytesseract, unicodedata, torch, fitz
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tempfile import NamedTemporaryFile

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

st.set_page_config(page_title="Multilingual OCR + Translation", layout="wide")

# ---------------- CONFIG ----------------
LANG_SCRIPT_MAP = {
    "Devanagari":"hin_Deva","Tamil":"tam_Taml","Telugu":"tel_Telu",
    "Malayalam":"mal_Mlym","Kannada":"kan_Knda","Gujarati":"guj_Gujr",
    "Bengali":"ben_Beng","Gurmukhi":"pan_Guru","Arabic":"urd_Arab","Latin":"eng_Latn"
}

MODEL="facebook/nllb-200-1.3B"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 1024

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
cache = {}

# ---------------- PDF RENDERER (PyMuPDF) ----------------
def pdf_to_images(pdf_path, dpi=600):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images

# ---------------- OCR HELPERS ----------------
def preprocess(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray,11,75,75)
    clahe = cv2.createCLAHE(2.0,(8,8))
    return clahe.apply(gray)

def get_script(c):
    o = ord(c)
    if 0x0900 <= o <= 0x097F: return "Devanagari"
    if 0x0B80 <= o <= 0x0BFF: return "Tamil"
    if 0x0C00 <= o <= 0x0C7F: return "Telugu"
    if 0x0D00 <= o <= 0x0D7F: return "Malayalam"
    if 0x0C80 <= o <= 0x0CFF: return "Kannada"
    if 0x0A80 <= o <= 0x0AFF: return "Gujarati"
    if 0x0980 <= o <= 0x09FF: return "Bengali"
    if 0x0A00 <= o <= 0x0A7F: return "Gurmukhi"
    if 0x0600 <= o <= 0x06FF: return "Arabic"
    if 0x0041 <= o <= 0x007A: return "Latin"
    return None

def split_lines(text):
    return [x.strip() for x in text.split("\n") if x.strip()]

# ---------------- TRANSLATION ----------------
def tokenize_chunks(text):
    ids = tokenizer(text).input_ids
    chunks, cur = [], []
    for i in ids:
        cur.append(i)
        if len(cur) >= MAX_TOKENS:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks

def translate(txt, lang):
    key = (txt, lang)
    if key in cache:
        return cache[key]

    tokenizer.src_lang = lang
    forced = tokenizer.convert_tokens_to_ids("eng_Latn")

    final = ""
    for c in tokenize_chunks(txt):
        inp = torch.tensor([c]).to(DEVICE)
        with torch.no_grad():
            out = model.generate(inp, forced_bos_token_id=forced)
        final += tokenizer.decode(out[0], skip_special_tokens=True) + " "

    cache[key] = final.strip()
    return cache[key]

# ---------------- STREAMLIT UI ----------------
st.title("📄 Multilingual OCR + AI Translation Engine")
st.markdown("Upload a scanned PDF and get **script-aware OCR + English translation**")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        pdf_path = f.name

    st.info("Running OCR… this is heavy stuff, give it a sec 💪")

    pages = pdf_to_images(pdf_path, dpi=600)
    raw_lines = []

    for i, p in enumerate(pages):
        img = preprocess(p)
        text = pytesseract.image_to_string(
            img,
            lang="eng+tam+hin+urd+tel+kan+mal+ben+guj+pan",
            config="--psm 6 --oem 3"
        )
        text = unicodedata.normalize("NFKC", text)
        for line in split_lines(text):
            raw_lines.append((i+1, line))

    char_count = {}
    total = 0
    for _, line in raw_lines:
        for c in line:
            s = get_script(c)
            if s:
                char_count[s] = char_count.get(s,0)+1
                total += 1

    script_ratio = {s:char_count[s]/total for s in char_count}
    valid_scripts = {s for s,r in script_ratio.items() if r >= 0.10}

    st.subheader("🧠 Detected Scripts")
    for s in valid_scripts:
        st.write(f"**{s}** → {script_ratio[s]*100:.2f}%")

    results = []
    for page, line in raw_lines:
        counts = {}
        for c in line:
            s = get_script(c)
            if s:
                counts[s] = counts.get(s,0)+1
        if not counts:
            continue

        dominant = max(counts, key=counts.get)
        if dominant not in valid_scripts:
            continue

        src = LANG_SCRIPT_MAP.get(dominant,"eng_Latn")
        if src == "eng_Latn":
            eng = line
        else:
            eng = translate(line, src)

        results.append((page, dominant, line, eng))

    st.subheader("📜 Page-wise OCR + Translation")
    for page, script, ocr, eng in results:
        st.markdown(f"### Page {page} — {script}")
        st.code(ocr)
        st.success(eng)
