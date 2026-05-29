import re
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


MODEL_FILE = "RF_Count_Vectors.sav"
VECTORIZER_FILE = "count_vect_model.sav"
ENCODER_FILE = "encoder.sav"
STOPWORD_FILE = "stopword.txt"

ALLOW_DEMO_MODE_IF_MODEL_MISSING = True


RUBRIC_FIELDS = [
    {
        "field_key": "variable",
        "title": "Variable Names",
        "expected_label": "Variable",
        "help": "Komentar sebaiknya membahas penamaan variabel, kejelasan nama, konsistensi, dan kesesuaian nama variabel dengan fungsi/purpose.",
        "placeholder": "Contoh: Nama variabel masih terlalu singkat seperti a, b, dan c. Sebaiknya gunakan nama yang lebih jelas seperti nilaiTugas, nilaiUjian, atau rataRata."
    },
    {
        "field_key": "expression",
        "title": "Expressions",
        "expected_label": "Expression",
        "help": "Komentar sebaiknya membahas ekspresi, rumus, tipe data, operasi perhitungan, dan kesederhanaan formula.",
        "placeholder": "Contoh: Rumus perhitungan rata-rata sudah benar, tetapi akan lebih jelas jika operasi perhitungan dipisahkan ke variabel khusus."
    },
    {
        "field_key": "control_flow",
        "title": "Control Flow",
        "expected_label": "Control Flow",
        "help": "Komentar sebaiknya membahas alur kontrol, percabangan, perulangan, kondisi, struktur if/else, dan penanganan kondisi khusus.",
        "placeholder": "Contoh: Struktur if-else sudah mudah dipahami, tetapi sebaiknya ditambahkan validasi untuk nilai yang tidak sesuai rentang."
    },
    {
        "field_key": "comments",
        "title": "Comments",
        "expected_label": "Comments",
        "help": "Komentar sebaiknya membahas keberadaan komentar program, header comment, inline comment, dan apakah komentar membantu memahami kode.",
        "placeholder": "Contoh: Kode belum memiliki komentar yang menjelaskan fungsi utama program. Tambahkan komentar singkat pada bagian perhitungan dan pengecekan status."
    },
    {
        "field_key": "layout",
        "title": "Layout and Formatting",
        "expected_label": "Layout and Formatting",
        "help": "Komentar sebaiknya membahas indentasi, spasi, kerapian layout, pengelompokan kode, dan konsistensi format.",
        "placeholder": "Contoh: Layout program sudah cukup rapi, tetapi spasi antarbagian dapat ditambahkan agar bagian input, proses, dan output lebih mudah dibaca."
    },
    {
        "field_key": "decomposition",
        "title": "Decomposition",
        "expected_label": "Decomposition",
        "help": "Komentar sebaiknya membahas pembagian kode ke fungsi/prosedur/modul/class, pengurangan duplikasi, dan pemisahan tanggung jawab.",
        "placeholder": "Contoh: Program masih ditulis seluruhnya di main method. Sebaiknya perhitungan rata-rata dan penentuan status dibuat dalam method terpisah."
    },
]


SAMPLE_CODE = """public class GradeChecker {
    public static void main(String[] args) {
        int a = 80;
        int b = 75;
        int c = 90;

        int d = (a + b + c) / 3;

        if (d >= 60) {
            System.out.println("Pass");
        } else {
            System.out.println("Fail");
        }
    }
}
"""


def remove_upper_case(text: str) -> str:
    text = str(text)
    words = text.split()
    stripped = [word.title() if word.isupper() else word for word in words]
    return " ".join(stripped)


def remove_url(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", str(text))


def remove_html(text: str) -> str:
    return re.sub(r"<.*?>", "", str(text))


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", str(text))


def text_to_word_sequence_like_keras(text: str) -> list[str]:
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    text = str(text).lower()
    for char in filters:
        text = text.replace(char, " ")
    return text.split()


@st.cache_data
def load_stopwords() -> set[str]:
    path = Path(STOPWORD_FILE)
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def preprocess_text(text: str) -> str:
    stopwords = load_stopwords()
    text = remove_upper_case(text)
    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)
    tokens = text_to_word_sequence_like_keras(text)
    tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)


@st.cache_resource
def load_artifacts():
    missing_files = [
        file_name
        for file_name in [MODEL_FILE, VECTORIZER_FILE, ENCODER_FILE]
        if not Path(file_name).exists()
    ]
    if missing_files:
        raise FileNotFoundError("File model belum lengkap: " + ", ".join(missing_files))

    with open(MODEL_FILE, "rb") as file:
        model = pickle.load(file)
    with open(VECTORIZER_FILE, "rb") as file:
        vectorizer = pickle.load(file)
    with open(ENCODER_FILE, "rb") as file:
        encoder = pickle.load(file)
    return model, vectorizer, encoder


def normalize_label(label) -> str:
    label = str(label).strip().lower()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    return label


def decode_prediction(raw_prediction, encoder):
    try:
        return encoder.inverse_transform(raw_prediction)[0]
    except Exception:
        return raw_prediction[0]


def get_class_labels(model, encoder):
    try:
        return list(encoder.inverse_transform(model.classes_))
    except Exception:
        return list(model.classes_)


def predict_with_model(text: str):
    model, vectorizer, encoder = load_artifacts()
    clean_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_text])
    raw_prediction = model.predict(vectorized_text)
    predicted_label = decode_prediction(raw_prediction, encoder)

    probability_df = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_text)[0]
        class_labels = get_class_labels(model, encoder)
        probability_df = pd.DataFrame({"Label": class_labels, "Probabilitas": probabilities}).sort_values("Probabilitas", ascending=False)

    return predicted_label, clean_text, probability_df


def demo_predict(text: str):
    clean_text = preprocess_text(text)
    lower_text = clean_text.lower()
    keyword_map = {
        "Variable": ["variabel", "variable", "nama", "penamaan", "singkat"],
        "Expression": ["rumus", "formula", "perhitungan", "operasi", "tipe data", "ekspresi"],
        "Control Flow": ["if", "else", "perulangan", "loop", "kondisi", "alur", "validasi"],
        "Comments": ["komentar", "comment", "penjelasan", "dokumentasi"],
        "Layout and Formatting": ["rapi", "indentasi", "spasi", "format", "layout", "baris"],
        "Decomposition": ["fungsi", "method", "modul", "class", "dekomposisi", "main", "dipisah"],
    }
    scores = {label: sum(1 for keyword in keywords if keyword in lower_text) for label, keywords in keyword_map.items()}
    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        best_label = "General"

    probability_df = pd.DataFrame({
        "Label": list(scores.keys()) + ["General"],
        "Probabilitas": [scores[label] for label in scores] + [1 if best_label == "General" else 0],
    })
    total = probability_df["Probabilitas"].sum()
    probability_df["Probabilitas"] = probability_df["Probabilitas"] / total if total > 0 else 0
    probability_df = probability_df.sort_values("Probabilitas", ascending=False)
    return best_label, clean_text, probability_df


def predict_comment(text: str, demo_mode: bool):
    if demo_mode:
        return demo_predict(text)
    return predict_with_model(text)


def build_revision_suggestion(expected_label: str, predicted_label: str) -> str:
    if normalize_label(predicted_label) == "general":
        return f"Komentar pada field ini terdeteksi terlalu umum. Perbaiki komentar agar lebih spesifik membahas aspek {expected_label}."
    return f"Komentar pada field ini lebih terdeteksi sebagai {predicted_label}, bukan {expected_label}. Cek kembali isi komentar dan fokuskan pada aspek {expected_label}."


st.set_page_config(page_title="PCR Feedback Alignment Prototype", page_icon="🧪", layout="wide")

st.title("Prototype Peer Code Review Feedback Alignment")
st.caption("Prototype ini menggambarkan integrasi model klasifikasi single-label ke dalam peer code review tool.")

with st.sidebar:
    st.header("Tujuan Sistem")
    st.write("Sistem membantu mengecek apakah komentar mahasiswa pada setiap field rubrik sudah sesuai dengan kriteria yang dimaksud.")
    st.divider()
    st.subheader("Label Model")
    st.write("Model dapat memprediksi label: Variable, Expression, Control Flow, Comments, Layout and Formatting, Decomposition, dan General.")
    st.info("UI prototype ini hanya menyediakan 6 field rubrik. Tidak ada field General, karena General digunakan sistem untuk mendeteksi komentar yang terlalu umum atau tidak sesuai dengan kriteria tertentu.")
    st.divider()

    try:
        load_artifacts()
        demo_mode = False
        st.success("Model ML aktif.")
    except Exception as error:
        demo_mode = ALLOW_DEMO_MODE_IF_MODEL_MISSING
        if demo_mode:
            st.warning("File model belum lengkap. Aplikasi berjalan dalam mode prototype/demo.")
            with st.expander("Detail file yang dibutuhkan"):
                st.code(f"{MODEL_FILE}\n{VECTORIZER_FILE}\n{ENCODER_FILE}\n{STOPWORD_FILE}", language="text")
        else:
            st.error(str(error))
            st.stop()

left_col, right_col = st.columns([1.1, 1.2], gap="large")

with left_col:
    st.subheader("Kode Program yang Direview")
    st.write("Contoh kode berikut diberikan kepada mahasiswa. Mahasiswa diminta memberi komentar berdasarkan setiap aspek kualitas kode.")
    st.code(SAMPLE_CODE, language="java")
    with st.expander("Cara kerja prototype"):
        st.markdown("""
        1. Mahasiswa membaca kode program.
        2. Mahasiswa mengisi komentar pada 6 field rubrik.
        3. Setelah menekan tombol submit, sistem memprediksi label setiap komentar.
        4. Sistem mengecek apakah prediksi label sesuai dengan field tempat komentar ditulis.
        5. Jika belum sesuai, sistem meminta mahasiswa memperbaiki komentar pada field tertentu.
        """)

with right_col:
    st.subheader("Form Peer Code Review")
    with st.form("peer_review_form"):
        user_comments = {}
        for item in RUBRIC_FIELDS:
            st.markdown(f"**{item['title']}**")
            st.caption(item["help"])
            user_comments[item["field_key"]] = st.text_area(
                label=f"Komentar untuk {item['title']}",
                placeholder=item["placeholder"],
                height=95,
                label_visibility="collapsed",
                key=item["field_key"],
            )
        submitted = st.form_submit_button("Submit Review", type="primary", use_container_width=True)

if submitted:
    st.divider()
    st.header("Hasil Pengecekan Otomatis")

    empty_fields = [item["title"] for item in RUBRIC_FIELDS if not user_comments[item["field_key"]].strip()]
    if empty_fields:
        st.error("Masih ada field yang belum diisi. Lengkapi komentar untuk: " + ", ".join(empty_fields))
        st.stop()

    results = []
    probability_outputs = {}

    with st.spinner("Sistem sedang memprediksi klasifikasi komentar dan mengecek kesesuaian field..."):
        for item in RUBRIC_FIELDS:
            comment = user_comments[item["field_key"]]
            expected_label = item["expected_label"]
            predicted_label, clean_text, probability_df = predict_comment(comment, demo_mode)
            is_match = normalize_label(predicted_label) == normalize_label(expected_label)
            results.append({
                "Kriteria Field": item["title"],
                "Label yang Diharapkan": expected_label,
                "Prediksi Sistem": predicted_label,
                "Status": "Sesuai" if is_match else "Perlu diperbaiki",
                "Saran": "Komentar sudah sesuai dengan kriteria field." if is_match else build_revision_suggestion(expected_label, predicted_label),
                "Hasil Preprocessing": clean_text,
            })
            probability_outputs[item["title"]] = probability_df

    result_df = pd.DataFrame(results)
    total_match = sum(result_df["Status"] == "Sesuai")
    total_fields = len(result_df)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Field sesuai", f"{total_match}/{total_fields}")
    metric_col2.metric("Field perlu revisi", f"{total_fields - total_match}")
    metric_col3.metric("Mode", "Demo" if demo_mode else "Model ML")

    if total_match == total_fields:
        st.success("Semua komentar sudah sesuai dengan kriteria rubrik. Review dapat dilanjutkan untuk dikirim.")
    else:
        st.warning("Beberapa komentar belum sesuai dengan kriteria field. Mahasiswa disarankan mengecek kembali field yang ditandai agar komentar lebih spesifik.")

    st.subheader("Ringkasan Prediksi")
    st.dataframe(result_df[["Kriteria Field", "Label yang Diharapkan", "Prediksi Sistem", "Status", "Saran"]], use_container_width=True, hide_index=True)

    st.subheader("Detail Setiap Field")
    for _, row in result_df.iterrows():
        status_icon = "✅" if row["Status"] == "Sesuai" else "⚠️"
        with st.expander(f"{status_icon} {row['Kriteria Field']} — {row['Status']}"):
            st.write(f"**Label yang diharapkan:** {row['Label yang Diharapkan']}")
            st.write(f"**Prediksi sistem:** {row['Prediksi Sistem']}")
            st.write(f"**Saran:** {row['Saran']}")
            st.markdown("**Hasil preprocessing komentar:**")
            st.code(row["Hasil Preprocessing"], language="text")

            proba = probability_outputs.get(row["Kriteria Field"])
            if proba is not None and not proba.empty:
                st.markdown("**Probabilitas prediksi:**")
                st.dataframe(proba, use_container_width=True, hide_index=True)

    st.caption("Catatan: Prototype ini tidak menggantikan penilaian dosen. Sistem hanya memberi sinyal awal agar komentar mahasiswa lebih sesuai dengan kriteria rubrik.")
