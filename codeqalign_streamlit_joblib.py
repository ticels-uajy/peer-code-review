import re
import joblib
import random
import html
from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Model and deployment configuration
# =========================================================
# The reproducible training notebook saves the best model as one full
# scikit-learn Pipeline. The pipeline should already contain the vectorizer
# and classifier, so Streamlit only needs to load this one .joblib file.
#
# Recommended deployment structure:
#   app.py
#   outputs_pcr_ml_reproducible/
#       models/
#           best_model.joblib
#
# If you place best_model.joblib in the same directory as this app, it will
# also be detected automatically.
MODEL_FILE = Path("outputs_pcr_ml_reproducible/models/best_model.joblib")
MODEL_CANDIDATES = [
    MODEL_FILE,
    Path("best_model.joblib"),
    Path("models/best_model.joblib"),
]

STOPWORD_FILE = "stopword.txt"
CODE_SAMPLES_DIR = Path("code_samples")

# Set to False only if the saved joblib pipeline already includes its own
# text-cleaning transformer. For the reproducible notebook pipeline, keep this
# True because the model was trained on cleaned/preprocessed text.
APPLY_STREAMLIT_PREPROCESSING = True

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
        "field_key": "expressions",
        "title": "Expressions",
        "expected_label": "Expressions",
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


FALLBACK_CODE = """public class GradeChecker {
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


def list_code_samples() -> list[Path]:
    if not CODE_SAMPLES_DIR.exists():
        return []
    allowed_extensions = {".java", ".py", ".cpp", ".c", ".js", ".txt"}
    return sorted(
        [path for path in CODE_SAMPLES_DIR.iterdir() if path.suffix.lower() in allowed_extensions]
    )


def detect_code_language(file_path: Path | None) -> str:
    if file_path is None:
        return "java"
    extension_to_language = {
        ".java": "java",
        ".py": "python",
        ".cpp": "cpp",
        ".c": "c",
        ".js": "javascript",
        ".txt": "text",
    }
    return extension_to_language.get(file_path.suffix.lower(), "text")


def select_random_code_sample(force_new: bool = False):
    samples = list_code_samples()
    if not samples:
        st.session_state["selected_code_file"] = "Fallback sample"
        st.session_state["selected_code_language"] = "java"
        st.session_state["selected_code_text"] = FALLBACK_CODE
        return

    previous_file = st.session_state.get("selected_code_file")
    candidates = samples
    if force_new and previous_file and len(samples) > 1:
        candidates = [sample for sample in samples if sample.name != previous_file]

    selected_file = random.choice(candidates)
    st.session_state["selected_code_file"] = selected_file.name
    st.session_state["selected_code_language"] = detect_code_language(selected_file)
    st.session_state["selected_code_text"] = selected_file.read_text(encoding="utf-8")
    clear_prediction_result()


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


def resolve_model_path() -> Path | None:
    """Return the first available joblib model path."""
    for candidate in MODEL_CANDIDATES:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate
    return None


@st.cache_resource
def load_artifacts():
    """Load the full scikit-learn Pipeline saved as a .joblib file.

    The old version of this app loaded three separate artifacts:
    model, vectorizer, and encoder. The reproducible training notebook now
    saves a single full Pipeline, so the app only needs one file.
    """
    model_path = resolve_model_path()
    if model_path is None:
        expected_files = ", ".join(str(path) for path in MODEL_CANDIDATES)
        raise FileNotFoundError(
            "File model joblib belum ditemukan. Letakkan best_model.joblib pada salah satu path berikut: "
            + expected_files
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Artefak joblib berhasil dimuat, tetapi object tidak memiliki method predict(). "
            "Pastikan file berisi scikit-learn Pipeline atau estimator yang sudah dilatih."
        )

    return model, model_path


def normalize_label(label) -> str:
    label = str(label).strip().lower()
    label = label.replace("_", " ")
    label = label.replace("-", " ")
    label = re.sub(r"\s+", " ", label)
    return label


def extract_pipeline_classes(model) -> list[str]:
    """Extract class labels from a scikit-learn Pipeline or estimator."""
    if hasattr(model, "classes_"):
        return [str(label) for label in model.classes_]

    named_steps = getattr(model, "named_steps", {})
    if named_steps:
        last_step = list(named_steps.values())[-1]
        if hasattr(last_step, "classes_"):
            return [str(label) for label in last_step.classes_]

    return []


def predict_with_model(text: str):
    model, _model_path = load_artifacts()

    clean_text = preprocess_text(text)
    model_input = clean_text if APPLY_STREAMLIT_PREPROCESSING else str(text)

    raw_prediction = model.predict([model_input])
    predicted_label = str(raw_prediction[0])

    probability_df = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([model_input])[0]
        class_labels = extract_pipeline_classes(model)

        if not class_labels or len(class_labels) != len(probabilities):
            class_labels = [f"Class {index}" for index in range(len(probabilities))]

        probability_df = (
            pd.DataFrame({"Label": class_labels, "Probabilitas": probabilities})
            .sort_values("Probabilitas", ascending=False)
            .reset_index(drop=True)
        )

    return predicted_label, clean_text, probability_df


def demo_predict(text: str):
    clean_text = preprocess_text(text)
    lower_text = clean_text.lower()
    keyword_map = {
        "Variable": ["variabel", "variable", "nama", "penamaan", "singkat"],
        "Expressions": ["rumus", "formula", "perhitungan", "operasi", "tipe data", "ekspresi"],
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


def html_escape(value) -> str:
    """Escape user/model text before rendering inside HTML cards."""
    return html.escape(str(value), quote=True)



# =========================================================
# UI helper functions
# =========================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.7rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        section[data-testid="stSidebar"] {
            width: 285px !important;
            min-width: 285px !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.1rem;
        }

        .hero-card {
            padding: 1.25rem 1.45rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #f3f8ff 0%, #edf5ff 58%, #ffffff 100%);
            border: 1px solid #dbeafe;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.3rem;
        }

        .hero-title {
            font-size: 2.25rem;
            font-weight: 850;
            color: #111827;
            line-height: 1.12;
            margin-bottom: 0.35rem;
        }

        .hero-subtitle {
            font-size: 1.02rem;
            color: #1d4ed8;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .hero-desc {
            color: #4b5563;
            font-size: 0.97rem;
            line-height: 1.45;
            margin: 0;
        }

        .side-card {
            padding: 0.78rem 0.85rem;
            border-radius: 15px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.7rem;
        }

        .side-title {
            font-size: 0.88rem;
            font-weight: 850;
            color: #111827;
            margin-bottom: 0.22rem;
        }

        .side-text {
            font-size: 0.8rem;
            color: #4b5563;
            line-height: 1.38;
        }

        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }

        .status-active {
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }

        .status-demo {
            background: #ffedd5;
            color: #9a3412;
            border: 1px solid #fdba74;
        }

        .rubric-card {
            padding: 0.95rem 1rem 0.85rem 1rem;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.035);
            margin-top: 0.9rem;
        }

        .rubric-title {
            font-size: 1.05rem;
            font-weight: 850;
            color: #111827;
            margin-bottom: 0.3rem;
        }

        .rubric-help {
            font-size: 0.88rem;
            color: #6b7280;
            line-height: 1.45;
            margin-bottom: 0.62rem;
        }

        .example-box {
            padding: 0.72rem 0.82rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .summary-card {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.85rem;
        }

        .summary-ok {
            border-left: 7px solid #22c55e;
        }

        .summary-warn {
            border-left: 7px solid #f59e0b;
        }

        .summary-title {
            font-size: 1.02rem;
            font-weight: 850;
            color: #111827;
            margin-bottom: 0.45rem;
        }

        .summary-line {
            font-size: 0.92rem;
            color: #374151;
            line-height: 1.48;
            margin-bottom: 0.25rem;
        }

        .summary-suggestion {
            margin-top: 0.55rem;
            padding: 0.68rem 0.75rem;
            border-radius: 12px;
            background: #f9fafb;
            color: #374151;
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .prediction-input-box {
            margin-top: 0.65rem;
            padding: 0.72rem 0.82rem;
            border-radius: 12px;
            background: #f8fafc;
            border: 1px solid #dbe4ef;
            color: #334155;
            font-size: 0.9rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .preprocess-box {
            margin-top: 0.5rem;
            padding: 0.62rem 0.75rem;
            border-radius: 12px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #7c2d12;
            font-size: 0.86rem;
            line-height: 1.45;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .fix-link {
            display: inline-block;
            margin-top: 0.7rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: #eff6ff;
            color: #1d4ed8 !important;
            border: 1px solid #bfdbfe;
            font-weight: 750;
            font-size: 0.88rem;
            text-decoration: none !important;
        }

        .fix-link:hover {
            background: #dbeafe;
            color: #1e40af !important;
        }

        .shortcut-panel {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            margin: 0.9rem 0 1.2rem 0;
        }

        .shortcut-title {
            font-weight: 850;
            color: #92400e;
            margin-bottom: 0.55rem;
        }

        .success-submit-card {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            background: #f0fdf4;
            border: 1px solid #86efac;
            color: #14532d;
            line-height: 1.5;
            margin-top: 0.9rem;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
        }

        div.stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_card(title: str, text: str):
    st.sidebar.markdown(
        f"""
        <div class="side-card">
            <div class="side-title">{title}</div>
            <div class="side-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status_card(demo_mode: bool):
    if demo_mode:
        pill_class = "status-demo"
        pill_text = "Demo mode"
        description = "File model joblib belum ditemukan. UI dapat diuji, tetapi prediksi belum memakai model ML asli."
    else:
        pill_class = "status-active"
        pill_text = "Model ML aktif"
        description = "Model joblib scikit-learn Pipeline berhasil dimuat."

    st.sidebar.markdown(
        f"""
        <div class="side-card">
            <div class="side-title">Status Sistem</div>
            <span class="status-pill {pill_class}">{pill_text}</span>
            <div class="side-text" style="margin-top:0.45rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">CodeQAlign</div>
            <div class="hero-subtitle">Code Quality Alignment System for Peer Code Review Feedback</div>
            <p class="hero-desc">Sistem ini membantu mengecek apakah komentar peer code review mahasiswa sudah sesuai dengan kriteria rubrik kualitas kode.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def example_text(item: dict) -> str:
    return str(item.get("example", item.get("placeholder", ""))).replace("Contoh: ", "").strip()


def initialize_comment_state():
    for item in RUBRIC_FIELDS:
        st.session_state.setdefault(item["field_key"], "")
    st.session_state.setdefault("has_prediction_result", False)
    st.session_state.setdefault("last_result_df", None)
    st.session_state.setdefault("peer_review_submitted", False)
    if "selected_code_text" not in st.session_state:
        select_random_code_sample(force_new=False)


def clear_prediction_result():
    st.session_state["has_prediction_result"] = False
    st.session_state["last_result_df"] = None
    st.session_state["peer_review_submitted"] = False


def use_example(field_key: str, text: str):
    st.session_state[field_key] = text
    clear_prediction_result()


def clear_field(field_key: str):
    st.session_state[field_key] = ""
    clear_prediction_result()


def field_anchor(field_key: str) -> str:
    return f"field-{field_key}"


def mark_review_as_submitted():
    st.session_state["peer_review_submitted"] = True


def choose_another_code_sample():
    select_random_code_sample(force_new=True)


# =========================================================
# App layout
# =========================================================
st.set_page_config(
    page_title="CodeQAlign",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
initialize_comment_state()

try:
    load_artifacts()
    demo_mode = False
except Exception:
    demo_mode = ALLOW_DEMO_MODE_IF_MODEL_MISSING
    if not demo_mode:
        st.error("File model belum lengkap dan demo mode tidak aktif.")
        st.stop()

# Sidebar dibuat ringkas supaya tidak memunculkan scrollbar panjang.
st.sidebar.markdown("### CodeQAlign")
sidebar_card(
    "Tujuan Sistem",
    "Mengecek kesesuaian komentar mahasiswa dengan setiap kriteria code quality rubric.",
)
sidebar_card(
    "Label Model",
    "Variable, Expressions, Control Flow, Comments, Layout and Formatting, Decomposition, dan General.",
)
sidebar_status_card(demo_mode)

if demo_mode:
    with st.sidebar.expander("File model yang dibutuhkan"):
        st.code(
            "\n".join(str(path) for path in MODEL_CANDIDATES) + f"\n{STOPWORD_FILE}  # opsional",
            language="text",
        )

render_hero()

left_col, right_col = st.columns([1.0, 1.25], gap="large")

with left_col:
    st.subheader("Kode Program yang Direview")
    st.markdown(
        "<p class='small-muted'>Mahasiswa membaca contoh kode berikut, lalu memberi komentar berdasarkan setiap aspek kualitas kode.</p>",
        unsafe_allow_html=True,
    )
    sample_col1, sample_col2 = st.columns([1.4, 1])
    with sample_col1:
        st.caption(f"Kode aktif: {st.session_state.get('selected_code_file', 'Fallback sample')}")
    with sample_col2:
        st.button(
            "Tampilkan kode lain",
            on_click=choose_another_code_sample,
            use_container_width=True,
            key="choose_another_code",
        )

    st.code(
        st.session_state.get("selected_code_text", FALLBACK_CODE),
        language=st.session_state.get("selected_code_language", "java"),
    )

    with st.expander("Cara kerja sistem"):
        st.markdown(
            """
            1. Mahasiswa membaca kode program.
            2. Mahasiswa mengisi komentar pada 6 field rubrik.
            3. Sistem memprediksi label setiap komentar.
            4. Sistem mengecek apakah prediksi sesuai dengan field yang diisi.
            5. Jika belum sesuai, mahasiswa mendapat saran perbaikan.
            """
        )

with right_col:
    st.subheader("Form Peer Code Review")
    st.markdown(
        "<p class='small-muted'>Klik tombol <b>Gunakan contoh</b> agar contoh komentar otomatis masuk ke text field. Teks contoh juga dapat diblok dan disalin secara manual.</p>",
        unsafe_allow_html=True,
    )

    for item in RUBRIC_FIELDS:
        current_example = example_text(item)
        st.markdown(f"<div id='{field_anchor(item['field_key'])}'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="rubric-card">
                <div class="rubric-title">{item['title']}</div>
                <div class="rubric-help">{item['help']}</div>
                <div class="example-box"><b>Contoh komentar:</b><br>{current_example}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            st.button(
                "Gunakan contoh",
                key=f"use_example_{item['field_key']}",
                on_click=use_example,
                args=(item["field_key"], current_example),
                use_container_width=True,
            )
        with btn_col2:
            st.button(
                "Kosongkan",
                key=f"clear_{item['field_key']}",
                on_click=clear_field,
                args=(item["field_key"],),
                use_container_width=True,
            )

        st.text_area(
            label=f"Komentar untuk {item['title']}",
            placeholder=f"Tulis komentar untuk {item['title']} di sini.",
            height=105,
            key=item["field_key"],
            on_change=clear_prediction_result,
        )

    submitted = st.button("Submit Review", type="primary", use_container_width=True)


if submitted:
    st.session_state["peer_review_submitted"] = False
    st.session_state["has_prediction_result"] = False
    st.session_state["last_result_df"] = None

    user_comments = {
        item["field_key"]: st.session_state.get(item["field_key"], "")
        for item in RUBRIC_FIELDS
    }

    empty_fields = [
        item["title"]
        for item in RUBRIC_FIELDS
        if not user_comments[item["field_key"]].strip()
    ]
    if empty_fields:
        st.divider()
        st.header("Hasil Pengecekan Otomatis")
        st.error("Masih ada field yang belum diisi. Lengkapi komentar untuk: " + ", ".join(empty_fields))
        st.stop()

    results = []

    with st.spinner("Sistem sedang memprediksi klasifikasi komentar dan mengecek kesesuaian field..."):
        for item in RUBRIC_FIELDS:
            comment = user_comments[item["field_key"]]
            expected_label = item["expected_label"]
            predicted_label, clean_text, probability_df = predict_comment(comment, demo_mode)
            is_match = normalize_label(predicted_label) == normalize_label(expected_label)

            results.append({
                "Field Key": item["field_key"],
                "Kriteria Field": item["title"],
                "Label yang Diharapkan": expected_label,
                "Prediksi Sistem": predicted_label,
                "Status": "Sesuai" if is_match else "Perlu diperbaiki",
                "Komentar Mahasiswa": comment,
                "Saran": "Komentar sudah sesuai dengan kriteria field." if is_match else build_revision_suggestion(expected_label, predicted_label),
                "Hasil Preprocessing": clean_text,
            })

    st.session_state["last_result_df"] = pd.DataFrame(results)
    st.session_state["has_prediction_result"] = True


if st.session_state.get("has_prediction_result") and st.session_state.get("last_result_df") is not None:
    st.divider()
    st.header("Hasil Pengecekan Otomatis")

    result_df = st.session_state["last_result_df"]
    total_match = sum(result_df["Status"] == "Sesuai")
    total_fields = len(result_df)
    all_fields_match = total_match == total_fields

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Field sesuai", f"{total_match}/{total_fields}")
    metric_col2.metric("Field perlu revisi", f"{total_fields - total_match}")
    metric_col3.metric("Mode", "Demo" if demo_mode else "Model ML")

    if all_fields_match:
        st.success("Semua komentar sudah sesuai dengan kriteria rubrik. Review dapat dilanjutkan untuk dikirim.")
    else:
        st.warning(
            "Beberapa komentar belum sesuai dengan kriteria field. "
            "Cek kembali teks yang masuk ke sistem pada setiap kartu hasil, lalu tekan Perbaiki field ini jika perlu revisi."
        )

    st.subheader("Ringkasan Prediksi")

    for _, row in result_df.iterrows():
        is_ok = row["Status"] == "Sesuai"
        icon = "✅" if is_ok else "⚠️"
        card_class = "summary-ok" if is_ok else "summary-warn"
        fix_link = ""
        if not is_ok:
            fix_link = f"<a class='fix-link' href='#{field_anchor(row['Field Key'])}'>↟ Perbaiki field ini</a>"

        safe_field = html_escape(row["Kriteria Field"])
        safe_expected = html_escape(row["Label yang Diharapkan"])
        safe_prediction = html_escape(row["Prediksi Sistem"])
        safe_status = html_escape(row["Status"])
        safe_comment = html_escape(row.get("Komentar Mahasiswa", ""))
        safe_preprocess = html_escape(row.get("Hasil Preprocessing", ""))
        safe_suggestion = html_escape(row["Saran"])

        st.markdown(
            f"""
            <div class="summary-card {card_class}">
                <div class="summary-title">{icon} {safe_field}</div>
                <div class="summary-line"><b>Label yang diharapkan:</b> {safe_expected}</div>
                <div class="summary-line"><b>Prediksi sistem:</b> {safe_prediction}</div>
                <div class="summary-line"><b>Status:</b> {safe_status}</div>
                <div class="prediction-input-box"><b>Komentar mahasiswa yang masuk ke sistem untuk diprediksi:</b><br>{safe_comment}</div>
                <div class="preprocess-box"><b>Teks setelah preprocessing:</b><br>{safe_preprocess}</div>
                <div class="summary-suggestion"><b>Saran:</b> {safe_suggestion}</div>
                {fix_link}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if all_fields_match:
        st.button(
            "Submit komentar peer code review ke sistem",
            type="primary",
            use_container_width=True,
            key="submit_to_peer_review_tool",
            on_click=mark_review_as_submitted,
        )

        if st.session_state.get("peer_review_submitted"):
            st.success(
                "Komentar telah disubmit ke peer code review tool. "
                "Teman sejawat dapat melihat hasil komentar terhadap kode yang telah di-assess."
            )
            st.markdown(
                """
                <div class="success-submit-card">
                    <b>Review berhasil dikirim.</b><br>
                    Feedback terhadap kode yang dinilai telah tersimpan di sistem peer code review.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "Catatan: Sistem ini tidak menggantikan penilaian dosen. Sistem hanya memberi sinyal awal "
        "agar komentar mahasiswa lebih spesifik dan sesuai dengan kriteria rubrik."
    )
