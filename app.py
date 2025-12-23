import streamlit as st
import torch
import os
import sys
import pickle

# --- 1. Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    # Import the correct class name (TwoStageModelPipeline)
    from src.model_pipeline import TwoStageModelPipeline 
except ImportError as e:
    st.error(f"Import Error: Could not load the TwoStageModelPipeline class. Check dependencies inside model_pipeline.py. Original Error: {e}")
    st.stop()

# --- 2. Global Configurations ---
MODEL_DIR = os.path.join(current_dir, 'models')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_sastrawi.pkl')
BERT_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'bert_model.pkl') # Must be .pkl
BERT_BASE_NAME = "indobenchmark/indobert-base-p2" 

# 🧠 Define the multi-class labels (Index to Name)
LABEL_MAP = {
    0: "Provokasi", 1: "Penghinaan", 2: "Pornografi",
    3: "Negatif Lainnya", 4: "SARA", 5: "Pencemaran Nama Baik"
}
LABEL_NAME_MAP = {v: k for v, k in LABEL_MAP.items()}

WARNING_MSG = {
    "Pornografi" : "Konten terdeteksi mengarah ke **pornografi**. Penyebaran konten asusila melanggar Pasal 27 ayat (1) UU ITE dan UU Pornografi. Hapus atau ubah pesan ini segera.",
    "Pencemaran Nama Baik" : "Pesan ini terindikasi **pencemaran nama baik**. Perbuatan ini diancam pidana berdasarkan Pasal 27 ayat (3) UU ITE atau Pasal 310 KUHP. Mohon ubah pernyataan Anda.",
    "Provokasi" : "Pesan ini terindikasi **provokasi/penghasutan**. Mengajak melakukan tindak pidana diatur dalam Pasal 160 KUHP. Hati-hati, ini termasuk tindak kriminal.",
    "SARA" : "Pesan ini terindikasi **ujaran kebencian (SARA)**. Menyebarkan informasi SARA melanggar Pasal 28 ayat (2) UU ITE. Segera ubah pesan Anda.",
    "Penghinaan" : "Pesan ini terindikasi **penghinaan**. Penghinaan tetap memiliki konsekuensi hukum. Pasal 315 KUHP berlaku.",
    "Negatif Lainnya" : "Pesan ini mengandung **konten negatif**. Walaupun belum terklasifikasi spesifik, pesan ini berpotensi melanggar peraturan pidana. Harap bijak berkomentar!"
}


# --- CRITICAL FIX: Custom unpickler for CUDA compatibility ---
class CPU_Unpickler(pickle.Unpickler):
    """
    A custom unpickler that remaps storage locations to 'cpu' during deserialization.
    This resolves the "Attempting to deserialize object on a CUDA device" error 
    when loading a pickled PyTorch model.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_local_legacy_format':
            # PyTorch's internal loading function is intercepted here 
            return lambda *args: torch.load(*args, map_location='cpu')
        else:
            return super().find_class(module, name)
# -----------------------------------------------------------


# --- 3. Model Loading (Cached) ---
@st.cache_resource
def load_full_pipeline():
    """Loads all models and initializes the pipeline class."""
    
    if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(TFIDF_VECTORIZER_PATH) or not os.path.exists(BERT_WEIGHTS_PATH):
        st.error(f"FATAL ERROR: One or more model files not found in '{MODEL_DIR}'.")
        st.info(f"Check file paths: RF={os.path.exists(RF_MODEL_PATH)}, TFIDF={os.path.exists(TFIDF_VECTORIZER_PATH)}, BERT={os.path.exists(BERT_WEIGHTS_PATH)}")
        st.stop()
        
    try:
        pipeline = TwoStageModelPipeline(
            rf_model_path=RF_MODEL_PATH,
            tfidf_path=TFIDF_VECTORIZER_PATH,
            bert_model_path=BERT_BASE_NAME, 
            label_map=LABEL_NAME_MAP 
        )

        device = pipeline.device
        
        with open(BERT_WEIGHTS_PATH, "rb") as f:
            loaded_bert_model = CPU_Unpickler(f).load() 
            
        pipeline.bert_model = loaded_bert_model
        pipeline.bert_model.eval()
        
        return pipeline

    except Exception as e:
        # If this fails, the error is likely the RF dtype mismatch (NumPy version)
        st.error(f"Error loading models or initializing pipeline. Final Error: {e}")
        st.stop()

try:
    pipeline = load_full_pipeline()
except SystemExit:
    pipeline = None

# --- 4. Streamlit UI State Management ---

if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
    
def classify_text():
    """Runs the model prediction and updates the session state."""
    if pipeline is None:
        st.error("Model pipeline failed to initialize.")
        return
        
    input_text = st.session_state.input_widget

    if input_text:
        with st.spinner('Running two-stage prediction...'):
            final_label = pipeline.predict(input_text) 
        
        st.session_state.input_text = input_text
        st.session_state.final_result = final_label
    else:
        st.session_state.input_text = ""
        st.session_state.final_result = None
        st.warning("Please enter some text to classify.")



# --- 5. Custom UI Components ---

def draw_detector_ui(header_title, result_label=None, sub_label=None, message=None):
    
    st.markdown("""
        <style>
            .input-box-display {
                background-color: #e6f7ff; 
                border: 2px solid #a6c8ff; 
                border-radius: 8px;
                padding: 10px;
                min-height: 100px;
                color: #333;
                margin-bottom: 10px;
                white-space: pre-wrap; 
            }
            .result-neg-main { background-color: #B50000; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em; }
            .result-neg-sub { background-color: #fce7e7; color: #d9363e; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em;}
            .result-non-neg { background-color: #8f8; color: black; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 1.2em; }
            .description-area { background-color: #eee; padding: 10px; border-radius: 5px; min-height: 80px; }
            .header-title-style { margin-bottom: 5px; }
        </style>
    """, unsafe_allow_html=True)
    
    # RESULT SECTION
    if result_label:
        st.subheader("RESULT")
        st.write("Komentar anda tergolong:")

        if result_label == "NEGATIF":
            st.markdown(f'<div class="result-neg-main">{result_label}</div>', unsafe_allow_html=True)
            if sub_label:
                st.write("")
                st.write("Terklasifikasi dalam subkategori:")
                st.markdown(f'<div class="result-neg-sub">{sub_label.upper()}</div>', unsafe_allow_html=True)
                st.write("")
                if sub_label in WARNING_MSG:
                    st.error(WARNING_MSG[sub_label])
                else:
                    st.info("Pesan ini termasuk dalam perilaku kriminal. Harap berhati-hati karena ada potensi melanggar peraturan pidana.")

            st.write("Mohon ubah komentar Anda sebelum di-*post* ke media sosial ya!")
            
        elif result_label == "NON-NEGATIF":
            st.markdown(f'<div class="result-non-neg">{result_label}</div>', unsafe_allow_html=True)
            st.write("")
            st.write("Aman untuk di-*post* ke media sosial $\\text{👍}$")


# --- Main App Body ---

# Sidebar
with st.sidebar:
    st.header("Penjelasan Subkategori Ujaran Kebencian")
    st.write("Klik kategori di bawah ini untuk memahami dasar hukum dan definisinya.")

    with st.expander("Pornografi"):
        st.markdown(
            """
            **Definisi:** Bentuk pesan (teks, gambar, dll) yang memuat kecabulan yang melanggar norma kesusilaan dalam masyarakat.

            **Landasan Hukum:** 
            - **UU Pornografi No. 44 Tahun 2008 Pasal 4**:
                Setiap orang dilarang memproduksi, membuat, memperbanyak, menyebarluaskan, dan/atau membuat dapat diaksesnya pornografi.
            - **Pasal 27 Ayat (1) UU ITE**:
                Dilarang mendistribusikan atau mentransmisikan informasi elektronik yang memiliki muatan melanggar kesusilaan.

            """
        )
    
    with st.expander("Pencemaran Nama Baik"):
        st.markdown(
            """
            **Definisi:** Merendahkan atau menyebarkan informasi yang tidak benar terkait reputasi seseorang.
            
            **Landasan Hukum:**
            - **KUHP pasal 310 - 321** (Terutama di pasal 310 dan 311)
                Pasal 310 Ayat 2:
                Pencemaran nama baik yang tertuju pada tindakan yang terjadi secara tertulis atau tidak langsung
                Menekankan pada pelanggaran yang terjadi secara tidak langsung, yaitu unggahan pada forum publik. Bisa didenda 4.5 juta, atau hukuman penjara paling lama 1 tahun 4 bulan.

            - **UU Nomor 19 Tahun 2016**:
                Perubahan dari UU No 11 tahun 2008, tentang informasi dan transaksi elektronik (ITE)
                Pasal 45 ayat 3, menjelaskan bahwa barang siapa sengaja menyebarkan informasi elektronik yang bermuatan penghinaan dan membuat citra orang lain rusak, terancam hukuman penjara maksimal 4 tahun dan/atau denda paling banyak 750 juta.

            - **UU ITE Pasal 27 ayat (3)**
                
            """
        )
    
    with st.expander("Provokasi"):
        st.markdown(
            """
            **Definisi:** Tindakan menghasut, pancingan untuk memicu reaksi dan tanggapan yang kuat (memancing perdebatan, memunculkan ketidaksetujuan).

            
            **Landasan Hukum:**
            - **Pasal 160 KUHP**:
                Barang siapa dengan lisan atau tulisan menghasut supaya melakukan perbuatan pidana, kekerasan terhadap penguasa umum atau tidak menuruti ketentuan undang - undang diancam pidana 6 tahun.
            - **Pasal 28 ayat (2) UU ITE**:
                Dilarang menyebarkan informasi elektronik yang mengandung ujaran kebencian atau permusuhan berdasarkan SARA.

            """
        )
        
    with st.expander("SARA"):
        st.markdown(
            """
            **Definisi:** Merendahkan, menyebarkan kebencian terhadap kelompok berdasarkan suku, agama, ras, golongan tertentu.
            
            **Landasan Hukum:** 
            - **Pasal 28 ayat (2) UU ITE No 19 Tahun 2006**:
                Setiap orang dengan sengaja dan tanpa hak menyebarkan informasi yang ditujukan untuk menimbulkan rasa kebencian atau permusuhan individu dan/atau kelompok masyarakat tertentu berdasarkan atas SARA dapat dipidana.
            - **Pasal 156 KUHP**:
                Barang siapa menyatakan perasaan permusuhan, kebencian, atau penghinaan terhadap satu atau beberapa golongan rakyat indonesia, diancam dipidana penjara maksimal 4 tahun.
            """
        )
    
    with st.expander("Penghinaan"):
        st.markdown(
            """
            **Definisi:** Perbuatan atau ucapan yang menyerang kehormatan atau martabat seseorang, biasanya merendahkan, mencemooh, mengejek, atau mempermalukan.
            
            **Landasan Hukum:** 
            - **Pasal 310 KUHP**
            - **Pasal 315 KUHP**

            """
        )

    with st.expander("Negatif Lainnya"):
        st.markdown(
            """
            **Definisi:** Segala bentuk ujaran kebencian yang merendahkan orang lain yang tidak termasuk dalam 5 subkategori di atas.

            """
        )

# --- Top Header & Instructions ---
st.set_page_config(page_title="🇮🇩 Detektor Komentar Negatif", layout="wide")
st.title("🇮🇩 DETEKTOR KOMENTAR NEGATIF")
st.markdown("""
Model ini akan mengklasifikasikan kalimat komentar yang Anda masukkan ke dalam salah satu kategori: **Negatif** atau **Non-Negatif**, dan subkategori: **Pornografi**, **Pencemaran Nama Baik**, **Provokasi**, **SARA**, **Penghinaan**, dan **Negatif Lainnya** (untuk komentar Negatif).
""")
st.info("💡**Pro tip**: Buka *sidebar* untuk melihat penjelasan tiap subkategori.")

# --- Input Area ---
st.subheader("Masukkan Teks Komentar")
main_input_text = st.text_area(
    "Teks Komentar:",
    placeholder="Masukkan kalimat komentar (Bahasa Indonesia) atau emoji di sini...",
    key='input_widget',
    height=100,
    label_visibility="collapsed"
)
st.markdown("""
    Contoh kata/kalimat :
    1. Emoji : $\\text{😤} \\text{😠} \\text{🤬} \\text{😡}$
    2. Slang / Singkatan: anj**, b**g**t
    3. Kalimat & Emoji : botak jijik banget $\\text{🤢} \\text{🤮}$
    4. Kalimat saja: muka lu kayak babi
""")

# --- Prediction Button ---
st.button("Proses Kalimat", type="primary", on_click=classify_text)

st.markdown("---")

# --- Result Display Logic ---
if st.session_state.get('final_result'):
    
    final_result = st.session_state.final_result
    input_text_val = st.session_state.input_text

    if final_result == "Non-Negatif":
         draw_detector_ui(
            header_title='Result', 
            result_label='NON-NEGATIF',
            message=input_text_val 
        )
    else:
        draw_detector_ui(
            header_title='Result', 
            result_label='NEGATIF', 
            sub_label=final_result,
            message=input_text_val 

        )
