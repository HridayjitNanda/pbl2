import pickle
from pathlib import Path
import streamlit as st
st.write("APP STARTED")

# Initialize paths FIRST before any functions
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
if not (DATA_DIR / "model.pkl").exists() or not (DATA_DIR / "vectorizer.pkl").exists():
    alt_data_dir = BASE_DIR / "ml"
    if alt_data_dir.exists():
        DATA_DIR = alt_data_dir

MODEL_PATH = DATA_DIR / "model.pkl"
VECTORIZER_PATH = DATA_DIR / "vectorizer.pkl"
IMAGE_PATH = DATA_DIR / "static" / "img1.png"
STYLE_PATH = DATA_DIR / "static" / "style.css"


# Define functions that reference paths
def load_css(path: Path) -> None:
    try:
        if path.exists():
            st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {str(e)}")


@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_vectorizer():
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading vectorizer: {str(e)}")
        return None


# Now configure page
st.set_page_config(
    page_title="Real or Fake Job Description Classifier",
    page_icon="🚨",
    layout="wide",
)

# Load CSS
load_css(STYLE_PATH)

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-copy">
            <span class="eyebrow">Job Posting Authentication</span>
            <h1>Real or Fake <strong>Job Description</strong> Classifier</h1>
            <p>The Real or Fake Job Description Classifier is an automated tool using machine learning based classification techniques to detect fraudulent or authentic job postings.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.15, 0.85], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="hero-card">
            <h2>Is it Real or Fake?</h2>
            <p>Enter the job posting fields below and press <strong>Predict</strong> to classify whether the posting is likely real or fake. The model combines the available job posting fields into a single text prediction.</p>
            <p class="hero-tip"><strong>Tip:</strong> Fill in the job title, description, requirements, or company profile for a better prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    try:
        if IMAGE_PATH.exists():
            st.image(str(IMAGE_PATH), width=400)
        else:
            st.warning("Image not found: static/img1.png")
    except Exception as e:
        st.warning(f"Could not load image: {str(e)}")

if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
    st.error(
        "Model files not found. Ensure `model.pkl` and `vectorizer.pkl` are in the same folder as this app."
    )
else:
    # Load models
    model = load_model()
    vectorizer = load_vectorizer()
    
    if model is None or vectorizer is None:
        st.error("Failed to load model or vectorizer.")
        st.stop()
    
    # Form section
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    with st.form(key="job_form"):
        st.markdown("### Job Posting Details")

        title = st.text_input("Job Title")

        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            location = st.text_input("Location")
            benefits = st.text_input("Benefits")
            education = st.text_input("Required Education")
            industry = st.text_input("Industry")

        with col_right:
            department = st.text_input("Department")
            employment_type = st.selectbox(
                "Employment Type",
                ["", "Full-Time", "Part-Time", "Contract", "Temporary", "Other"],
            )
            experience = st.selectbox(
                "Required Experience",
                [
                    "",
                    "Internship",
                    "Entry level",
                    "Associate",
                    "Mid-Senior level",
                    "Director",
                    "Executive",
                    "Not applicable",
                    "Other",
                ],
            )
            function = st.text_input("Function")

        profile = st.text_area("Company Profile", height=140)
        requirements = st.text_area("Requirements", height=140)
        description = st.text_area("Description", height=140)

        predict_button = st.form_submit_button("Predict")
    st.markdown('</div>', unsafe_allow_html=True)

    # Make prediction
    if predict_button:
        try:
            combined_text = " ".join(
                [
                    title,
                    location,
                    department,
                    benefits,
                    employment_type,
                    experience,
                    education,
                    industry,
                    function,
                    profile,
                    requirements,
                    description,
                ]
            ).strip()

            if not combined_text:
                st.warning("Please fill in at least one field before predicting.")
            else:
                transformed_text = vectorizer.transform([combined_text])
                prediction = model.predict(transformed_text)
                if prediction[0] == 1:
                    st.error("⚠ Fake Job Posting")
                else:
                    st.success("✅ Real Job Posting")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
