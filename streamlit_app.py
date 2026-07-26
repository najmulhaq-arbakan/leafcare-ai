import streamlit as st
import tensorflow as tf
import numpy as np
import json
import tempfile

from tensorflow.keras.preprocessing import image
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LeafCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""
<style>

.main{
    padding-top:1rem;
}

h1,h2,h3{
    color:#2E7D32;
}

div.stButton > button{
    background-color:#2E7D32;
    color:white;
    border-radius:8px;
    height:3em;
    width:100%;
    font-size:16px;
}

div.stButton > button:hover{
    background-color:#1B5E20;
    color:white;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# Load AI Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    disease_model = tf.keras.models.load_model("plantdisease.keras")
    leaf_model = tf.keras.models.load_model("leaf_detector.keras")

    return disease_model, leaf_model


disease_model, leaf_model = load_models()

# --------------------------------------------------
# Load JSON Files
# --------------------------------------------------

with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("disease_info.json", "r", encoding="utf-8") as f:
    disease_database = json.load(f)

# --------------------------------------------------
# Settings
# --------------------------------------------------

LEAF_THRESHOLD = 0.90

# --------------------------------------------------
# Leaf Detection
# --------------------------------------------------

def is_leaf(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = leaf_model.predict(img, verbose=0)[0][0]

    leaf_probability = 1 - prediction

    return leaf_probability >= LEAF_THRESHOLD, leaf_probability * 100


# --------------------------------------------------
# Disease Prediction
# --------------------------------------------------

def predict_disease(img_path):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = disease_model.predict(
        img,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence
# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🌿 LeafCare AI")

st.sidebar.markdown(
    "AI-Based Plant Disease Detection"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 About This Project",
        "👨‍💻 Developer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.title("🌿 LeafCare AI")

    st.markdown("""
    ### AI-Powered Plant Disease Detection

    Upload a clear image of a plant leaf and let Artificial Intelligence identify plant diseases within seconds.
    """)

    st.success("🍃 Fast • Accurate • User-Friendly")

    st.markdown("---")

    st.write("""
    Welcome to **LeafCare AI**.

    

Upload a clear image of a plant leaf.

"""
    )
    st.markdown("---")

    st.subheader("📷 Try a Sample Image")

    sample_choice = st.selectbox(
        "Choose a sample image",
        [
            "None",
            "Healthy Tomato",
            "Potato Late Blight",
            "Grape Black Rot"
        ]
    )
    sample_paths = {
        "Healthy Tomato": "assets/samples/healthy tomato.JPG",
        "Potato Late Blight": "assets/samples/potato lateblight.JPG",
        "Grape Black Rot": "assets/samples/grape blackrot.JPG"
    }
    uploaded_file = st.file_uploader(
         "📤 Upload a Plant Leaf Image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None or sample_choice != "None":
    
        left_col, right_col = st.columns([0.8, 1.4])
    
        with left_col:
    
            if uploaded_file is not None:

                st.image(
                    uploaded_file,
                    caption="Uploaded Leaf",
                    width=250
                )

            else:

                st.image(
                    sample_paths[sample_choice],
                    caption=sample_choice,
                    width=250
                )
    
        with right_col:
    
            st.write("### Ready for Prediction")

            detect_button = st.button(
                "🔍 Detect Disease",
                use_container_width=True
            )

            if detect_button:

                with st.spinner("🧠 AI is analyzing the image..."):

                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

                    if uploaded_file is not None:
                        temp_file.write(uploaded_file.getvalue())

                    else:
                        with open(sample_paths[sample_choice], "rb") as f:
                            temp_file.write(f.read())

                    temp_file.close()

                    leaf_ok, leaf_confidence = is_leaf(temp_file.name)

                    if not leaf_ok:

                        st.error("❌ The uploaded image is not recognized as a plant leaf.")

                        st.write(
                            f"Leaf Confidence: **{leaf_confidence:.2f}%**"
                        )

                    else:

                        prediction, confidence = predict_disease(temp_file.name)

                        if "___" in prediction:

                            plant_name, disease_name = prediction.split("___")

                        else:

                            plant_name = prediction
                            disease_name = ""

                        plant_name = (
                            plant_name
                            .replace("_(including_sour)", " (including sour)")
                            .replace("_(maize)", " (maize)")
                            .replace("_", " ")
                        )

                        disease_name = (
                            disease_name
                            .replace("_", " ")
                            .replace("healthy", "Healthy")
                        )

                        st.success("✅ Disease detected successfully!")

                        st.markdown("---")

                        card1, card2, card3 = st.columns(3)

                        with card1:
                            st.markdown("### 🌱 Plant")
                            st.success(plant_name)

                        with card2:
                            st.markdown("### 🦠 Disease")
                            st.success(disease_name)

                        with card3:
                            st.markdown("### Confidence")
                            st.success(f"{confidence:.2f}%")

                        st.progress(confidence / 100)
                        # =====================================
                        # Confidence Level
                        # =====================================

                        if confidence >= 95:
                            st.success(f"🟢 High Confidence ({confidence:.2f}%)")

                        elif confidence >= 80:
                            st.warning(f"🟡 Moderate Confidence ({confidence:.2f}%)")

                        else:
                            st.error(f"🔴 Low Confidence ({confidence:.2f}%)")
                        # -----------------------------------------------
                        # Disease Information
                        # -----------------------------------------------

                        info = disease_database.get(prediction)

                        if info:

                            st.markdown("---")

                            st.header("📖 Disease Information")

                            tab1, tab2, tab3 = st.tabs(
                                [
                                    "📝 Description",
                                    "⚠ Symptoms",
                                    "💊 Treatment"
                                ]
                            )

                            with tab1:

                                st.info(info["description"])

                                st.subheader("🧬 Causes")

                                st.write(info["causes"])

                            with tab2:

                                st.warning(info["symptoms"])

                            with tab3:

                                st.success(info["treatment"])

                                st.subheader("🛡 Prevention")

                                st.success(info["prevention"])

# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "📖 About This Project":

    st.title("📖 About LeafCare AI")

    st.markdown("""
LeafCare AI is a **personal AI learning project** developed to strengthen practical skills in:

- 🤖 Deep Learning
- 🌱 Plant Disease Detection
- 👁 Computer Vision
- 🌐 AI Web Application Development
""")

    st.markdown("---")

    st.header("🚀 Features")

    st.markdown("""
- 🌿 Plant Leaf Detection
- 🦠 Plant Disease Classification
- 📊 Prediction Confidence
- 📖 Disease Information
- 💊 Treatment Suggestions
- 🛡 Prevention Recommendations
""")

    st.markdown("---")

    st.header("🧠 AI Models")

    st.markdown("""
This application uses two TensorFlow models:

1. **Leaf Detection Model**
   - Determines whether the uploaded image is a plant leaf.

2. **Plant Disease Classification Model**
   - Predicts the plant species and disease.
""")

    st.markdown("---")

    st.header("📚 Datasets")

    st.markdown("""
### 🌿 Plant Disease Dataset

- PlantVillage Dataset
- 38 plant disease classes

### 🍃 Leaf / Non-Leaf Dataset

Custom dataset created manually for this project.

Classes:

- Leaf
- Non-Leaf
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    st.markdown("""
- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pillow
""")
    st.markdown("---")

    st.subheader("🌱 Supported Plants")

    st.write("""
    LeafCare AI was trained using the **PlantVillage** dataset and can detect diseases in the following **14 plant species**:

    - 🍎 Apple
    - 🫐 Blueberry
    - 🍒 Cherry (including Sour Cherry)
    - 🌽 Corn (Maize)
    - 🍇 Grape
    - 🍊 Orange
    - 🍑 Peach
    - 🫑 Bell Pepper
    - 🥔 Potato
    - 🍓 Strawberry
    - 🍅 Tomato
    - 🫘 Soybean
    - 🍈 Squash
    - ❤️ Raspberry
    """)

    st.markdown("---")

    st.header("⚠ Limitations")

    st.warning("""
- Only diseases included in the trained dataset can be recognized.
- Best results require a clear image of a **single leaf**.
- Low lighting or blurry images may reduce accuracy.
- Images containing multiple leaves or complex backgrounds can affect predictions.
- This project is intended for educational and learning purposes.
""")

# ==================================================
# DEVELOPER PAGE
# ==================================================

elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            "assets/profile.jpg",
            width=220
        )

    with col2:

        st.header("Najmul Haq")

        st.write("AI & Machine Learning Enthusiast")

        st.write("""
This project was developed as a **personal AI learning project**
to strengthen practical skills in:

- Deep Learning
- Computer Vision
- TensorFlow
- Streamlit
- AI Deployment
""")

    st.markdown("---")

    st.subheader("📧 Email")

    st.write("najmulhaq446@gmail.com")

    st.subheader("🐙 GitHub")

    st.markdown(
        "[github.com/najmulhaq-arbakan](https://github.com/najmulhaq-arbakan)"
    )

    st.subheader("💼 LinkedIn")

    st.markdown(
        "[Najmul Haq LinkedIn](https://www.linkedin.com/in/najmul-haq-912591380)"
    )

    st.markdown("---")

    st.success("Thank you for visiting LeafCare AI! 🌿")

st.markdown("---")

st.caption(
    "🌿 LeafCare AI • Built with Streamlit & TensorFlow • © 2026 Najmul Haq"
)