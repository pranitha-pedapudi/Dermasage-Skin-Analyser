import numpy as np
from PIL import Image
import streamlit as st
from utils import set_img_as_background, load_trained_model, predict_model

# Define logo path
LOGO_PATH = "asset/background.jpg"  # Update this path as per your file location

def home():
    """
    Display the home page content with a background image and a title.
    """
    set_img_as_background("asset/dashboard.png")
    st.markdown(""" 
        ## <span style='color:#F79BD3'>Know your skin type with just a selfie</span>
    """, unsafe_allow_html=True)

def classification(model):
    """
    Perform skin type classification based on user-uploaded or captured images.
    """
    list_class = ["Oily Skin", "Dry Skin", "Combination Skin", "Normal Skin", "Sensitive Skin"]
    
    skin_tips = {
        "Oily Skin": "💡 Use oil-free moisturizers and gentle cleansers.",
        "Dry Skin": "💡 Keep skin hydrated with deep moisturizing creams.",
        "Combination Skin": "💡 Balance your routine with a mix of light & rich products.",
        "Normal Skin": "💡 Maintain with a simple routine and SPF protection.",
        "Sensitive Skin": "💡 Use fragrance-free products and avoid harsh chemicals."
    }

    def _inference(upload_file):
        image = Image.open(upload_file).convert("RGB")  # Ensure RGB mode
        image = image.resize((150, 150))  # Resize to match model input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.reshape(image, (-1, 150, 150, 3))  # Ensure correct shape

        # Predict the skin type
        result = predict_model(image, model)[0]
        
        # Get the predicted class and its confidence score
        pred_index = np.argmax(result)
        pred_class = list_class[pred_index]
        confidence = round(result[pred_index] * 100, 2)

        # Display results
        section_img, section_table = st.columns([2, 3])
        section_img.image(upload_file, caption="Uploaded Image", use_column_width=True)
        
        with section_table:
            st.markdown(f"### 🏷️ **Predicted Skin Type:** `{pred_class}`")
            st.progress(confidence / 100)  # Confidence progress bar
            st.markdown(f"🔎 **Confidence:** `{confidence}%`")
            st.markdown(f"📌 **Tip:** {skin_tips[pred_class]}")  # Display recommendations

        # Display full probability table
        st.table({
            "Skin Type": list_class, 
            "Confidence (%)": [round(i * 100, 2) for i in result]
        })

    options = st.selectbox(
        "📸 Choose an option:", 
        ["", "Select Image", "Take a Picture"]
    )
    
    upload_file = None
    if options == "Select Image":
        upload_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if options == "Take a Picture":
        upload_file = st.camera_input("Capture an image")

    if upload_file is not None:
        _inference(upload_file)

def about():
    """
    Display information about different skin types based on user selection.
    """
    options = st.selectbox(
        "Find out more about your skin type here👇", 
        ["", "Normal Skin", "Oily Skin", "Dry Skin", "Combination Skin", "Sensitive Skin"]
    )
    if options:
        skin_info = {
            "Normal Skin": ("asset/dashboard.png", "Normal skin is well-balanced, not too dry or oily. It is less prone to issues."),
            "Oily Skin": ("asset/berminyak.jpg", "Oily skin produces excess sebum, causing shine and acne."),
            "Dry Skin": ("asset/kering.jpg", "Dry skin lacks moisture, leading to roughness and irritation."),
            "Combination Skin": ("asset/kombinasi.jpg", "Combination skin has both oily and dry areas."),
            "Sensitive Skin": ("asset/sensitif.jpeg", "Sensitive skin reacts easily to products and environmental factors.")
        }
        set_img_as_background(skin_info[options][0])
        st.markdown(f"## {options}\n\n{skin_info[options][1]}")

def interface():
    """
    Create the Streamlit app interface with tabs for Home, Classification, and About.
    """
    model = load_trained_model()

    # Add logo in the top left corner
    col1, col2 = st.columns([1, 4])  # Adjust column widths
    with col1:
        st.image(LOGO_PATH, width=100)  # Adjust size as needed
    with col2:
        st.title("Skin Type Analyzer")

    tab_home, tab_clf, tab_about = st.tabs(["Home", "Classification", "About"])
    with tab_home:
        home()

    with tab_clf:
        classification(model)

    with tab_about:
        about()

if __name__ == "__main__":
    interface()
