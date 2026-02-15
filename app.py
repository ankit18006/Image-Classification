import streamlit as st
import jax
import jax.numpy as jnp
import pickle
import numpy as np
from PIL import Image
from model import AdvancedCNN

st.set_page_config(page_title="AI Digit Recognizer", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Digit Recognition System")
st.markdown("Upload handwritten digit image (0-9)")

@st.cache_resource
def load_model():
    model = AdvancedCNN()
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((1,28,28,1))
    params = model.init(key, dummy)['params']
    
    with open("saved_model.pkl","rb") as f:
        trained_params = pickle.load(f)
    return model, trained_params

model, params = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28,28))
    st.image(image, caption="Uploaded Image", width=200)

    img_array = np.array(image)/255.0
    img_array = img_array.reshape((1,28,28,1))
    img_array = jnp.array(img_array)

    if st.button("🔍 Predict"):
        logits = model.apply({'params': params}, img_array)
        probs = jax.nn.softmax(logits)
        prediction = int(jnp.argmax(probs))
        confidence = float(jnp.max(probs))*100

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}%")