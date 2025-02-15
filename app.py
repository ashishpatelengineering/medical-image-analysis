import streamlit as st
import base64
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step 1: Streamlit App Title and Description
st.header("AI-Assisted Medical Image Analysis", anchor=False, divider="blue")
st.write("This app provides analysis of medical images to assist with understanding potential health conditions. However, the results generated should not be considered as medical advice. It is important to consult with a qualified healthcare professional for proper diagnosis and treatment.")

# Step 2: API Key from Environment Variable
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ API Key not found. Please set it in your environment variables.")
    st.stop()

# Step 3: Image Upload
st.subheader("Upload an Image", anchor=False, divider="rainbow")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Step 4: Query Input
    st.subheader("Enter your Query",anchor=False, divider="blue")
    query = st.text_input("What would you like to ask about the image?", placeholder="Type your query here...")

    # Step 5: Main Logic
    if query:
        st.write("Analyzing image...")

        # Convert image to Base64
        def encode_image(file):
            """Encodes an image to Base64 format."""
            return base64.b64encode(file.read()).decode('utf-8')

        encoded_image = encode_image(uploaded_file)

        # Analyze the image
        def analyze_image_with_query(api_key, query, model, encoded_image):
            """Analyzes an image using GROQ API."""
            client = Groq(api_key=api_key)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ]
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model
            )
            return chat_completion.choices[0].message.content

        model = "llama-3.2-90b-vision-preview"

        try:
            response = analyze_image_with_query(api_key, query, model, encoded_image)
            st.success("Analysis Complete!")
            st.write("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an image to proceed.")
