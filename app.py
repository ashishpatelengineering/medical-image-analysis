import streamlit as st
import base64
from groq import Groq

# Step 1: Streamlit App Title
st.title("Medical Image Analysis")

# Step 2: API Key Input
st.subheader("Enter GROQ API Key")
api_key = st.text_input(
    "GROQ API Key", 
    placeholder="Enter your GROQ API key here", 
    type="password"
)

if not api_key:
    st.info("Please enter your GROQ API key to proceed.")
    st.stop()

# Step 3: Image Upload
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Step 4: Query Input
    st.subheader("Enter your Query")
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
