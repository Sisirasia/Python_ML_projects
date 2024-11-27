import streamlit as st
from PIL import Image, ImageDraw, ImageFont

def add_watermark(image, watermark_text):
    """Add a watermark to the provided image."""
    watermark_image = image.copy()
    draw = ImageDraw.Draw(watermark_image)

    # Define font size based on the image size
    font_size = int(min(watermark_image.size) / 10)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_width, text_height = draw.textsize(watermark_text, font=font)
    position = (watermark_image.size[0] - text_width - 10, watermark_image.size[1] - text_height - 10)
    draw.text(position, watermark_text, fill=(255, 255, 255, 128), font=font)
    return watermark_image

# Streamlit app
st.title("Image Watermarking App")
st.write("Upload an image, add a custom watermark, and save the result!")

# File uploader for image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Input watermark text
    watermark_text = st.text_input("Enter Watermark Text", value="Your Watermark")

    # Add watermark button
    if st.button("Add Watermark"):
        if watermark_text.strip() == "":
            st.warning("Please enter a watermark text.")
        else:
            # Add watermark to the image
            watermarked_image = add_watermark(uploaded_image, watermark_text)

            # Display the watermarked image
            st.image(watermarked_image, caption="Watermarked Image", use_column_width=True)

            # Download option for the watermarked image
            output_buffer = st.file_uploader
            output_buffer = st.download_button(
                "Download Watermarked Image",
                data=watermarked_image.tobytes(),
                file_name="watermarked_image.png",
                mime="image/png",
            )
