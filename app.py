import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
model.config.tie_word_embeddings = False  # optional: silences the warning you saw

def caption_image(img: Image.Image):
    if img is None:
        return "Please upload an image."
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)

    return processor.decode(out[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Caption"),
    title="Image Captioning with BLIP",
    description="Upload an image and get an AI-generated caption."
)

if __name__ == "__main__":
    demo.launch()