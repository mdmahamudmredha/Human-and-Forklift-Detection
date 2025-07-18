import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

# YOLOv5 model load (custom trained model)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def detect_objects(image):
    # Image → NumPy → BGR (YOLO expects BGR)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detection
    results = model(img)

    # Render bounding boxes
    results.render()
    result_img = results.ims[0]  # result image in BGR format

    # Convert BGR → RGB for display
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_img)

# Gradio Interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Image(type="pil", label="Detected output"),
    title="🚧 Human & Forklift Detection",
    description="Upload an image to detect humans and forklifts using a YOLOv5 custom model."
)

if __name__ == "__main__":
    demo.launch()
