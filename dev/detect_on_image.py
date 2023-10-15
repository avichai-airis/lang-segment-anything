import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import time
from dev.drawing import display_image_with_boxes

def detect_bb_on_image(model: LangSAM, image: Image,text_prompt: str,  box_threshold=0.3, text_threshold=0.25, show_run_time=False):
    if show_run_time:
        # start time
        start = time.time()
    boxes, logits, phrases = model.predict_dino(image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    if show_run_time:
        # end time
        end = time.time()
        print(f"Time: {(end - start):.2f}s")
    return boxes, logits, phrases
def run_inference( image_path: str,text_prompt: str,  box_threshold=0.3, text_threshold=0.25, show_run_time=False):
    model = LangSAM()
    image = Image.open(image_path).convert("RGB")
    boxes, logits, phrases = detect_bb_on_image(model, image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, show_run_time=show_run_time)
    display_image_with_boxes(image, boxes, logits)

if __name__ == '__main__':
    image_path =  "data/crop_building.png"
    run_inference(image_path, "buildings")
