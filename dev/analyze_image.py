import warnings
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from lang_sam import LangSAM
import time
from dev.drawing import display_image_with_boxes
from dev.postprocessing import crop_image

def detect_bb_on_image(model: LangSAM, image: Image,text_prompt: str,  box_threshold=0.3, text_threshold=0.25, show_run_time=False, show_results=True):
    if show_run_time:
        # start time
        start = time.time()
    boxes, logits, phrases = model.predict_dino(image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    if show_run_time:
        # end time
        end = time.time()
        print(f"Time: {(end - start):.2f}s")
    if len(boxes) == 0:
        warnings.warn("No bounding boxes found")
        return boxes, logits, phrases
    if show_results:
        display_image_with_boxes(image, boxes, logits, return_fig_as_numpy=False)
        # img_c = crop_image(image, boxes)
        # save cropped image
        # img_c.save("results/crop_building3.png")
        # plt.imshow(img_c)
        # plt.show()
    return boxes, logits, phrases

def run_inference( image: Image,text_prompt: str,model: LangSAM,  box_threshold=0.25, text_threshold=0.25, show_run_time=False):
    boxes, logits, phrases = detect_bb_on_image(model, image, text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, show_run_time=show_run_time, show_results=True)
    if len(boxes) == 0:
        warnings.warn("No bounding boxes found")
        return boxes, logits, phrases
    return boxes, logits, phrases


if __name__ == '__main__':
    image_path =  "data/crop_building3.png"
    model = LangSAM()
    text_prompt = "buildings"
    image = Image.open(image_path).convert("RGB")
    run_inference(image, text_prompt, model)
