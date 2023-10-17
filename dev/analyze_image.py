import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lang_sam import LangSAM
import time
from dev.postprocessing import crop_image

class Frame:
    """
    Frame class to hold image and text prompt
    """
    def __init__(self, image: Image, text_prompt: str):
        self.image = image
        self.text_prompt = text_prompt
        self.no_bb = True
        self.boxes = None
        self.logits = None
        self.phrases = None
        self.image_with_boxes = None

    @staticmethod
    def cal_max_bb(boxes):
        # calculate the crop size based on the bounding box
        x_min, y_min = boxes.numpy()[:, 0:2].min(axis=0)
        x_max, y_max = boxes.numpy()[:, 2:4].max(axis=0)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def crop_image(image, boxes):
        # calculate the crop size based on the bounding box
        x_min, y_min, x_max, y_max = self.cal_max_bb(boxes)
        image = image.crop((x_min, y_min, x_max, y_max))
        return image
    def crop_image(self, boxes):
        return crop_image(self.image, boxes)

    def convert_fig_to_numpy(self, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def set_boxes_on_image(self,):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        ax.set_title("Image with Bounding Boxes")
        ax.axis('off')

        for box, logit in zip(self.boxes, self.logits):
            x_min, y_min, x_max, y_max = box
            confidence_score = round(logit.item(), 2)  # Convert logit to a scalar before rounding
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Draw bounding box
            rect = plt.Rectangle((x_min, y_min), box_width, box_height, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

            # Add confidence score as text
            ax.text(x_min, y_min, f"Confidence: {confidence_score}", fontsize=8, color='red', verticalalignment='top')

        self.image_with_boxes = self.convert_fig_to_numpy(fig)
        plt.close(fig)

    def detect_bb(self, model: LangSAM, box_threshold=0.3, text_threshold=0.25):
        start = time.time()
        self.boxes, self.logits, self.phrases = model.predict_dino(self.image, self.text_prompt, box_threshold=box_threshold,
                                                    text_threshold=text_threshold)
        end = time.time()
        print(f"Time: {(end - start):.2f}s")
        if len(self.boxes) == 0:
            warnings.warn("No bounding boxes found")
        else:
            self.no_bb = False

    def show_image_with_boxes(self):
        plt.imshow(self.image_with_boxes)
        plt.show()


def run_inference( image: Image,text_prompt: str,model: LangSAM,  box_threshold=0.25, text_threshold=0.25, show_run_time=False):
    frame = Frame(image, text_prompt)
    frame.detect_bb(model, box_threshold=box_threshold, text_threshold=text_threshold)
    return frame


if __name__ == '__main__':
    image_path = "data/crop_building3.png"
    model = LangSAM()
    text_prompt = "buildings"
    image = Image.open(image_path).convert("RGB")
    run_inference(image, text_prompt, model)
