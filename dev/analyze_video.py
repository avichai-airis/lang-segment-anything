import numpy as np
from dev.analyze_image import Frame
from dev.analyze_image import run_inference
from dev.drawing import display_image_with_boxes
from lang_sam import LangSAM
import cv2
from PIL import Image
from dev.postprocessing import crop_image
from tqdm import tqdm
from dev.utils import cal_max_bb
import shutil
import os
import matplotlib.pyplot as plt
import imageio as iio

class Video:
    def __init__(self, video_path: str, text_prompt: str = 'buildings'):
        self.text_prompt = text_prompt
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.fps = int(iio.get_reader(self.video_path).get_meta_data()['fps'])
        self.frames = []
        self.resize_frames = []
        self.crop_frames = []
        self.read_video()

    def read_video(self):
        while (self.video_capture.isOpened()):
            ret, frame = self.video_capture.read()
            if ret == False:
                break
            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frames.append(Image.fromarray(frame))
        self.video_capture.release()


    def get_next_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.frames.append(frame)
            return frame
        else:
            self.release()
            return None

    def release(self):
        self.video_capture.release()

    @staticmethod
    def pad_frame(frame, max_x, max_y):
        return np.pad(frame, ((0, max_x - frame.shape[0]), (0, max_y - frame.shape[1]), (0, 0)),
                      'constant', constant_values=min(frame.flatten()))

    def convert_frames_to_video(self, output_path, fps=30):
        # Write the frames to a video.
        height, width, channels = np.array(self.resize_frames[0]).shape
        width = width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # pad all images to the same size
        for i in range(len(self.crop_frames)):
            # self.crop_frames[i] = self.pad_frame(np.array(self.crop_frames[i]), max_x, max_y)
            # concat original and cropped
            curr_con = np.concatenate((np.array(self.resize_frames[i]), np.array(self.crop_frames[i])), axis=1)

            # convert to RGB
            # frame = cv2.cvtColor(self.crop_frames[i], cv2.COLOR_BGR2RGB)
            video.write(curr_con)
        # Release the video.
        video.release()

    def analyze_video(self, box_threshold=0.3, text_threshold=0.25, output_path="results/",save_original_vs_cropped:bool = False, debug: bool = False):
        model = LangSAM()
        i = 0
        # create results dir
        if not os.path.exists(output_path + self.video_path.split("/")[-1].split(".")[0]):
            os.makedirs(output_path + self.video_path.split("/")[-1].split(".")[0])
        else:
            # remove old results
            shutil.rmtree(output_path + self.video_path.split("/")[-1].split(".")[0])
            os.makedirs(output_path + self.video_path.split("/")[-1].split(".")[0])
        for pil_frame in tqdm(self.frames[::10]):
            frame = Frame(pil_frame,  self.text_prompt)
            frame.detect_bb(model, box_threshold=box_threshold, text_threshold=text_threshold)
            if debug:
                frame.set_boxes_on_image()
                # save image with boxes
                frame.image_with_boxes.save(output_path + self.video_path.split("/")[-1].split(".")[0] + f"/{i}_boxes.png")
                i += 1
                continue
            # get the cropped image
            crop_frame = frame.crop_image()
            self.crop_frames.append(crop_frame)

            # save cropped frame
            if save_original_vs_cropped:
                resize_frame = frame.image.resize((512, 512))
                Image.fromarray(np.concatenate((np.array(resize_frame), np.array(crop_frame)), axis=1)).save('debug/'+ self.video_path.split("/")[-1].split(".")[0] + f"/{i}.png")
            else:
                crop_frame.save(output_path+ self.video_path.split("/")[-1].split(".")[0] + f"/{i}.png")
            i += 1
            print(f"Frame {i} done")
        print("Done")



def run_dir():
    # go over all videos in dir
    for video in os.listdir("data/real_data"):
        if video.endswith(".mp4"):
            print(f"Analyzing {video}")
            video = Video("data/real_data/" + video)
            video.analyze_video()
    print("********************** finished analyzing all videos **********************")
if __name__ == '__main__':

    video = Video("data/real_data/W7_EDlXWTBiXAEEniNoMPwAAYamdpeGl2cXZqAYsGfNuqAYsGfNtTAAAAAQ.mp4")
    video.analyze_video(output_path="debug/", debug=False)



