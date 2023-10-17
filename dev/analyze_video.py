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

    def analyze_video(self, box_threshold=0.3, text_threshold=0.25):
        model = LangSAM()
        i = 0
        # create results dir
        if not os.path.exists("results/" + self.video_path.split("/")[-1].split(".")[0]):
            os.makedirs("results/" + self.video_path.split("/")[-1].split(".")[0])
        else:
            # remove old results
            shutil.rmtree("results/" + self.video_path.split("/")[-1].split(".")[0])
            os.makedirs("results/" + self.video_path.split("/")[-1].split(".")[0])
        for pil_frame in tqdm(self.frames[::self.fps]):
            frame = Frame(pil_frame,  self.text_prompt)
            frame.detect_bb(model, box_threshold=box_threshold, text_threshold=text_threshold)
            # get the cropped image
            crop_frame = frame.crop_image()
            self.crop_frames.append(crop_frame)

            # save cropped frame
            # resize_frame = frame.image.resize((512, 512))
            # Image.fromarray(np.concatenate((np.array(resize_frame), np.array(crop_frame)), axis=1)).save("results/"+ self.video_path.split("/")[-1].split(".")[0] + f"/{i}.png")
            crop_frame.save("results/"+ self.video_path.split("/")[-1].split(".")[0] + f"/{i}.png")
            i += 1
            print(f"Frame {i} done")
        print("Done")


#     def read_video_imageio(video_path):
#         # read video
#         frames = iio.mimread(video_path, memtest=False)
#         # calculate the fps
#         fps = iio.get_reader(video_path).get_meta_data()['fps']
#         return frames, fps
#         model = LangSAM()
#         cropped_frames = []
#         original_frames = []
#         frames, fps = read_video_cv2(video_path)
#
#         for frame in tqdm(frames[::int(fps)]):
#             # convert numpy to PIL image
#             frame = Image.fromarray(frame)
#             boxes, logits, phrases = run_inference(frame, "buildings", model)
#             if len(boxes) == 0:
#                 curr_crop_frame = frame
#             else:
#                 curr_crop_frame = crop_image(frame, boxes)
#             curr_crop_frame = np.array(curr_crop_frame)
#             cropped_frames.append(curr_crop_frame)
#
#             # if max_x < curr_crop_frame.shape[0]:
#             #     max_x = curr_crop_frame.shape[0]
#             # if max_y < curr_crop_frame.shape[1]:
#             #     max_y = curr_crop_frame.shape[1]
#             original_frames.append(np.array(frame))
#             # save cropped frame
#             # curr_crop_frame.save("results/"+ video_path.split("/")[-1].split(".")[0] + f"/cropped_{i}.png")
#             i += 1
#         max_x = original_frames[0].shape[0]
#         max_y = original_frames[0].shape[1]
#         convert_frames_to_video(cropped_frames, original_frames, "results/" + video_path.split("/")[-1], max_x, max_y,
#                                 fps=1)
#
#
#     def read_video_cv2(video_path):
#         # read video
#         cap = cv2.VideoCapture(video_path)
#         # calculate the fps
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frames = []
#         while(cap.isOpened()):
#             ret, frame = cap.read()
#             if ret == False:
#                 break
#             # convert to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         cap.release()
#         return frames, fps
#
# def run_global_crop(video_path):
#     g_bb_min_x = 100000
#     g_bb_min_y = 100000
#     g_bb_max_x = 0
#     g_bb_max_y = 0
#
#     model = LangSAM()
#     cropped_frames = []
#     original_frames = []
#     frames, fps = read_video_cv2(video_path)
#
#     for frame in tqdm(frames[::int(fps)]):
#         # convert numpy to PIL image
#         frame = Image.fromarray(frame)
#         boxes, logits, phrases = run_inference(frame, "buildings", model)
#         original_frames.append(np.array(frame))
#         if len(boxes) != 0:
#             bb_min_x, bb_min_y, bb_max_x, bb_max_y = cal_max_bb(boxes)
#             if bb_min_x < g_bb_min_x:
#                 g_bb_min_x = bb_min_x
#             if bb_min_y < g_bb_min_y:
#                 g_bb_min_y = bb_min_y
#             if bb_max_x > g_bb_max_x:
#                 g_bb_max_x = bb_max_x
#             if bb_max_y > g_bb_max_y:
#                 g_bb_max_y = bb_max_y
#
#     for frame in tqdm(original_frames):
#         cropped_frames.append(frame[int(g_bb_min_x):int(g_bb_max_x), int(g_bb_min_y):int(g_bb_max_y), :])
#     max_x = original_frames[0].shape[0]
#     max_y = original_frames[0].shape[1]
#     convert_frames_to_video(cropped_frames, original_frames, "results/global_crop/" + video_path.split("/")[-1], max_x, max_y,
#                             fps=1)
#
#
#     pass
#
#
# def run(video_path):
#     model = LangSAM()
#     cropped_frames = []
#     original_frames = []
#     frames, fps = read_video_cv2(video_path)
#     # make results dir
#
#     # if not os.path.exists("results/"+ video_path.split("/")[-1].split(".")[0]):
#     #     os.makedirs("results/"+ video_path.split("/")[-1].split(".")[0])
#     # else:
#     #     # remove old results
#     #
#     #     shutil.rmtree("results/"+ video_path.split("/")[-1].split(".")[0])
#     i = 0
#     for frame in tqdm(frames[::int(fps)]):
#         # convert numpy to PIL image
#         frame = Image.fromarray(frame)
#         boxes, logits, phrases = run_inference(frame,"buildings", model)
#         if len(boxes) == 0:
#             curr_crop_frame = frame
#         else:
#             curr_crop_frame = crop_image(frame, boxes)
#         curr_crop_frame = np.array(curr_crop_frame)
#         cropped_frames.append(curr_crop_frame)
#
#
#
#         # if max_x < curr_crop_frame.shape[0]:
#         #     max_x = curr_crop_frame.shape[0]
#         # if max_y < curr_crop_frame.shape[1]:
#         #     max_y = curr_crop_frame.shape[1]
#         original_frames.append(np.array(frame))
#         # save cropped frame
#         # curr_crop_frame.save("results/"+ video_path.split("/")[-1].split(".")[0] + f"/cropped_{i}.png")
#         i += 1
#     max_x = original_frames[0].shape[0]
#     max_y = original_frames[0].shape[1]
#     convert_frames_to_video(cropped_frames,original_frames, "results/"+ video_path.split("/")[-1],max_x, max_y, fps=1)



# img_c = crop_image(image, boxes)
# plt.imshow(img_c)
# plt.show()

if __name__ == '__main__':

    # import imageio as iio
    # import numpy as np
    # # for idx, frame in enumerate(iio.imiter("data/gaza1.mp4")):
    # #     print(f"Frame {idx}: avg. color {np.sum(frame, axis=-1)}")
    # # read video
    # frames = iio.mimread("data/gaza2.mp4", memtest=False)
    # # calculate the fps
    # fps = iio.get_reader("data/gaza2.mp4").get_meta_data()['fps']
    # print(f"fps {fps}")
    # frames = iio.imread("data/gaza1.mp4", plugin="pyav")

    video = Video("data/20231007_072204_hamza20300_159828.mp4")
    video.analyze_video()
    # video.convert_frames_to_video('results/' + video.video_path.split("/")[-1].split(".")[0] + '.mp4', fps=1)
    # print(f"num frames {len(frames)}")
    # run_global_crop("data/gaza1.mp4")
    # run_global_crop("data/gaza2.mp4")
    # run_global_crop("data/20231007_072204_hamza20300_159828.mp4")
