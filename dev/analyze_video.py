from dev.analyze_image import run_inference
from lang_sam import LangSAM
import cv2

def read_video(video_path):
    frame_rate = 30
    model = LangSAM()
    # read video
    cap = cv2.VideoCapture(video_path)
    # Create a list to store the frames.
    frames = []
    i = 0
    # Loop over the frames in the video.
    while True:

        # Capture a frame.
        ret, frame = cap.read()

        # If the frame is empty, break out of the loop.
        if not ret:
            break
        if i % frame_rate == 0:

            # Convert the frame from BGR to RGB color space.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to PIL format.
            frame_pil = Image.fromarray(frame)

            # Detect objects in the frame.
            masks, boxes, _, logits = detect_sign_in_image(frame_pil, text_prompt, model)
            if len(masks) == 0:
                print(f"No objects of the '{text_prompt}' prompt detected in the image.")
            else:
                # Display the image with bounding boxes and confidence scores
                data = display_image_with_boxes(frame_pil, boxes, logits)
                frames.append(data)
                print(f'frame number {i}')
        i += 1
    # Release the video.
    cap.release()

# img_c = crop_image(image, boxes)
# plt.imshow(img_c)
# plt.show()

if __name__ == '__main__':
    pass