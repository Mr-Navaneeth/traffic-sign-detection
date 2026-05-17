import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

model = YOLO("C:/Users/navan/Desktop/Traffic_sign_detection/yolov3_Dataset/runs/detect/train/weights/best.pt")

def camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.60)

        annotated_frame = results[0].plot()

        cv2.imshow("Traffic Sign Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def video_upload(vpath):
    cap = cv2.VideoCapture(vpath)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.60)

        annotated_frame = results[0].plot()

        cv2.imshow("Traffic Sign Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# print("1 - Camera Input")
# print("2 - Video Input")
#
# choice = int(input("Enter your choice: "))
# if choice == 1:
#     camera()
# elif choice == 2:
#     video_path = input("Enter video path: ")
#     video_upload(video_path)


# Gradio Camera Function

def gradio_camera(image):
    frame = np.array(image)

    results = model(frame)

    annotated_frame = results[0].plot()

    return annotated_frame

def run_camera():
    camera()
    return "Camera closed"

def run_video_opencv(video):
    video_upload(video)
    return "Video processed"

# Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("# Traffic Sign Detection System")

# ----------- OpenCV REAL-TIME -----------
    with gr.Tab("Real-Time (OpenCV)"):
        gr.Markdown("Click button to start real-time detection")

        cam_btn = gr.Button("Start Camera")
        cam_status = gr.Textbox(label="Status")

        cam_btn.click(run_camera, outputs=cam_status)

# ----------- Gradio Image Detection -----------
    with gr.Tab("Image Detection"):
        cam_input = gr.Image(type="numpy", label="Capture / Upload Image")
        cam_output = gr.Image(label="Output")

        cam_input.change(gradio_camera, inputs=cam_input, outputs=cam_output)

# ----------- OpenCV Video -----------
    with gr.Tab("Video (OpenCV)"):
        video_input = gr.Video()
        video_status = gr.Textbox(label="Status")

        video_input.change(run_video_opencv, inputs=video_input, outputs=video_status)

demo.launch()