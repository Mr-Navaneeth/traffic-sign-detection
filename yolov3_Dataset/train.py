# import torch
# print(torch.cuda.is_available())

from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    model.train(
        data='C:/Users/navan/Desktop/Traffic_sign_detection/yolov3_Dataset/data.yaml',
        epochs=80,
        imgsz=640,
        device=0
    )
if __name__ == '__main__':
    main()