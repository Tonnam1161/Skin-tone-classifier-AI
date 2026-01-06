from ultralytics import YOLO

def main():
    model = YOLO('yolov8n-cls.pt')

    result = model.train(
        data="./dataset",
        epochs=5,
        imgsz=224,
        device="cpu"
    )

    success = model.export(formatr='pt')

if __name__ == '__main__':
    main()