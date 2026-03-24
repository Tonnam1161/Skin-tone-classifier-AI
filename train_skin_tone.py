from ultralytics import YOLO
def main():
    model = YOLO('yolov8n-cls.pt')
    results = model.train(
        data="./dataset",
        epochs=5,
        imgsz=224,
        device="cpu"
    )
    print(f"Training finished. Results saved in: {results.save_dir}")
    export_path = model.export(format='torchscript')
    print(f"Export finished: {export_path}")
if __name__ == '__main__':
    main()