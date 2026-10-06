from ultralytics import YOLO

def convert_rpi():
    # Carrega o modelo pré-treinado
    model = YOLO("yolov8n.pt")

    # Exporta em formato ONNX otimizado (CPU ARM)
    model.export(format="onnx", imgsz=640, simplify=True)