import os
from ultralytics import YOLO


def train():
    data_path = "dataset/pedestrian"
    result_path = "results"

    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found")

    print("Loading model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded")

    # Rodando na CPU elimina qualquer crash do ROCm/MIOpen
    results = model.predict(
        source=data_path,
        classes=[0],
        conf=0.5,
        save=True,
        project=result_path,
        name="teste_pedestre",
        exist_ok=True,
        device="cpu"
    )

    print("teste concluido")