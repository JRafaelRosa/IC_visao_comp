import os
import time
import cv2
import numpy as np
import onnxruntime as ort

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../yolov8n.onnx")

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

video_path = os.path.join(script_dir, "video/bf.mp4")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")

prev_time = 0


def postprocess(outputs, img_w, img_h, conf_threshold=0.45, nms_threshold=0.5):
    predictions = np.squeeze(outputs[0])
    if predictions.shape[0] == 84:
        predictions = predictions.T

    # Extrai coordenadas e scores
    boxes_raw = predictions[:, :4]
    scores_raw = predictions[:, 4:]

    # Filtra apenas classe 0 pessoa
    person_scores = scores_raw[:, 0]
    mask = person_scores > conf_threshold

    boxes_filtered = boxes_raw[mask]
    scores_filtered = person_scores[mask]

    if len(boxes_filtered) == 0:
        return []

    boxes = []
    confidences = []

    # Escala para a dimensão do vídeo
    x_factor = img_w / 640.0
    y_factor = img_h / 640.0

    for box, score in zip(boxes_filtered, scores_filtered):
        cx, cy, w, h = box

        # Converte para [x_min, y_min, largura, altura]
        left = int((cx - 0.5 * w) * x_factor)
        top = int((cy - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)

        boxes.append([left, top, width, height])
        confidences.append(float(score))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            results.append((x, y, x + w, y + h, confidences[i]))

    return results


window_name = "Inferencia - Raspberry Pi 5"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo.")
        break

    h_orig, w_orig = frame.shape[:2]

    # Pré-processamento
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Inferência
    start_time = time.time()
    outputs = session.run(None, {input_name: img})
    infer_time = (time.time() - start_time) * 1000

    # Pós-processamento com NMS
    detections = postprocess(outputs, w_orig, h_orig, conf_threshold=0.45, nms_threshold=0.5)

    # Desenho das detecções
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pessoa {score:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Métricas de FPS e Latência
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Latencia: {infer_time:.1f} ms", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()