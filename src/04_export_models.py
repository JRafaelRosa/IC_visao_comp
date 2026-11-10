#!/usr/bin/env python3
"""
04_export_models.py - Exporta o best.pt para ONNX (na CPU) e confere o resultado.

Uso:
    python src/04_export_models.py --weights runs/ic_tinyml/base_v1/weights/best.pt
    python src/04_export_models.py --weights ... --imgsz 416 --opset 12

Gera ao lado do .pt:
    best.onnx          modelo para o Raspberry Pi / Tinker Board
    export_info.json   classes, tamanho de entrada e formato da saida

Dependencias: pip install onnx onnxslim onnxruntime
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    sys.exit("Faltam dependencias. Rode: pip install onnx onnxslim onnxruntime")

import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Exporta best.pt para ONNX")
    p.add_argument("--weights", required=True, help="Caminho do best.pt")
    p.add_argument("--imgsz", type=int, default=416, help="Mesmo imgsz do treino")
    p.add_argument("--opset", type=int, default=12, help="12 e compativel com onnxruntime antigos")
    return p.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        sys.exit(f"ERRO: pesos nao encontrados: {weights}")

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    print(f"Classes ({len(names)}): {names}")

    # Exporta na CPU de proposito: evita a convolucao na GPU (MIOpen/ROCm) que da segfault.
    onnx_path = Path(model.export(
        format="onnx", imgsz=args.imgsz, simplify=True, opset=args.opset,
        dynamic=False, half=False, device="cpu",
    )).resolve()

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]

    n_anchors = sum((args.imgsz // s) ** 2 for s in (8, 16, 32))
    expected = [1, 4 + len(names), n_anchors]
    print(f"\nEntrada : {inp.name} {inp.shape} ({inp.type})")
    print(f"Saida   : {out.name} {out.shape}  (esperado {expected})")
    if list(out.shape) != expected:
        print("  ! AVISO: formato de saida diferente do esperado - o pos-processamento do Pi precisa se adaptar")

    # Paridade numerica: PyTorch x ONNX com a mesma entrada
    x = np.random.RandomState(0).rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    y_onnx = sess.run(None, {inp.name: x})[0]
    with torch.no_grad():
        y_pt = model.model.float().eval().cpu()(torch.from_numpy(x))
    y_pt = (y_pt[0] if isinstance(y_pt, (list, tuple)) else y_pt).numpy()
    diff = float(np.abs(y_onnx - y_pt).max())
    print(f"\nDiferenca maxima PyTorch x ONNX: {diff:.5f}", "(ok)" if diff < 0.05 else "  ! AVISO: alta")

    # Tempo na CPU do PC (so referencia: o Pi sera bem mais lento)
    for _ in range(5):
        sess.run(None, {inp.name: x})
    t0 = time.perf_counter()
    for _ in range(30):
        sess.run(None, {inp.name: x})
    ms = (time.perf_counter() - t0) / 30 * 1000
    print(f"Inferencia onnxruntime na CPU do PC: {ms:.1f} ms/imagem (nao representa o Pi)")

    size_mb = onnx_path.stat().st_size / 1e6
    info = {
        "weights": str(weights),
        "onnx": onnx_path.name,
        "imgsz": args.imgsz,
        "opset": args.opset,
        "names": names,
        "input": {"name": inp.name, "shape": inp.shape},
        "output": {"name": out.name, "shape": out.shape},
        "size_mb": round(size_mb, 2),
        "preprocess": "letterbox para imgsz x imgsz (padding 114), BGR->RGB, /255, NCHW float32",
    }
    info_path = onnx_path.with_name("export_info.json")
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))

    print(f"\nModelo : {onnx_path} ({size_mb:.1f} MB)")
    print(f"Info   : {info_path}")
    print(f"\nPara enviar ao Pi:\n  scp {onnx_path} {info_path} pi@IP_DO_PI:/home/pi/ic_tinyml/")


if __name__ == "__main__":
    main()