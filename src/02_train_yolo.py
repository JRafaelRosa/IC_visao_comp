#!/usr/bin/env python3
"""
02_train_yolo.py - Treino base do YOLO na RX 6600 (ROCm, Linux).

Uso:
    python 02_train_yolo.py --data /caminho/data.yaml
    python 02_train_yolo.py --data data.yaml --epochs 100 --name exp_v2
    python 02_train_yolo.py --data data.yaml --no-amp   # se o AMP der problema

Saida previsivel: runs/ic_tinyml/<name>/weights/best.pt
"""

# 1) Variaveis de ambiente DEVEM ser definidas antes de importar o torch.
#    A RX 6600 (RDNA2, gfx1032) nao e oficialmente suportada pelo ROCm;
#    o override faz o runtime tratar a placa como gfx1030 (10.3.0).
import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Treino YOLO (ROCm)")
    p.add_argument("--data", required=True, help="Caminho do data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Pesos iniciais (yolov8n.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=6, help="Threads do dataloader")
    p.add_argument("--patience", type=int, default=20, help="Early stopping (epocas)")
    p.add_argument("--project", default="runs/ic_tinyml")
    p.add_argument("--name", default="yolov8n_416")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0", help="'0' (GPU) ou 'cpu'")
    p.add_argument("--no-miopen", action="store_true",
                   help="Desliga o MIOpen (use se a convolucao der segfault no ROCm)")
    p.add_argument("--no-amp", action="store_true", help="Desliga precisao mista")
    p.add_argument("--resume", action="store_true", help="Retoma o ultimo treino do --name")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_gpu() -> int:
    """No ROCm, a GPU aparece como 'cuda' no PyTorch."""
    print(f"PyTorch: {torch.__version__} | HIP: {torch.version.hip}")
    if not torch.cuda.is_available():
        print("ERRO: GPU nao detectada pelo PyTorch.")
        print("  - Confirme que instalou o torch com build ROCm (nao CPU/CUDA).")
        print("  - Confirme que seu usuario esta nos grupos 'render' e 'videos'.")
        print("  - Teste: rocminfo | grep gfx")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0


def main():
    args = parse_args()

    data_path = Path(args.data).expanduser()
    if data_path.is_file():
        data_path = data_path.resolve()
    elif data_path.suffix != ".yaml" or data_path.parent != Path("."):
        sys.exit(f"ERRO: data.yaml nao encontrado: {data_path}")
    # Nome solto como 'coco8.yaml' e um config embutido do Ultralytics:
    # ele baixa o mini dataset sozinho (bom para teste de fumaca).

    if args.no_miopen:
        # No ROCm, desligar o 'cudnn' desliga o MIOpen: o PyTorch usa convolucoes nativas
        torch.backends.cudnn.enabled = False
        print("MIOpen desligado (convolucoes nativas do PyTorch)")

    device = "cpu" if args.device == "cpu" else int(args.device)
    if device != "cpu":
        check_gpu()
    set_seed(args.seed)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=device,
        project=str(Path(args.project).resolve()),  # absoluto: evita o prefixo runs/detect/
        name=args.name,
        exist_ok=True,        # caminho fixo, sem train2/train3...
        seed=args.seed,
        deterministic=True,
        amp=not args.no_amp,
        resume=args.resume,
        plots=True,
        verbose=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    # Registra a configuracao usada, para reproducao pelo orientador/colegas
    cfg = vars(args) | {
        "data_resolved": str(data_path),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": sys.version.split()[0],
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" else "cpu",
    }
    (save_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))

    print("\n=== Treino concluido ===")
    print(f"Melhor modelo : {best}")
    print(f"Config salva  : {save_dir / 'train_config.json'}")
    if best.is_file():
        metrics = YOLO(str(best)).val(
            data=str(data_path), imgsz=args.imgsz, device=device,
            project=str(save_dir), name="val_best", exist_ok=True,
        )
        print(f"mAP@50    : {metrics.box.map50:.4f}")
        print(f"mAP@50-95 : {metrics.box.map:.4f}")
        print(f"Precisao  : {metrics.box.mp:.4f}")
        print(f"Recall    : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()