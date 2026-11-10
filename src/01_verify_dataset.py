#!/usr/bin/env python3
"""
01_verify_dataset.py - Verifica um dataset YOLO (export do Roboflow) e gera data_fixed.yaml.

Checa:
  - imagens legiveis e pares imagem/label
  - formato dos labels: 5 colunas, classe valida, valores em [0,1], largura/altura > 0
  - caixas minusculas (a imgsz de treino), imagens sem objeto (fundo)
  - contagem de caixas por classe em cada split
  - VAZAMENTO entre splits: mesma foto de origem (nome antes de '.rf.') e
    imagens quase identicas (dHash) em splits diferentes

Uso:
    python src/01_verify_dataset.py --root dataset/ic_object
    python src/01_verify_dataset.py --root dataset/ic_object --imgsz 416
"""

import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = [("train", "train"), ("valid", "val"), ("test", "test")]  # (pasta, chave no yaml)


def source_key(stem: str) -> str:
    """Roboflow nomeia copias como '<original>.rf.<hash>'; o prefixo identifica a foto de origem."""
    return stem.split(".rf.")[0]


def dhash(gray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


def load_names(cfg: dict) -> list:
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        sys.exit("ERRO: data.yaml sem 'names'.")
    return list(names)


def check_split(root: Path, split: str, names: list, imgsz: int) -> dict:
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    out = {"n_img": 0, "errors": [], "warnings": [], "boxes": Counter(),
           "empty": 0, "tiny": 0, "keys": defaultdict(list), "hashes": []}
    if not img_dir.is_dir():
        out["warnings"].append(f"pasta ausente: {img_dir}")
        return out

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    out["n_img"] = len(imgs)
    stems = {p.stem for p in imgs}

    for p in imgs:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            out["errors"].append(f"imagem ilegivel: {p.name}")
            continue
        out["hashes"].append((dhash(gray), p.name))
        out["keys"][source_key(p.stem)].append(p.name)

        lbl = lbl_dir / f"{p.stem}.txt"
        if not lbl.is_file():
            out["warnings"].append(f"sem arquivo de label (tratada como fundo): {p.name}")
            out["empty"] += 1
            continue

        rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not rows:
            out["empty"] += 1
        for i, tok in enumerate(rows, 1):
            where = f"{lbl.name}:{i}"
            if len(tok) != 5:
                kind = "poligono (segmentacao?)" if len(tok) > 5 else "colunas faltando"
                out["errors"].append(f"{where} tem {len(tok)} colunas - {kind}")
                continue
            try:
                cls = int(tok[0])
                x, y, w, h = map(float, tok[1:])
            except ValueError:
                out["errors"].append(f"{where} valor nao numerico: {' '.join(tok)}")
                continue
            if not 0 <= cls < len(names):
                out["errors"].append(f"{where} classe {cls} fora de 0..{len(names) - 1}")
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                out["errors"].append(f"{where} valores fora de [0,1]: {' '.join(tok)}")
                continue
            if w <= 0 or h <= 0:
                out["errors"].append(f"{where} largura/altura zero")
                continue
            out["boxes"][cls] += 1
            if min(w, h) * imgsz < 8:
                out["tiny"] += 1

    for lbl in lbl_dir.glob("*.txt") if lbl_dir.is_dir() else []:
        if lbl.stem not in stems:
            out["errors"].append(f"label sem imagem: {lbl.name}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Verifica dataset YOLO (Roboflow)")
    ap.add_argument("--root", required=True, help="Pasta com data.yaml, train/, valid/, test/")
    ap.add_argument("--imgsz", type=int, default=416, help="Resolucao de treino (p/ caixas minusculas)")
    ap.add_argument("--hash-dist", type=int, default=4, help="Distancia dHash p/ quase-duplicata")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        sys.exit(f"ERRO: {yaml_path} nao encontrado")
    cfg = yaml.safe_load(yaml_path.read_text())
    names = load_names(cfg)
    print(f"Dataset: {root}\nClasses ({len(names)}): {names}\n")

    res = {s: check_split(root, s, names, args.imgsz) for s, _ in SPLITS}
    n_err = 0

    print(f"{'split':<7}{'imgs':>6}{'fundo':>7}{'tiny':>6}  caixas por classe")
    for s, _ in SPLITS:
        r = res[s]
        per_cls = ", ".join(f"{names[c]}={r['boxes'][c]}" for c in range(len(names)))
        print(f"{s:<7}{r['n_img']:>6}{r['empty']:>7}{r['tiny']:>6}  {per_cls}")
        n_err += len(r["errors"])
        if r["n_img"] and any(r["boxes"][c] == 0 for c in range(len(names))):
            print(f"  ! AVISO: {s} nao tem nenhuma caixa de alguma classe - a metrica dela nao existe aqui")

    print("\n=== Vazamento entre splits ===")
    leaks = 0
    for a, b in combinations([s for s, _ in SPLITS], 2):
        shared = set(res[a]["keys"]) & set(res[b]["keys"])
        if shared:
            leaks += len(shared)
            ex = ", ".join(sorted(shared)[:3])
            print(f"  ! {a} x {b}: {len(shared)} foto(s) de origem em comum (ex.: {ex})")
        near = [(na, nb) for ha, na in res[a]["hashes"] for hb, nb in res[b]["hashes"]
                if bin(ha ^ hb).count("1") <= args.hash_dist]
        if near:
            leaks += len(near)
            print(f"  ! {a} x {b}: {len(near)} par(es) quase identico(s) (ex.: {near[0][0]} ~ {near[0][1]})")
    if not leaks:
        print("  nenhum vazamento detectado (por nome de origem e por dHash)")
    else:
        print("  -> mAP de validacao/teste fica inflado. Separe por sessao de captura.")

    print("\n=== Problemas ===")
    for s, _ in SPLITS:
        for msg in res[s]["errors"][:15]:
            print(f"  ERRO [{s}] {msg}")
        extra = len(res[s]["errors"]) - 15
        if extra > 0:
            print(f"  ... mais {extra} erro(s) em {s}")
    warns = sum(len(res[s]["warnings"]) for s, _ in SPLITS)
    if not n_err:
        print("  nenhum erro de formato")
    if warns:
        print(f"  {warns} aviso(s) (ex.: imagens sem arquivo de label)")

    # data.yaml com caminhos absolutos (o do Roboflow usa caminhos relativos que quebram)
    fixed = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    for folder, key in SPLITS:
        if (root / folder / "images").is_dir():
            fixed[key] = f"{folder}/images"
    out_yaml = root / "data_fixed.yaml"
    out_yaml.write_text(yaml.safe_dump(fixed, sort_keys=False, allow_unicode=True))
    print(f"\nGerado: {out_yaml}")

    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()