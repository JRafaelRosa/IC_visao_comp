#!/usr/bin/env python3
"""
06_infer_video.py - Inferencia ONNX em video/camera com cronometros e telemetria.
Roda igual no PC, no Raspberry Pi 5 e na Tinker Board (so CPU, via onnxruntime).

Uso:
    python src/06_infer_video.py --model runs/ic_tinyml/base_v1/weights/best.onnx --source video/mesa.mp4
    python src/06_infer_video.py --model best.onnx --source video/mesa.mp4 --no-video   # mede FPS sem gravar
    python src/06_infer_video.py --model best.onnx --source http://IP:81/stream          # ESP32-CAM (futuro)
    python src/06_infer_video.py --model best.onnx --source 0 --show                     # webcam

Saidas (em --out-dir): annotated.mp4, frames.csv, summary.json
Dependencias: opencv-python, numpy, onnxruntime (psutil opcional)
"""
import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("Falta onnxruntime. Rode: pip install onnxruntime")
try:
    import psutil
except ImportError:
    psutil = None

PALETTE = [(60, 76, 231), (113, 204, 46), (219, 152, 52), (15, 196, 241), (182, 89, 155), (156, 188, 26)]


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia ONNX em video com telemetria")
    p.add_argument("--model", required=True, help="Caminho do .onnx")
    p.add_argument("--source", required=True, help="Arquivo de video, indice de webcam (0) ou URL de stream")
    p.add_argument("--names", default=None, help="Classes separadas por virgula (se nao houver export_info.json)")
    p.add_argument("--imgsz", type=int, default=416, help="Usado so se a entrada do modelo for dinamica")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4, help="Threads do onnxruntime")
    p.add_argument("--warmup", type=int, default=5, help="Frames iniciais fora das estatisticas")
    p.add_argument("--max-frames", type=int, default=0, help="0 = ate o fim")
    p.add_argument("--telemetry-every", type=int, default=30)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-video", action="store_true", help="Nao grava o video anotado (mede sem custo de escrita)")
    p.add_argument("--show", action="store_true", help="Mostra janela (q para sair)")
    return p.parse_args()


def cpu_temp():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return None


def ram_mb():
    if psutil:
        return psutil.Process().memory_info().rss / 1e6
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # pico, em MB (Linux)


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return img, r, left, top


def preprocess(frame, size):
    img, r, left, top = letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True, crop=False)  # NCHW, RGB, 0-1
    return blob, r, left, top


def postprocess(out, r, left, top, shape, conf_thr, iou_thr):
    """out: (1, 4+nc, N) com caixas cx,cy,w,h no espaco do letterbox. Retorna lista (x1,y1,x2,y2,conf,cls)."""
    preds = out[0].T
    scores = preds[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    keep = conf >= conf_thr
    if not keep.any():
        return []
    p, cls, conf = preds[keep], cls[keep], conf[keep]
    H, W = shape[:2]
    x1 = np.clip((p[:, 0] - p[:, 2] / 2 - left) / r, 0, W - 1)
    y1 = np.clip((p[:, 1] - p[:, 3] / 2 - top) / r, 0, H - 1)
    x2 = np.clip((p[:, 0] + p[:, 2] / 2 - left) / r, 0, W - 1)
    y2 = np.clip((p[:, 1] + p[:, 3] / 2 - top) / r, 0, H - 1)
    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    off = cls[:, None].astype(np.float32) * 10000.0  # deslocamento por classe => NMS por classe
    shifted = xywh + np.concatenate([off, off, np.zeros_like(off), np.zeros_like(off)], axis=1)
    idx = np.array(cv2.dnn.NMSBoxes(shifted.tolist(), conf.tolist(), conf_thr, iou_thr)).flatten()
    return [(x1[i], y1[i], x2[i], y2[i], float(conf[i]), int(cls[i])) for i in idx]


def draw(frame, dets, names, inf_ms):
    for x1, y1, x2, y2, c, k in dets:
        col = PALETTE[k % len(PALETTE)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{names[k]} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), col, -1)
        cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"inf {inf_ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def stat(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95))}


def main():
    a = parse_args()
    model_path = Path(a.model).expanduser().resolve()
    if not model_path.is_file():
        sys.exit(f"ERRO: modelo nao encontrado: {model_path}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    inp, out_meta = sess.get_inputs()[0], sess.get_outputs()[0]
    imgsz = inp.shape[2] if isinstance(inp.shape[2], int) else a.imgsz
    nc = out_meta.shape[1] - 4 if isinstance(out_meta.shape[1], int) else None

    info = model_path.with_name("export_info.json")
    if a.names:
        names = [n.strip() for n in a.names.split(",")]
    elif info.is_file():
        names = json.loads(info.read_text())["names"]
    else:
        names = [f"classe_{i}" for i in range(nc or 1)]
    if nc is not None and len(names) != nc:
        sys.exit(f"ERRO: o modelo tem {nc} classes mas foram informados {len(names)} nomes")
    nc = len(names)

    src = a.source
    is_cam = src.isdigit()
    cap = cv2.VideoCapture(int(src) if is_cam else src)
    if not cap.isOpened():
        sys.exit(f"ERRO: nao consegui abrir a fonte: {src}")
    label = "cam" if (is_cam or "://" in src) else Path(src).stem
    out_dir = Path(a.out_dir or f"results/infer_{label}_{model_path.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fps_src = cap.get(cv2.CAP_PROP_FPS)
    fps_src = fps_src if 1 <= fps_src <= 120 else 30.0
    writer = None

    print(f"Modelo: {model_path.name} | entrada {imgsz}x{imgsz} | classes {names} | threads {a.threads}")
    print(f"Fonte : {src} | conf {a.conf} | iou {a.iou} | saida: {out_dir}\n")

    rows, telemetry = [], []
    cap_t, pre_t, inf_t, post_t, draw_t = [], [], [], [], []
    frames_with = np.zeros(nc)
    conf_sum = np.zeros(nc)
    det_total = np.zeros(nc, dtype=int)
    t_start = time.perf_counter()
    i = 0
    max_temp, peak_ram = None, 0.0

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        blob, r, left, top = preprocess(frame, imgsz)
        t2 = time.perf_counter()
        out = sess.run(None, {inp.name: blob})[0]
        t3 = time.perf_counter()
        dets = postprocess(out, r, left, top, frame.shape, a.conf, a.iou)
        t4 = time.perf_counter()

        if not a.no_video or a.show:
            draw(frame, dets, names, (t3 - t2) * 1000)
        if not a.no_video:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))
            writer.write(frame)
        if a.show:
            cv2.imshow("06_infer_video (q = sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        t5 = time.perf_counter()

        per_cls_n, per_cls_conf = np.zeros(nc, dtype=int), np.zeros(nc)
        for *_, c, k in dets:
            per_cls_n[k] += 1
            per_cls_conf[k] = max(per_cls_conf[k], c)
            conf_sum[k] += c
        det_total += per_cls_n
        frames_with += per_cls_n > 0

        ms = [(t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000, (t5 - t4) * 1000]
        row = {"frame": i, "cap_ms": ms[0], "pre_ms": ms[1], "inf_ms": ms[2], "post_ms": ms[3], "draw_ms": ms[4]}
        for k, n in enumerate(names):
            row[f"{n}_n"], row[f"{n}_conf"] = int(per_cls_n[k]), float(per_cls_conf[k])
        rows.append(row)
        if i >= a.warmup:
            for lst, v in zip((cap_t, pre_t, inf_t, post_t, draw_t), ms):
                lst.append(v)

        i += 1
        if i % a.telemetry_every == 0:
            recent = [r_["pre_ms"] + r_["inf_ms"] + r_["post_ms"] for r_ in rows[-a.telemetry_every:]]
            fps = 1000.0 / (sum(recent) / len(recent))
            temp, ram = cpu_temp(), ram_mb()
            peak_ram = max(peak_ram, ram)
            if temp is not None:
                max_temp = temp if max_temp is None else max(max_temp, temp)
            telemetry.append({"frame": i, "fps_pipeline": round(fps, 2), "temp_c": temp, "ram_mb": round(ram, 1)})
            t_txt = f"{temp:.1f} C" if temp is not None else "n/d"
            print(f"[frame {i:5d}] {fps:6.1f} FPS (pre+inf+pos) | temp {t_txt} | RAM {ram:.0f} MB")
        if a.max_frames and i >= a.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if a.show:
        cv2.destroyAllWindows()
    wall = time.perf_counter() - t_start
    if i == 0:
        sys.exit("ERRO: nenhum frame lido da fonte")

    pipe = [p + q + s for p, q, s in zip(pre_t, inf_t, post_t)]
    total = [c + p for c, p in zip(cap_t, pipe)] if pipe else []
    summary = {
        "model": model_path.name, "source": src, "imgsz": imgsz, "threads": a.threads,
        "conf": a.conf, "iou": a.iou, "frames": i, "warmup_frames": min(a.warmup, i),
        "wall_s": round(wall, 2),
        "ms": {"captura": stat(cap_t), "pre": stat(pre_t), "inferencia": stat(inf_t),
               "pos_nms": stat(post_t), "desenho_escrita": stat(draw_t)},
        "fps_pipeline": round(1000.0 / stat(pipe)["mean"], 2) if pipe else 0.0,
        "fps_com_captura": round(1000.0 / stat(total)["mean"], 2) if total else 0.0,
        "temp_max_c": max_temp, "ram_pico_mb": round(max(peak_ram, ram_mb()), 1),
        "classes": {n: {"deteccoes": int(det_total[k]),
                        "frames_com_a_classe_pct": round(100.0 * frames_with[k] / i, 1),
                        "conf_media": round(float(conf_sum[k] / det_total[k]), 3) if det_total[k] else 0.0}
                    for k, n in enumerate(names)},
        "telemetry": telemetry,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(out_dir / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    m = summary["ms"]
    print("\n=== Resumo (pronto para copiar) ===")
    print(f"Frames: {i} (fora das estatisticas: {summary['warmup_frames']}) | tempo total: {wall:.1f} s")
    print("| Etapa | media ms | p50 ms | p95 ms |\n|---|---|---|---|")
    for k, v in m.items():
        print(f"| {k} | {v['mean']:.2f} | {v['p50']:.2f} | {v['p95']:.2f} |")
    print(f"\nFPS (pre+inferencia+pos): {summary['fps_pipeline']} | FPS com captura: {summary['fps_com_captura']}")
    t_txt = f"{max_temp:.1f} C" if max_temp is not None else "n/d (sem sensor em /sys/class/thermal)"
    print(f"Temperatura maxima: {t_txt} | RAM: {summary['ram_pico_mb']} MB")
    print("\nPor classe (frames em que apareceu | conf media):")
    for n, v in summary["classes"].items():
        print(f"  {n:<16} {v['frames_com_a_classe_pct']:5.1f}% | {v['conf_media']:.2f} | {v['deteccoes']} deteccoes")
    print(f"\nSalvo em: {out_dir}")


if __name__ == "__main__":
    main()