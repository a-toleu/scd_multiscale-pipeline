#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, os, sys, json, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.io import wavfile

try:
    from scd_pipeline import (
        SCDConfig, SCDPipline
    )
except ImportError:
    print("[ERROR] 未找到 scd_pipeline.py。请确保它在同一目录中。", file=sys.stderr)
    sys.exit(1)

# ----------------------
# 工具函数
# ----------------------

def to_mono_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    else:
        x = x.astype(np.float32)
    return x


def save_plot(png_path: Path, x: np.ndarray, sr: int, bounds_sec: np.ndarray, title: str = None):
    import matplotlib.pyplot as plt
    t = np.arange(len(x)) / float(sr)
    plt.figure(figsize=(10, 2.3))
    plt.plot(t, x, linewidth=0.6)
    for b in bounds_sec:
        plt.axvline(b, linestyle='--', linewidth=0.8)
    if title:
        plt.title(title)
    plt.xlabel('Time (s)'); plt.ylabel('amp')
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()


def derive_file_id(wav_path: Path, file_id_mode: str) -> str:
    if file_id_mode == "stem":
        return wav_path.stem
    if file_id_mode == "before_dot":
        filename = wav_path.name
        file_stem = filename.rsplit('.', 1)[0]
        file_id = file_stem.split('.')[0]
        return file_id
    if file_id_mode == "parent_stem":
        return wav_path.parent.name
    return wav_path.stem

import numpy as np

def add_white_noise_level(x: np.ndarray, noise_db: float, rng: np.random.RandomState = None) -> np.ndarray:
    """
    在波形 x 上按给定“噪声等级” noise_db 加白噪声，
    这里 noise_db 是 Noise-to-Signal Ratio (NSR) 的 dB 值：
        noise_db = 10 * log10(P_noise / P_signal)
    因此 noise_db 越大，噪声越大。

    例如：
      noise_db = -10  -> 噪声功率是信号的 0.1
      noise_db = 0    -> 噪声功率等于信号功率
      noise_db = +10  -> 噪声功率是信号的 10 倍
    """
    if rng is None:
        rng = np.random.RandomState()

    x = x.astype(np.float32)
    sig_power = np.mean(x ** 2)
    if sig_power == 0.0:
        return x.copy()

    # 根据 NSR_dB 计算噪声功率: P_noise = P_signal * 10^(noise_db/10)
    nsr_linear = 10 ** (noise_db / 10.0)
    noise_power = sig_power * nsr_linear

    # 生成单位功率白噪声，并缩放到目标噪声功率
    noise = rng.randn(*x.shape).astype(np.float32)
    noise = noise / np.sqrt(np.mean(noise ** 2) + 1e-12)  # 功率 ≈ 1
    noise = noise * np.sqrt(noise_power)

    y = x + noise
    y = np.clip(y, -1.0, 1.0)
    return y


# ----------------------
# 主流程
# ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Batch SCD pipline on CSV with audio_path column",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # I/O
    g_io = ap.add_argument_group("I/O")
    g_io.add_argument("--ref-csv", type=str, required=True, help="CSV with at least 'audio_path' column")
    g_io.add_argument("--outdir", type=str, default="batch_out", help="Output directory for per-file results")
    g_io.add_argument("--plot", action="store_true", help="Save basic waveform+boundaries PNG")
    g_io.add_argument("--file-id-mode", type=str, default="before_dot",
                      choices=["before_dot", "stem", "parent_stem"],
                      help="How to derive file_id from path (use 'before_dot' for AMI)")

    # 管线控制（对应 integrated 版本）
    g_pipe = ap.add_argument_group("Pipeline Control")
    g_pipe.add_argument("--backend", type=str, default="cluster_hysteresis",
                        choices=["multiscale_only", "viterbi_only", "cluster_hysteresis"],
                        help="选择后端 (默认: cluster_hysteresis)")
   
    # 算法参数（J‑Only 版）
    g_alg = ap.add_argument_group("Algorithm parameters")
    g_alg.add_argument("--embed", type=str, default="logmel",
                   choices=["logmel","mfcc","ecapa","xvector","wavlm","wav2vec2"],
                   help="Embedding backend.")
    #g_alg.add_argument("--method", type=str, default="agglom", choices=["agglom","spectral","dbscan"], help="Clustering backend.")                       help="简化示例里只留 logmel（若你已替换为 ECAPA 等，可在 scd_pipeline 中扩展）")
    g_alg.add_argument("--block-win", type=float, default=0.8)
    g_alg.add_argument("--block-hop", type=float, default=0.4)
    g_alg.add_argument("--nfft", type=int, default=1024)
    g_alg.add_argument("--hop", type=int, default=256)
    g_alg.add_argument("--nmels", type=int, default=80)

    g_alg.add_argument("--peak-quantile", type=float, default=0.75)
    g_alg.add_argument("--peak-min-dist", type=float, default=0.60)
    g_alg.add_argument("--min-seg", type=float, default=1.20)
    g_alg.add_argument("--hyst-high", type=float, default=0.60)
    g_alg.add_argument("--hyst-low", type=float, default=0.40)
    g_alg.add_argument("--agglom-th", type=float, default=0.60)
    g_alg.add_argument("--no-flux-align", action="store_true", help="禁用 flux 对齐（仅 cluster_hysteresis 有效）")
    
    g_alg.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.4, 0.8, 1.6],
        help="Multi-scale detector 的窗口长度（秒），例如: --scales 0.4 0.8 1.6",
    )
    
    # 其它
    g_misc = ap.add_argument_group("Misc")
    g_misc.add_argument("--gpu-id", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES")
    g_misc.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试）")

    args = ap.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(args.ref_csv)

    if "audio_path" not in df.columns:
        raise ValueError("ref-csv must contain column 'audio_path'")
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"[INFO] Limited to first {args.limit} files")

    # 构建配置（与 integrated 管线参数一致）
    cfg = SCDConfig(
        embed=args.embed,
        block_win_s=args.block_win,
        block_hop_s=args.block_hop,
        n_fft=args.nfft,
        stft_hop=args.hop,
        n_mels=args.nmels,
        peak_quantile=args.peak_quantile,
        min_peak_distance_s=args.peak_min_dist,
        backend=args.backend,
        min_segment_s=args.min_seg,
        hysteresis_high=args.hyst_high,
        hysteresis_low=args.hyst_low,
        agglom_distance_threshold=args.agglom_th,
        use_flux_align=(not args.no_flux_align),
        scales = args.scales,
    )

    # 打印配置
    print(f"[CONFIG] Backend: {cfg.backend}")
    print(f"[CONFIG] Peak Quantile: {cfg.peak_quantile}")
    print(f"[CONFIG] MinSeg: {cfg.min_segment_s}")
    print(f"[CONFIG] AgglomTH: {cfg.agglom_distance_threshold}")
    print(f"[CONFIG] Embed: {cfg.embed}")
    print(f"[CONFIG] Scale: {cfg.scales}")

    print()

    pred_rows: List[Dict[str, float]] = []
    timing_rows: List[Dict[str, float]] = []
    errors: List[str] = []

    # 进度显示
    try:
        from tqdm import tqdm
        iterator = tqdm(df.itertuples(index=False), total=len(df), desc=f"Processing {outdir.name}")
    except Exception:
        iterator = df.itertuples(index=False)
        print(f"[INFO] Processing {len(df)} files for {outdir.name}...")

    scd = SCDPipline(cfg)

    for row in iterator:
        audio_path_str = getattr(row, "audio_path")
        if not audio_path_str or pd.isna(audio_path_str):
            continue
        wav_path = Path(str(audio_path_str))
        file_id = derive_file_id(wav_path, args.file_id_mode)

        try:
            if not wav_path.exists():
                raise FileNotFoundError(f"Missing: {wav_path}")
            sr, wav = wavfile.read(wav_path.as_posix())
            x = to_mono_float32(wav)
            x_noisy = add_white_noise_level(x, noise_db=10)  # 10 dB 噪声
            #print("!!!!!!!!!!!!!+++++++++++ NOIZY")
            duration_sec = len(x_noisy) / float(sr) if sr > 0 else 0.0
            t0 = time.time()
            res = scd.run(x_noisy, sr)
            proc_sec = time.time() - t0
            rtf = (proc_sec / duration_sec) if duration_sec > 0 else 0.0

            bounds_sec = np.asarray(res.get('bounds_sec', []), dtype=float)
            confs = res.get('scores', None)

            # 写单文件 CSV
            out_csv = outdir / f"{file_id}.scd.csv"
            with out_csv.open('w', encoding='utf-8') as f:
                if confs is not None and confs.size > 0:
                    # 由于 scores 是帧/块级的分数，不与边界一一对应，这里只存边界秒数。
                    f.write("boundary_seconds\n")
                else:
                    f.write("boundary_seconds\n")
                for t in bounds_sec:
                    f.write(f"{float(t):.3f}\n")

            # 聚合预测
            for t in bounds_sec:
                pred_rows.append({"file_id": file_id, "boundary_seconds": float(t)})

            # 可选绘图
            if args.plot:
                png_path = outdir / f"{file_id}.png"
                save_plot(png_path, x, sr, bounds_sec, title=wav_path.name)

            timing_rows.append({
                "file_id": file_id,
                "audio_duration_sec": float(duration_sec),
                "processing_time_sec": float(proc_sec),
                "real_time_factor": float(rtf),
            })

        except Exception as e:
            msg = f"[ERROR] {file_id}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            continue

    # 写聚合与计时
    outdir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame(pred_rows, columns=["file_id", "boundary_seconds"]).sort_values(["file_id", "boundary_seconds"]) if pred_rows else pd.DataFrame(columns=["file_id", "boundary_seconds"])
    out_pred = outdir / "pred_fileid_boundary.csv"
    pred_df.to_csv(out_pred, index=False)

    timing_df = pd.DataFrame(timing_rows, columns=["file_id", "audio_duration_sec", "processing_time_sec", "real_time_factor"])
    out_timing = outdir / "timing_report.csv"
    timing_df.to_csv(out_timing, index=False)

    total_audio = float(timing_df["audio_duration_sec"].sum()) if not timing_df.empty else 0.0
    total_proc = float(timing_df["processing_time_sec"].sum()) if not timing_df.empty else 0.0
    avg_rtf = (total_proc / total_audio) if total_audio > 0 else 0.0

    stats = {
        "total_files": int(len(df)),
        "success_files": int(len(timing_rows)),
        "failed_files": int(len(errors)),
        "total_audio_hours": total_audio / 3600.0,
        "total_processing_hours": total_proc / 3600.0,
        "average_rtf": float(avg_rtf),
    }
    with (outdir / "stats.json").open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    if errors:
        with (outdir / "errors.log").open('w', encoding='utf-8') as f:
            f.write("\n".join(errors))
        print(f"\n[WARN] {len(errors)} files failed. See {outdir / 'errors.log'}")

    print(f"\n[DONE] Batch processing complete for {outdir.name}")
    print(f"[OUTPUT] Prediction file: {out_pred}")
    print(f"[OUTPUT] Timing report: {out_timing}")
    print(f"[STATS] Total audio processed: {stats['total_audio_hours']:.2f} hours")
    print(f"[STATS] Total processing time: {stats['total_processing_hours']:.2f} hours")
    print(f"[STATS] Average Real-Time Factor (RTF): {stats['average_rtf']:.4f}")


if __name__ == "__main__":
    main()
