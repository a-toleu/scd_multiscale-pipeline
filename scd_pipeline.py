#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse, os, sys, warnings
import numpy as np
from scipy import signal

import argparse, os, sys, warnings
from scipy.io import wavfile
from pathlib import Path
from scipy.fftpack import dct  # for MFCC
import torch

########################
# Small utilities      #
########################
import numpy as np

# ---- torchaudio backend fallback (optional but recommended) ----
try:
    import torchaudio, sys
    backs = set(torchaudio.list_audio_backends())
    if "soundfile" in backs:
        torchaudio.set_audio_backend("soundfile")
    elif "sox_io" in backs:
        torchaudio.set_audio_backend("sox_io")
    else:
        print("[WARN] No torchaudio backend available. Install libsndfile or sox.", file=sys.stderr)
except Exception as e:
    print(f"[WARN] torchaudio backend setup failed: {e}", file=sys.stderr)
# ----------------------------------------------------------------


def _rng(a):
    """safe range (ptp) for numpy 1.x/2.x"""
    a = np.asarray(a)
    return float(np.ptp(a)) if a.size else 0.0


def to_mono_float32(wav: np.ndarray) -> np.ndarray:
    x = wav
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    else:
        x = x.astype(np.float32)
    return np.clip(x, -1.0, 1.0)

def stft_mag(x: np.ndarray, n_fft: int, hop_length: int, win_length: Optional[int] = None) -> np.ndarray:
    win_length = win_length or n_fft
    f, t, Z = signal.stft(x, nperseg=win_length, noverlap=win_length-hop_length,
                          nfft=n_fft, window='hann', boundary=None, padded=False)
    return np.abs(Z)

def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 50.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or (sr/2)
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f/700.0)
    def mel_to_hz(m): return 700.0 * (10.0**(m/2595.0) - 1.0)
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    f_points = mel_to_hz(m_points)
    bins = np.floor((n_fft + 1) * f_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2 + 1), dtype=np.float32)
    for m in range(1, n_mels+1):
        f0, f1, f2 = bins[m-1], bins[m], bins[m+1]
        if f1 <= f0: f1 = f0 + 1
        if f2 <= f1: f2 = f1 + 1
        for k in range(f0, min(f1, fb.shape[1])):
            fb[m-1, k] = (k - f0) / max(1, (f1 - f0))
        for k in range(f1, min(f2, fb.shape[1])):
            fb[m-1, k] = (f2 - k) / max(1, (f2 - f1))
    fb = fb / (np.maximum(fb.sum(axis=1, keepdims=True), 1e-8))
    return fb

def logmel_spectrogram(x: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 64, eps: float = 1e-6) -> np.ndarray:
    mag = stft_mag(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = np.dot(fb, mag[:fb.shape[1], :])
    return np.log(mel + eps)

def mfcc_from_logmel(logmel: np.ndarray, n_mfcc: int = 20, lifter: int = 22) -> np.ndarray:
    """log-mel (n_mels x T) -> MFCC (n_mfcc x T) via DCT-II with optional lifter."""
    mfcc = dct(logmel, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    if lifter and lifter > 0:
        n = np.arange(mfcc.shape[0])
        lift = 1 + (lifter / 2.0) * np.sin(np.pi * (n + 1) / lifter)
        mfcc = mfcc * lift[:, None]
    return mfcc

def spectral_flux(mag: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mag = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + eps)
    diff = np.diff(mag, axis=1)
    flux = np.sqrt((diff**2).sum(axis=0))
    flux = np.concatenate([[flux[0]], flux], axis=0)
    return flux

def sliding_blocks(x: np.ndarray, sr: int, win_s: float, hop_s: float) -> List[Tuple[int,int]]:
    win = int(win_s * sr); hop = int(hop_s * sr)
    n = len(x)
    starts = np.arange(0, max(1, n - win + 1), hop, dtype=int)
    return [(s, min(s + win, n)) for s in starts]

def aggregate(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    mean = arr.mean(axis=axis); std = arr.std(axis=axis)
    return np.concatenate([mean, std], axis=0)

def normalize_vecs(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms

def pick_peaks(x: np.ndarray, distance: int, thresh: float) -> np.ndarray:
    peaks, _ = signal.find_peaks(x, distance=distance, height=thresh)
    return peaks

def hysteresis_mask(scores: np.ndarray, high: float, low: float) -> np.ndarray:
    on = False; mask = np.zeros_like(scores, dtype=bool)
    for i, s in enumerate(scores):
        if not on and s >= high: on = True
        if on: mask[i] = True
        if on and s < low: on = False
    return mask

def temporal_smooth(labels: np.ndarray, min_len: int = 1) -> np.ndarray:
    """Merge runs < min_len into neighbors (mode filter over runs)."""
    if min_len <= 1 or labels.size == 0: return labels
    y = labels.copy()
    i = 0
    while i < len(y):
        j = i
        while j+1 < len(y) and y[j+1] == y[i]: j += 1
        run_len = j - i + 1
        if run_len < min_len:
            left = y[i-1] if i-1 >= 0 else None
            right = y[j+1] if j+1 < len(y) else None
            fill = right if right is not None else left
            if fill is not None:
                y[i:j+1] = fill
        i = j + 1
    return y

# Optional sklearn clustering
try:
    from sklearn.cluster import AgglomerativeClustering, DBSCAN, SpectralClustering
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False




def cluster_segments(X: np.ndarray, method: str = "agglom", **kwargs) -> np.ndarray:
    n = X.shape[0]
    if n <= 2: return np.arange(n)
    if not SKLEARN_OK:
        warnings.warn("scikit-learn not found; falling back to trivial labels")
        return (np.arange(n) % 2)
    if method == "agglom":
        n_clusters = kwargs.get("n_clusters", None)
        distance_threshold = kwargs.get("distance_threshold", 0.6)
        linkage = kwargs.get("linkage", "average")
        try:
            model = AgglomerativeClustering(n_clusters=None, affinity="cosine",
                                            linkage=linkage, distance_threshold=distance_threshold) if n_clusters is None \
                    else AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage=linkage)
        except TypeError:
            model = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                            linkage=linkage, distance_threshold=distance_threshold) if n_clusters is None \
                    else AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage=linkage)
        labels = model.fit_predict(X)
    elif method == "spectral":
        k = kwargs.get("n_clusters", 2)
        labels = SpectralClustering(n_clusters=k, affinity="nearest_neighbors").fit_predict(X)
    else:
        eps = kwargs.get("eps", 0.25); min_samples = kwargs.get("min_samples", 2)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)
        # Map noise -1 to its own unique indices to avoid all -1
        if np.any(labels < 0):
            noise_idx = np.where(labels < 0)[0]
            labels = labels.copy()
            for k, idx in enumerate(noise_idx):
                labels[idx] = labels.max() + 1 + k
    return labels

########################
# Embedding backends   #
########################

# Silence TF logs for transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES","0"))

def _torch_device():
    try:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception:
        return None

class Embedder:
    def __init__(self, kind: str = "logmel", sr: int = 16000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64):
        self.kind = kind.lower()
        self.device = _torch_device()
        self.sr = sr; self.n_fft = n_fft; self.hop = hop; self.n_mels = n_mels
        self._init_models()

    def _init_models(self):
        self.model = None
        if self.kind == "ecapa":
            try:
                from speechbrain.pretrained import EncoderClassifier

                dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                self.device = dev

                # 关键：在 from_hparams 就声明 device（别再只 .to）
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": str(self.device), "dtype": "float32"},
                )
                self.model.eval()
                # 保险：确保所有子模块是 float32
                for m in self.model.mods.modules():
                    try:
                        m.float()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] ECAPA not available ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"

         
        elif self.kind == "wavlm":
            try:
                from transformers import AutoFeatureExtractor, AutoModel
                self.model = AutoModel.from_pretrained("microsoft/wavlm-base")
                self.model.to(self.device); self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
            except Exception as e:
                print(f"[WARN] WavLM not available ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"
        elif self.kind == "xvector":
            # Placeholder: integrate your own extractor here
            print("[WARN] x-vector backend not wired; fallback to logmel.", file=sys.stderr)
            self.kind = "logmel"
        elif self.kind == "wav2vec2":
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                self.model = AutoModel.from_pretrained("facebook/wav2vec2-base")
                self.model.to(self.device)
                self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            except Exception as e:
                print(f"[WARN] wav2vec2 not available ({e}), fallback to logmel.", file=sys.stderr)
            self.kind = "logmel"

        elif self.kind == "wavlm":
            try:
                from transformers import AutoFeatureExtractor, AutoModel
                self.model = AutoModel.from_pretrained("microsoft/wavlm-base")
                self.model.to(self.device); self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")
            except Exception as e:
                print(f"[WARN] WavLM not available ({e}), fallback to logmel.", file=sys.stderr)
                self.kind = "logmel"
        elif self.kind == "xvector":
            # Placeholder: integrate your own extractor here
            print("[WARN] x-vector backend not wired; fallback to logmel.", file=sys.stderr)
            self.kind = "logmel"
        elif self.kind == "wav2vec2":
            try:
                from transformers import AutoModel, AutoFeatureExtractor
                self.model = AutoModel.from_pretrained("facebook/wav2vec2-base")
                self.model.to(self.device)
                self.model.eval()
                self.feat_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            except Exception as e:
                print(f"[WARN] wav2vec2 not available ({e}), fallback to logmel.", file=sys.stderr)
            self.kind = "logmel"

    def block_embed(self, x: np.ndarray, blocks: List[Tuple[int,int]]) -> np.ndarray:
        if self.kind == "logmel":
            logmel = logmel_spectrogram(x, self.sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels)
            frame_times = np.arange(logmel.shape[1]) * (self.hop / self.sr)
            vecs = []
            for (s, e) in blocks:
                t0, t1 = s/self.sr, e/self.sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (self.hop / self.sr))]).clip(0, logmel.shape[1]-1)
                feat = logmel[:, idx]
                vecs.append(aggregate(feat, axis=1))
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)

        elif self.kind == "ecapa":
            xs = []
            with torch.no_grad():
                for (s, e) in blocks:
                    seg = torch.from_numpy(x[s:e]).to(self.device, dtype=torch.float32)  # [T] -> float32 on device
                    seg = seg.unsqueeze(0)  # [1, T]
                    emb = self.model.encode_batch(seg)  # SB 内部会用同一 device 的 compute_features -> embedding_model
                    emb = emb.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                    xs.append(emb)
            X = np.vstack(xs).astype(np.float32)
            return normalize_vecs(X)


        elif self.kind == "wavlm":
            x_t = torch.from_numpy(x).float().to(self.device or "cpu")
            sr = self.sr
            chunk_sec = 20.0
            chunk = int(chunk_sec * sr)
            feats = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                 enabled=(self.device is not None and str(self.device).startswith("cuda"))):
                for i in range(0, len(x), chunk):
                    seg = x_t[i:i+chunk].unsqueeze(0)  # (1, Tchunk)
                    inputs = self.feat_extractor(seg.squeeze().cpu().numpy(),
                                                 sampling_rate=sr, return_tensors="pt", padding=False)
                    inputs = {k: v.to(self.device or "cpu") for k, v in inputs.items()}
                    out = self.model(**inputs).last_hidden_state.squeeze(0)  # (T,D) on device
                    feats.append(out.detach().cpu())
            H = torch.cat(feats, dim=0).numpy()
            T = H.shape[0]
            frame_times = np.linspace(0, len(x)/sr, T, endpoint=False)
            vecs = []
            for (s,e) in blocks:
                t0, t1 = s/sr, e/sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (len(x)/sr) * T)]).clip(0, T-1)
                feat = H[idx, :]
                vecs.append(np.concatenate([feat.mean(0), feat.std(0)], axis=0))
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)

        elif self.kind == "mfcc":
            # 先算 log-mel → MFCC（默认 20 维，含 C0）
            logmel = logmel_spectrogram(x, self.sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels)
            mfcc = mfcc_from_logmel(logmel, n_mfcc=20, lifter=22)
            frame_times = np.arange(mfcc.shape[1]) * (self.hop / self.sr)
            vecs = []
            for (s, e) in blocks:
                t0, t1 = s/self.sr, e/self.sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (self.hop / self.sr))]).clip(0, mfcc.shape[1]-1)
                feat = mfcc[:, idx]
                vecs.append(aggregate(feat, axis=1))  # mean+std
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)
        elif self.kind == "wav2vec2":
            x_t = torch.from_numpy(x).float().to(self.device)
            sr = self.sr
            chunk_sec = 20.0
            chunk = int(chunk_sec * sr)
            feats = []
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(self.device is not None and getattr(self.device, "type", "cpu")=="cuda")):
                for i in range(0, len(x), chunk):
                    seg = x_t[i:i+chunk].unsqueeze(0)
                    inputs = self.feat_extractor(seg.squeeze().cpu().numpy(), sampling_rate=sr, return_tensors="pt", padding=False)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    out = self.model(**inputs).last_hidden_state.squeeze(0)  # (T,D)
                    feats.append(out.detach().cpu())
            H = torch.cat(feats, dim=0).numpy() if feats else np.zeros((0, 768), dtype=np.float32)
            T = H.shape[0]
            if T == 0:
                return np.zeros((len(blocks), 2), dtype=np.float32)
            frame_times = np.linspace(0, len(x)/sr, T, endpoint=False)
            vecs = []
            for (s,e) in blocks:
                t0, t1 = s/sr, e/sr
                idx = np.where((frame_times >= t0) & (frame_times < t1))[0]
                if len(idx) == 0:
                    idx = np.array([int(t0 / (len(x)/sr) * T)]).clip(0, T-1)
                feat = H[idx, :]
                vecs.append(np.concatenate([feat.mean(0), feat.std(0)], axis=0))
            X = np.vstack(vecs).astype(np.float32)
            return normalize_vecs(X)
        else:
            raise ValueError(f"Unknown embed kind: {self.kind}")
########################
# Config               #
########################

@dataclass
class SCDConfig:
#     # Embedding / framing
#     embed: str = "logmel"
#     block_win_s: float = 0.8
#     block_hop_s: float = 0.4
#     n_fft: int = 1024
#     stft_hop: int = 256
#     n_mels: int = 80

#     # Seeding
#     scales: Tuple[float, ...] = (0.4, 0.8, 1.6)
#     peak_quantile: float = 0.9
#     min_peak_distance_s: float = 0.5
    
#     # Backend choice
#     backend: str = "viterbi_only"  # "multiscale_only" | "viterbi_only" | "cluster_hysteresis"

#     # Viterbi / hysteresis
#     use_viterbi: bool = True
#     min_segment_s: float = 1.2
#     hysteresis_high: float = 0.6
#     hysteresis_low: float = 0.4

#     # Clustering backend params
#     cluster_method: str = "agglom"
#     agglom_distance_threshold: float = 0.6
#     min_run_segments: int = 1

#     # Flux alignment
#     use_flux_align: bool = True
#     align_window_ms: float = 250.0
    
#     alpha_seed: float = 0.6
    
    #---------
    # 全局
    block_win_s: float = 0.5
    block_hop_s: float = 0.25
    n_fft: int = 1024
    stft_hop: int = 256
    min_segment_s: float = 1.2
    n_mels: int = 80
    embed: str = "logmel"

    # seed 相关
    alpha_seed: float = 0.5
    min_peak_distance_s: float = 0.8
    peak_quantile: float = 0.8

    # multiscale
    scales: Tuple[float, ...] = (0.4, 0.8, 1.6)
    ms_min_dist_s: float = 0.5
    ms_jump_percentile: float = 75.0
    ms_cluster_window_s: float = 0.2
    ms_vote_score_thresh: float = 0.5
    ms_conf_score_thresh: float = 0.7

    # cluster + hysteresis
    cluster_method: str = "agglom"
    agglom_distance_threshold: float = 0.5
    n_clusters: int = 8
    w_cluster: float = 0.5
    w_jump: float = .3
    hysteresis_high: float = 0.7
    hysteresis_low: float = 0.3
    use_flux_align: bool = False
    flux_align_window_s: float = 0.2

    # viterbi
    viterbi_switch_prob: float = 0.01
    viterbi_init_boundary_prob: float = 0.01
    viterbi_beta_boundary: float = 1.0
    viterbi_beta_inside: float = 1.0

    # 其他
    backend: str = "cluster_hysteresis"  # or viterbi_only / multiscale_only
    
    min_run_segments:int =1
    



########################
# Main SCD class       #
########################

class SCDPipline:
    def __init__(self, cfg: SCDConfig, sr_hint: int = 16000):
        self.cfg = cfg
        self.embedder = None
        self.sr_hint = sr_hint

    def _ensure_embedder(self, sr: int):
        if self.embedder is None or self.embedder.sr != sr or self.embedder.kind != self.cfg.embed.lower():
            self.embedder = Embedder(self.cfg.embed, sr, self.cfg.n_fft, self.cfg.stft_hop, self.cfg.n_mels)

     ########################################################
    # New: feature extraction for training (multi+pipeline) #
    ########################################################
    def extract_training_candidates(
        self,
        x: np.ndarray,
        sr: int,
        true_bounds_sec: np.ndarray,
        label_tolerance_s: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取用于训练的候选边界特征（沿用 single-scale + clustering + hysteresis 里
        的 J/C 逻辑 + multiscale fused peaks 逻辑）。

        返回:
          X: [n_candidates, n_features]  每个候选边界一个样本
          y: [n_candidates]             0/1 是否是 GT 边界 (|t - true| <= tol)
          centers: [n_candidates]       候选边界的时间 (秒)
        """
        self._ensure_embedder(sr)

        # ===== 1) 单尺度块 + embedding（和 run() 里完全一样的 framing） =====
        blocks = sliding_blocks(x, sr, self.cfg.block_win_s, self.cfg.block_hop_s)
        if len(blocks) == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        block_vecs = self.embedder.block_embed(x, blocks)

        # ===== 2) 单尺度 seeding（和 _seed_from_single 一致，用于 Pipeline / clustering）=====
        J_single, peaks_single = self._seed_candidates(x, sr,block_vecs)

        # ===== 3) run_cluster_hysteresis_backend 的 1~3 步，得到 segments / labels / J_list / C_list / scores / centers =====
        # 3.1) segments from peaks
        segments = self._segments_from_peaks(len(blocks), blocks, peaks_single)

        # 如果一个峰都没有，segments 至少会有 [0, 最后]，这里直接返回空
        if len(segments) <= 1:
            return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        # 3.2) segment-level embeddings & clustering（和 _run_cluster_hysteresis_backend 一致）
        block_starts = np.array([b[0] for b in blocks])
        seg_block_ranges = []
        for (s, e) in segments:
            idx = np.where((block_starts >= s) & (block_starts < e))[0]
            if len(idx) == 0:
                idx = np.array([np.argmin(np.abs(block_starts - s))])
            seg_block_ranges.append(idx)
        seg_vecs = self._segment_embeddings(block_vecs, seg_block_ranges)
        if self.cfg.cluster_method == "agglom":
            labels = cluster_segments(seg_vecs, method="agglom",
                                      distance_threshold=self.cfg.agglom_distance_threshold)
        elif self.cfg.cluster_method == "spectral":
            labels = cluster_segments(seg_vecs, method="spectral", n_clusters=2)
        else:
            labels = cluster_segments(seg_vecs, method="dbscan", eps=0.25, min_samples=2)
        labels = temporal_smooth(labels, min_len=self.cfg.min_run_segments)

        # 3.3) boundary scores: J_ij + C_ij
        J_list, C_list = [], []
        for i in range(len(segments) - 1):
            idx_i = seg_block_ranges[i]
            idx_j = seg_block_ranges[i + 1]
            bi = idx_i[-1]
            bj = idx_j[0]
            Jij = np.linalg.norm(block_vecs[bi] - block_vecs[bj])
            Cij = 1.0 if labels[i] != labels[i + 1] else 0.0
            J_list.append(Jij)
            C_list.append(Cij)
        J_arr = np.array(J_list, dtype=float)
        if J_arr.size and _rng(J_arr) > 0:
            J_arr = (J_arr - J_arr.min()) / (_rng(J_arr) + 1e-8)
        C_arr = np.array(C_list, dtype=float)
        scores = 1.0 * C_arr + 1.0 * J_arr  # 与原代码一致 w_cluster=1, w_jump=1
        if scores.size and _rng(scores) > 0:
            scores = (scores - scores.min()) / (_rng(scores) + 1e-8)

        # 3.4) boundary centers（和原来的 centers 计算完全一致）
        centers = np.array(
            [(segments[i][1] + segments[i + 1][0]) / (2.0 * sr) for i in range(len(segments) - 1)],
            dtype=float,
        )

        # ===== 4) multiscale seeding：用 _seed_from_multiscale 得到 fused peaks，做成特征 =====
        ms_flag = np.zeros_like(centers, dtype=float)
        ms_conf = np.zeros_like(centers, dtype=float)

        # 即使 cfg.multiscale=False，我们也可以强制调用 _seed_from_multiscale 当作特征用
        try:
            J_ms, peak_idx_ms, base_blocks_ms, fused = self._seed_from_multiscale(x, sr)
        except Exception:
            # 如果搞不定（比如 scales 配置不合法），就退化为不使用 multiscale 特征
            fused = []

        if fused:
            fused_times = np.array([t for (t, c) in fused], dtype=float)
            fused_conf = np.array([c for (t, c) in fused], dtype=float)
            # 在一个比较小的窗口内看某个候选 center 附近有没有 multiscale fused peak
            win = self.cfg.min_peak_distance_s / 2.0
            for i, t in enumerate(centers):
                idx = np.where(np.abs(fused_times - t) <= win)[0]
                if idx.size > 0:
                    ms_flag[i] = 1.0
                    ms_conf[i] = fused_conf[idx].max()

        # ===== 5) 组装特征 =====
        # 特征包含：
        #   scores   : cluster_hysteresis 中的最终 boundary score（未经过 hysteresis decode）
        #   J_arr    : 归一化的 J_ij
        #   C_arr    : 聚类标签是否变化 (0/1)
        #   ms_flag  : 附近是否有 multiscale fused peak
        #   ms_conf  : 附近 multiscale fused peak 的最大置信度
        X = np.stack(
            [scores.astype(np.float32),
             J_arr.astype(np.float32),
             C_arr.astype(np.float32),
             ms_flag.astype(np.float32),
             ms_conf.astype(np.float32)],
            axis=1,
        )

        # ===== 6) 打 label：|center - true_bound| <= tol 的记为 1 =====
        y = np.zeros(len(centers), dtype=np.int64)
        if true_bounds_sec is not None and true_bounds_sec.size > 0:
            tb = np.asarray(true_bounds_sec, dtype=float)
            for t_ref in tb:
                idx = np.where(np.abs(centers - t_ref) <= label_tolerance_s)[0]
                y[idx] = 1

        return X, y, centers.astype(np.float32)
    
    def _seed_candidates(self, x: np.ndarray, sr: int, block_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        jump = np.linalg.norm(np.diff(block_vecs, axis=0), axis=1)
        jump = (jump - np.min(jump)) / (_rng(jump) + 1e-8)
        jump = np.concatenate([[jump[0]], jump])
        mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
        flux = spectral_flux(mag)
        flux = (flux - np.min(flux)) / (_rng(flux) + 1e-8)
        frames = len(flux)
        frame_times = np.arange(frames) * (self.cfg.stft_hop / sr)
        block_times = np.array([b[0]/sr for b in sliding_blocks(x, sr, self.cfg.block_win_s, self.cfg.block_hop_s)])
        flux_block = np.interp(block_times, frame_times, flux)
        seed = self.cfg.alpha_seed * jump + (1.0 - self.cfg.alpha_seed) * flux_block
        seed = (seed - np.min(seed)) / (_rng(seed) + 1e-8)
        min_dist_blocks = max(1, int(self.cfg.min_peak_distance_s / self.cfg.block_hop_s))
        thresh = np.quantile(seed, self.cfg.peak_quantile)
        peaks = pick_peaks(seed, distance=min_dist_blocks, thresh=thresh)
        return seed, peaks

    def _segments_from_peaks(self, n_blocks: int, blocks: List[Tuple[int,int]], peaks: np.ndarray) -> List[Tuple[int,int]]:
        cutpoints = [0] + list(map(int, peaks)) + [n_blocks - 1]
        segments = []
        for i in range(len(cutpoints)-1):
            b0 = cutpoints[i]; b1 = cutpoints[i+1]
            s = blocks[b0][0]; e = blocks[b1][1]
            segments.append((s,e))
        return segments
    
#     def _seed_candidates(
#         self,
#         x: np.ndarray,
#         sr: int,
#         block_vecs: np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """基于 block embedding 跳变 + 谱通量 的综合 seed 曲线"""
#         # 1) block embedding 跳变
#         jump = np.linalg.norm(np.diff(block_vecs, axis=0), axis=1)
#         jump = (jump - np.min(jump)) / (_rng(jump) + 1e-8)
#         jump = np.concatenate([[jump[0]], jump])

#         # 2) 谱通量
#         mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
#         flux = spectral_flux(mag)
#         flux = (flux - np.min(flux)) / (_rng(flux) + 1e-8)

#         frames = len(flux)
#         frame_times = np.arange(frames) * (self.cfg.stft_hop / sr)

#         # 和 base blocks 对齐（使用 cfg.block_win_s / block_hop_s）
#         block_times = np.array(
#             [b[0] / sr for b in sliding_blocks(
#                 x, sr, self.cfg.block_win_s, self.cfg.block_hop_s
#             )]
#         )
#         flux_block = np.interp(block_times, frame_times, flux)

#         # 3) 融合 jump + flux_block
#         seed = self.cfg.alpha_seed * jump + (1.0 - self.cfg.alpha_seed) * flux_block
#         seed = (seed - np.min(seed)) / (_rng(seed) + 1e-8)

#         # 4) 在 seed 上做峰值检测
#         min_dist_blocks = max(
#             1, int(self.cfg.min_peak_distance_s / self.cfg.block_hop_s)
#         )
#         thresh = np.quantile(seed, self.cfg.peak_quantile)
#         peaks = pick_peaks(seed, distance=min_dist_blocks, thresh=thresh)

#         return seed, peaks
    
    def _segment_embeddings(self, block_vecs: np.ndarray, seg_block_ranges: List[np.ndarray]) -> np.ndarray:
        vecs = []
        for idx in seg_block_ranges:
            idx = np.asarray(idx)
            v = block_vecs[idx].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v.astype(np.float32))
        return np.vstack(vecs) if vecs else np.zeros((0, block_vecs.shape[1]), dtype=np.float32)

    ########################
    # Backends
    ########################
    
    def multiscale_detection(
        self,
        x: np.ndarray,
        sr: int,
        embedder: "Embedder",
        scales: List[float] = [0.4, 0.8, 1.6],
    ) -> Dict[str, np.ndarray]:
        """在多个时间尺度上检测边界"""
        all_boundaries = []
        all_confidences = []

        for scale in scales:
            # 该尺度下的分块
            blocks = sliding_blocks(x, sr, win_s=scale, hop_s=scale / 2)
            block_vecs = embedder.block_embed(x, blocks)

            # 邻接块 embedding 跳变
            jump = np.linalg.norm(np.diff(block_vecs, axis=0), axis=1)
            if jump.ptp() > 0:
                jump = (jump - jump.min()) / (jump.ptp() + 1e-8)
            else:
                jump = np.zeros_like(jump)
            jump = np.concatenate([[jump[0]], jump])

            # 峰值检测
            min_dist = max(1, int(self.cfg.min_peak_distance_s / (scale / 2)))  # 至少 0.5s 间隔
            thresh = np.percentile(jump, self.cfg.peak_quantile)           # 上分位数阈值
            peaks, properties = signal.find_peaks(
                jump, distance=min_dist, height=thresh
            )

            # 转成时间 + 置信度
            block_times = np.array([b[0] / sr for b in blocks])
            boundary_times = block_times[peaks]
            confidences = jump[peaks]

            all_boundaries.append(boundary_times)
            all_confidences.append(confidences)

        if len(all_boundaries) == 0:
            return {"boundaries": np.array([]), "confidences": np.array([])}

        all_times = np.concatenate(all_boundaries)
        all_conf = np.concatenate(all_confidences)

        if len(all_times) == 0:
            return {"boundaries": np.array([]), "confidences": np.array([])}

        # 按时间排序
        sorted_idx = np.argsort(all_times)
        all_times = all_times[sorted_idx]
        all_conf = all_conf[sorted_idx]

        # 时域聚类 + 投票
        final_boundaries = []
        final_confidences = []

        i = 0
        while i < len(all_times):
            cluster = [i]
            j = i + 1
            # 0.2s 内视作同一个 cluster
            while j < len(all_times) and (all_times[j] - all_times[i]) < 0.2:
                cluster.append(j)
                j += 1

            cluster_times = all_times[cluster]
            cluster_conf = all_conf[cluster]

            vote_score = len(cluster) / len(scales)   # 尺度同意度
            conf_score = cluster_conf.mean()          # 平均置信度

            # 条件：至少 ~2 个尺度同意 且 置信度高
            if vote_score >= 0.5 and conf_score >= 0.7:
                final_boundaries.append(cluster_times.mean())
                final_confidences.append(cluster_conf.max())

            i = j

        return {
            "boundaries": np.array(final_boundaries),
            "confidences": np.array(final_confidences),
        }


    def _run_cluster_hysteresis_backend(
            self,
            x: np.ndarray,
            sr: int,
            blocks: List[Tuple[int,int]],
            block_vecs: np.ndarray,
        ) -> Dict[str, np.ndarray]:

            # === 用 _seed_candidates 做种子 ===
            seed, peaks = self._seed_candidates(x, sr, block_vecs)
            # 后面全部沿用原来的逻辑，用新的 peaks 来构造 segments

            # 1) build segments from peaks
            segments = self._segments_from_peaks(len(blocks), blocks, peaks)

            # 2) segment-level embeddings and clustering
            block_starts = np.array([b[0] for b in blocks])
            seg_block_ranges = []
            for (s, e) in segments:
                idx = np.where((block_starts >= s) & (block_starts < e))[0]
                if len(idx) == 0:
                    idx = np.array([np.argmin(np.abs(block_starts - s))])
                seg_block_ranges.append(idx)

            seg_vecs = self._segment_embeddings(block_vecs, seg_block_ranges)

            if self.cfg.cluster_method == "agglom":
                labels = cluster_segments(
                    seg_vecs,
                    method="agglom",
                    distance_threshold=self.cfg.agglom_distance_threshold,
                )
            elif self.cfg.cluster_method == "spectral":
                labels = cluster_segments(
                    seg_vecs,
                    method="spectral",
                    n_clusters=self.cfg.n_clusters,
                )
            else:
                labels = cluster_segments(seg_vecs, method="agglom")

            # 3) boundary scores (J between adjacent segment edges) + C (label change)
            J_list, C_list = [], []
            for i in range(len(segments) - 1):
                idx_i = seg_block_ranges[i]
                idx_j = seg_block_ranges[i + 1]
                bi = idx_i[-1]
                bj = idx_j[0]
                Jij = np.linalg.norm(block_vecs[bi] - block_vecs[bj])
                Cij = 1.0 if labels[i] != labels[i + 1] else 0.0
                J_list.append(Jij)
                C_list.append(Cij)

            J_arr = np.array(J_list, dtype=float)
            if J_arr.size and _rng(J_arr) > 0:
                J_arr = (J_arr - J_arr.min()) / (_rng(J_arr) + 1e-8)

            #scores = 1.0 * np.array(C_list, dtype=float) + 1.0 * J_arr  # w_cluster=1, w_jump=1
            scores = self.cfg.w_cluster * np.array(C_list, dtype=float) + self.cfg.w_jump * J_arr  # w_cluster=1, w_jump=1

            if scores.size and _rng(scores) > 0:
                scores = (scores - scores.min()) / (_rng(scores) + 1e-8)

            centers = np.array(
                [
                    (segments[i][1] + segments[i + 1][0]) / (2.0 * sr)
                    for i in range(len(segments) - 1)
                ],
                dtype=float,
            )

            # 4) decode with hysteresis
            mask = hysteresis_mask(
                scores, self.cfg.hysteresis_high, self.cfg.hysteresis_low
            )
            cand_times = centers[mask]

            # 5) min-seg filter
            final_bounds = []
            last_t = None
            for t in cand_times:
                if last_t is None or (t - last_t) >= self.cfg.min_segment_s:
                    final_bounds.append(t)
                    last_t = t

            # 6) optional flux alignment（保持你原来的实现）
            if self.cfg.use_flux_align and len(final_bounds) > 0:
                mag = stft_mag(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.stft_hop)
                flux = spectral_flux(mag)
                flux = (flux - flux.min()) / (_rng(flux) + 1e-8)
                # ... 保留你原来的对 final_bounds 的微调逻辑 ...

            return {
                "bounds_sec": np.array(final_bounds, dtype=np.float32),
                "segments_samples": np.array(segments, dtype=int),
                "seg_labels": labels,
                "scores": scores,
            }



    def run(self, x: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        self._ensure_embedder(sr)
        blocks = sliding_blocks(
            x, sr, self.cfg.block_win_s, self.cfg.block_hop_s
        )
        block_vecs = self.embedder.block_embed(x, blocks)
        
        if self.cfg.backend == "multiscale_only":
            det = self.multiscale_detection(x, sr, self.embedder, scales=list(self.cfg.scales))
            bounds = det["boundaries"]
            confs = det["confidences"]

            # 用边界时间构造 segments（按样本）
            if bounds.size > 0:
                b_samps = (bounds * sr).astype(int)
                segments = []
                s0 = 0
                for b in b_samps:
                    segments.append((s0, int(b)))
                    s0 = int(b)
                segments.append((s0, len(x)))
            else:
                segments = [(0, len(x))]

            return {
                "bounds_sec": bounds.astype(np.float32),
                "segments_samples": np.array(segments, dtype=int),
                "seg_labels": np.arange(len(segments), dtype=int),
                "scores": confs,  # 这里就直接用多尺度置信度了
            }

        elif self.cfg.backend == "cluster_hysteresis":
            return self._run_cluster_hysteresis_backend(x, sr, blocks, block_vecs)

        elif self.cfg.backend == "viterbi_only":
            return self._run_viterbi_backend(x, sr, blocks, block_vecs)

        else:
            raise ValueError(f"Unknown backend: {self.cfg.backend}")

########################
# CLI (minimal)        #
########################

def _load_wav(path: str):
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype.kind in "iu":
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / maxv
    return x, sr

def write_csv_boundaries(path: str, bounds_sec: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("boundary_seconds\n")
        for t in bounds_sec:
            f.write(f"{float(t):.3f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, required=False, help="input wav for single-file run")
    ap.add_argument("--out", type=str, required=False, help="output csv of boundaries")
    ap.add_argument("--backend", type=str, default="cluster_hysteresis",
                    choices=["multiscale_only","viterbi_only","cluster_hysteresis"])
    ap.add_argument("--embed", type=str, default="logmel")
    ap.add_argument("--peak-quantile", type=float, default=0.9)
    ap.add_argument("--min-peak-dist", type=float, default=0.6)
    ap.add_argument("--min-seg", type=float, default=1.2)
    ap.add_argument("--hyst-high", type=float, default=0.6)
    ap.add_argument("--hyst-low", type=float, default=0.4)
    ap.add_argument("--agglom-th", type=float, default=0.6)
    ap.add_argument("--flux-align", action="store_true", default=True)
    ap.add_argument("--no-flux-align", action="store_false", dest="flux_align")

    args = ap.parse_args()

    cfg = SCDConfig(
        embed=args.embed,
        backend=args.backend,
        peak_quantile=args.peak_quantile,
        min_peak_distance_s=args.min_peak_dist,
        min_segment_s=args.min_seg,
        hysteresis_high=args.hyst_high,
        hysteresis_low=args.hyst_low,
        agglom_distance_threshold=args.agglom_th,
        use_flux_align=args.flux_align,
    )

    scd = SCDPipline(cfg)
    if args.wav:
        x, sr = _load_wav(args.wav)
        res = scd.run(x, sr)
        if args.out:
            write_csv_boundaries(args.out, res["bounds_sec"])
        else:
            print("Bounds (s):", np.round(res["bounds_sec"],3))
    else:
        print("[INFO] Library mode: import and call JOnlySCD(SCDConfig).run(x, sr).")

if __name__ == "__main__":
    main()
