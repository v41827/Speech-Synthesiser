#!/usr/bin/env python3
import os
import argparse
from datetime import datetime

import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# 你自己的分析程式
import lpc


# --------------------------------------------------------
# 1) 最底層：給我 LPC → 還你一段合成波形
# --------------------------------------------------------
def synth_from_lpc(a, f0_hz, fs, duration_sec=1.0, excitation_gain=1.0):
    """
    Synthesis using periodic impulse train + all-pole filter 1/A(z).
    This is the classic LPC source–filter model used in the assignment.
    """
    n_samples = int(duration_sec * fs)

    # --- build excitation ---
    if (f0_hz is None) or (f0_hz <= 0) or np.isnan(f0_hz):
        # fallback: unvoiced-ish noise, but small
        excitation = np.random.randn(n_samples) * 0.03
    else:
        period = max(1, int(round(fs / f0_hz)))
        excitation = np.zeros(n_samples, dtype=float)
        excitation[::period] = 1.0
        excitation *= excitation_gain

    # --- filter: all-pole 1 / A(z) ---
    y = lfilter([1.0], a, excitation)

    # --- normalise to avoid clipping ---
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


# --------------------------------------------------------
# 2) 畫合成後的頻譜（可選）
# --------------------------------------------------------
def plot_synth_spectrum(y, fs, title="Synthesised vowel spectrum", save_path=None, show=False):
    nfft = 4096
    X = np.fft.rfft(y, n=nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    amp = np.abs(X) / len(y)
    amp_db = 20 * np.log10(amp + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, amp_db, label="Synth amplitude (dB)")
    plt.xlim(0, fs / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


# --------------------------------------------------------
# 3) 單一檔案：分析 → 合成 → 存 → (可選)畫圖
# --------------------------------------------------------
def synth_file(
    audio_path: str,
    order: int = 48,
    duration_sec: float = 1.0,
    out_dir: str = "synth_out",
    prefix: str = "synthesised",
    do_plot: bool = False,
    show_plot: bool = False,
):
    """
    1. 用 lpc.compute_lpc(...) 分析這個 segment
    2. 用分析結果做 LPC 合成
    3. 存成一個帶時間戳的 wav
    4. (可選) 存一張頻譜圖
    """
    # --- analysis (reuse your lpc.py) ---
    result = lpc.compute_lpc(audio_path=audio_path, order=order)
    a = result["a"]
    f0 = result["F0_mean"]
    fs = result["fs"]

    # --- synthesis ---
    y_syn = synth_from_lpc(a, f0, fs, duration_sec=duration_sec)

    # --- naming ---
    base = os.path.splitext(os.path.basename(audio_path))[0]  # e.g. "hod_m_segment"
    clean_name = base.replace("_segment", "")

    # try to infer 50ms / 100ms / 200ms from parent folder (same trick as lpc.py)
    parent = os.path.basename(os.path.dirname(audio_path))
    dur_tag = ""
    if "50ms" in parent:
        dur_tag = "50ms"
    elif "100ms" in parent:
        dur_tag = "100ms"
    elif "200ms" in parent:
        dur_tag = "200ms"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(out_dir, exist_ok=True)

    if dur_tag:
        wav_name = f"{prefix}_{clean_name}_order{order}_{dur_tag}_{timestamp}.wav"
    else:
        wav_name = f"{prefix}_{clean_name}_order{order}_{timestamp}.wav"

    wav_path = os.path.join(out_dir, wav_name)

    # --- write wav ---
    sf.write(wav_path, y_syn, fs)
    print(f"✓ saved: {wav_path}")

    # --- optional plot ---
    if do_plot:
        if dur_tag:
            plot_title = f"Synth spectrum ({clean_name}, order={order}, {dur_tag})"
            png_name = f"synth_spectrum_{clean_name}_order{order}_{dur_tag}_{timestamp}.png"
        else:
            plot_title = f"Synth spectrum ({clean_name}, order={order})"
            png_name = f"synth_spectrum_{clean_name}_order{order}_{timestamp}.png"

        png_path = os.path.join(out_dir, png_name)
        plot_synth_spectrum(y_syn, fs, title=plot_title, save_path=png_path, show=show_plot)

    return wav_path


# --------------------------------------------------------
# 4) 資料夾版：loop
# --------------------------------------------------------
def synth_folder(
    folder_path: str,
    order: int = 48,
    duration_sec: float = 1.0,
    out_dir: str = "synth_out",
    prefix: str = "synthesised",
    do_plot: bool = False,
    show_plot: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            synth_file(
                fpath,
                order=order,
                duration_sec=duration_sec,
                out_dir=out_dir,
                prefix=prefix,
                do_plot=do_plot,
                show_plot=show_plot,
            )
        except Exception as e:
            print(f"✗ failed on {fpath}: {e}")


# --------------------------------------------------------
# 5) CLI
# --------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LPC-based synthesis (single or bulk)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="synthesise a single wav segment")
    group.add_argument("--folder", type=str, help="synthesise all wav segments in a folder")

    parser.add_argument("--order", type=int, default=48, help="LPC order (must match analysis)")
    parser.add_argument("--dur", type=float, default=1.0, help="synthesis duration in seconds (assignment ~1s)")
    parser.add_argument("--outdir", type=str, default="synth_out", help="output folder for wavs")
    parser.add_argument("--prefix", type=str, default="synthesised", help="output filename prefix")
    parser.add_argument("--plot", action="store_true", help="also save spectrum plots for the synthesised signals")
    parser.add_argument("--show", action="store_true", help="show plots interactively (use with --plot)")

    args = parser.parse_args()

    if args.file:
        synth_file(
            args.file,
            order=args.order,
            duration_sec=args.dur,
            out_dir=args.outdir,
            prefix=args.prefix,
            do_plot=args.plot,
            show_plot=args.show,
        )
    else:
        synth_folder(
            args.folder,
            order=args.order,
            duration_sec=args.dur,
            out_dir=args.outdir,
            prefix=args.prefix,
            do_plot=args.plot,
            show_plot=args.show,
        )


if __name__ == "__main__":
    main()