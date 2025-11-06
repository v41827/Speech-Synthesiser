import numpy as np
import os
import soundfile as sf
import librosa
from scipy.signal import get_window, freqz, find_peaks
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
from datetime import datetime
import time
import re

# --- pre-defined parameters ---
AUDIO_PATH = "../speech_segments_100ms/en_segment.wav"
DEFAULT_ORDER = 48

def compute_lpc(audio_path=AUDIO_PATH, order=DEFAULT_ORDER):
    """
    Compute LPC coefficients and mean F0 from a given audio file.
    - audio_path: path to the audio file
    - order: LPC order
    Returns: (lpc_coefficients, mean_F0, sample_rate)
    """
    # --- load audio segments ---
    y, fs = sf.read(audio_path)
    print("Sampling rate:", fs)
    print("Audio shape:", y.shape)
    print(type(y))
    y = np.asarray(y, dtype=float).flatten()
    print("Audio type after conversion:", type(y))
    y = y - np.mean(y) # remove DC offset, optional

    # --- windowing for spectrum ---
    #win = get_window('hamming', len(y), fftbins=True)
    win = np.ones(len(y))# no window 
    yw = y * win

    # --- LPC analysis ---
    # rule-of-thumb for vowels at 16–24 kHz: 2 + fs/1000 to 2 + 2*fs/1000
    #order = min(36, max(20, int(2 + 1.5*(fs/1000))))  # e.g., 26–36
    a = librosa.lpc(yw, order=order)  # LPC denominator (a[0] ≈ 1)

    # --- mean F0 (Hz) ---
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=fs)
    print ("f0 contour (Hz):", f0)
    F0_mean = float(np.nanmean(f0))
    print("Mean F0 (Hz):", F0_mean)

    # --- plot amplitude spectrum (one-sided, in dB) ---
    nfft = 4096  # high resolution display
    X = rfft(yw, n=nfft)
    freqs = rfftfreq(nfft, 1/fs)
    #Amp = np.abs(X) / (np.sum(win)/2)  # simple amplitude normalization
    Amp = np.abs(X)/len(yw)
    Amp_dB = 20*np.log10(Amp + 1e-12)

    # --- LPC frequency response (in dB) on same grid ---
    gain = 1e-3  # adjust for display
    w_hz, H = freqz(b=[gain, 0.0], a=a, worN=nfft, fs=fs)
    H_dB = 20*np.log10(np.abs(H) + 1e-12)

    # --- formant picking from LPC envelope (limit to speech band) ---
    band = (w_hz >= 0) & (w_hz <= min(5000, fs/2))  # first 3 formants are <5 kHz
    peaks, _ = find_peaks(H_dB[band], distance=int(200/(w_hz[1]-w_hz[0])), prominence=3)
    formant_freqs = w_hz[band][peaks]
    formant_amps = H_dB[band][peaks]
    formant_freqs = np.sort(formant_freqs)[:3]  # F1–F3
    print("Formants (Hz):", ", ".join(f"F{i+1}={f:.0f}" for i,f in enumerate(formant_freqs)))

    return {
        "audio_path": audio_path,
        "order": order,
        "y": y,
        "fs": fs,
        "a": a,
        "F0_mean": F0_mean,
        "freqs": freqs,
        "Amp_dB": Amp_dB,
        "w_hz": w_hz,
        "H_dB": H_dB,
        "formants": formant_freqs,
    }

# ❶ 匯入的時候就先算好給別人用（你 synth 要用）
try: #comment out now when plotting, to avoid printing unrealted stuff in the terminal
    _result = compute_lpc()
    a = _result["a"]
    F0_mean = _result["F0_mean"]
    order = _result["order"]
    audio_path = _result["audio_path"]
except Exception as e:
    print(f"Warning: failed to compute LPC on default audio path '{AUDIO_PATH}': {e}")
    _result = None
    a = None
    F0_mean = None
    order = DEFAULT_ORDER
    audio_path = AUDIO_PATH

def plot_figure( _result=_result):
    if _result is None:
        _result = compute_lpc(audio_path=AUDIO_PATH, order=DEFAULT_ORDER)
    plot_lpc_result(_result, out_dir=None)

# ❷ 只有「直接跑 python lpc.py」的時候才畫圖＆存圖
import argparse

def plot_lpc_result(_result, save=True, show=False, out_dir=None):
    freqs = _result["freqs"]
    Amp_dB = _result["Amp_dB"]
    w_hz = _result["w_hz"]
    H_dB = _result["H_dB"]
    formants = _result["formants"]
    fs = _result["fs"]
    order = _result["order"]
    audio_path = _result["audio_path"]
    F0_mean = _result["F0_mean"]

    plt.figure(figsize=(10, 5))
    plt.plot(freqs, Amp_dB, linewidth=1, label="Amplitude spectrum (dB)")
    plt.plot(w_hz, H_dB, linewidth=2, alpha=0.6, label="LPC response (dB)")
    for i, f in enumerate(formants[:3], 1):
        plt.axvline(f, linestyle="--", linewidth=1, color="orange", alpha=0.6)
        plt.text(f, np.max(H_dB)-5-5*i, f"F{i} ≈ {f:.0f} Hz")

    plt.xlim(0, fs/2)
    plt.ylim(-120, 5)

    # --- annotate mean fundamental frequency on the figure (not a plotted line) ---
    # place it near the top-right corner within current axes limits
    text_x = 0.82 * (fs / 2)
    # choose a readable y slightly below the top limit
    text_y = -10
    plt.text(
        text_x,
        text_y,
        f"Mean F0 ≈ {F0_mean:.1f} Hz",
        ha="left",
        va="top",
        color="red",
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none"),
    )

    # Extract base filename
    base_filename = os.path.basename(audio_path)
    name_part = os.path.splitext(base_filename)[0]
    # Remove trailing '_segment' if present
    name_part = name_part.replace("_segment", "")
    # Get parent folder name
    folder_name = os.path.basename(os.path.dirname(audio_path))
    # Extract duration substring like '50ms', '100ms', etc.
    match = re.search(r"_(\d+)ms", folder_name)
    if match:
        duration = match.group(1) + "ms"
        title_str = f"Amplitude spectrum with LPC envelope ({name_part}, order={order}, {duration})"
    else:
        title_str = f"Amplitude spectrum with LPC envelope ({name_part}, order={order})"

    plt.title(title_str)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    save_name = f"lpc_spectrum_{base_name}_order{order}_{timestamp}.png"
    if save:
        if out_dir is None:
            out_dir = "."
        else:
            os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()

def plot_lpc_for_file(audio_path, order=DEFAULT_ORDER, save=True, show=False, out_dir=None):
    result = compute_lpc(audio_path=audio_path, order=order)
    plot_lpc_result(result, save=save, show=show, out_dir=out_dir)

def plot_lpc_for_folder(folder_path, order=DEFAULT_ORDER, save=True, show=False, out_dir=None):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".wav"):
            fpath = os.path.join(folder_path, fname)
            print(f"Processing: {fpath}")
            try:
                result = compute_lpc(audio_path=fpath, order=order)
                plot_lpc_result(result, save=save, show=show, out_dir=out_dir)
                if args.pz:
                    plot_pz_from_a(result["a"], result["fs"], result["order"], result["audio_path"], save=not args.nosave, show=args.show, out_dir=args.outdir)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")

def plot_figure(_result=_result):
    # Backward compatibility: keep original plot_figure signature
    plot_lpc_result(_result, out_dir=None)

def draw_circle():
    pass  # existing helper function placeholder

def draw_freq_db_circle():
    pass  # existing helper function placeholder

def plot_pz_from_a(a, fs, order, audio_path, save=True, show=False, out_dir=None):
    """
    Make a separate pole plot with the unit circle.
    - a: LPC denominator coefficients (a[0] ~ 1)
    - fs: sampling rate (for context in title)
    - order: LPC order (for title/filename)
    - audio_path: used to name the output
    """
    try:
        roots = np.roots(a) if a is not None else np.array([])
    except Exception as e:
        print(f"[plot_pz_from_a] Failed to compute roots: {e}")
        roots = np.array([])

    # Figure
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linestyle='--', linewidth=1.0, alpha=0.6, label='Unit circle')

    # Axes lines
    ax.axhline(0, color='k', linewidth=0.6, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.6, alpha=0.5)

    # Plot poles (x markers)
    if roots.size > 0:
        ax.plot(roots.real, roots.imag, 'x', markersize=7, mew=1.5, label='Poles')

    # Limits and aspect
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Titles and grid
    base_filename = os.path.basename(audio_path)
    name_part = os.path.splitext(base_filename)[0]
    # Extract duration (e.g., '50ms', '100ms') from parent folder name if present
    folder_name = os.path.basename(os.path.dirname(audio_path))
    match = re.search(r"_(\d+)ms", folder_name)
    if match:
        duration = match.group(1) + "ms"
        title_str = f"Pole plot with unit circle\n({name_part}, order={order}, fs={fs} Hz, {duration})"
    else:
        title_str = f"Pole plot with unit circle\n({name_part}, order={order}, fs={fs} Hz)"
    ax.set_title(title_str)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.85)

    # Stability hint
    if roots.size > 0 and np.any(np.abs(roots) >= 1.0):
        ax.text(
            0.02, 0.02, "Warning: Unstable poles (|z| ≥ 1)",
            transform=ax.transAxes, color='crimson',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='crimson', linewidth=0.8)
        )

    # Save/show
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_name = f"pz_{name_part}_order{order}_{timestamp}.png"
    if save:
        if out_dir is None:
            out_dir = "."
        else:
            os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, save_name)
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_pz(_result, save=True, show=False, out_dir=None):
    if _result is None:
        print("[plot_pz] No LPC result available.")
        return
    plot_pz_from_a(_result["a"], _result["fs"], _result["order"], _result["audio_path"], save=save, show=show, out_dir=out_dir)

def plot_both_for_file(audio_path, order=DEFAULT_ORDER, save=True, show=False, out_dir=None, circle=False, pz=False):
    res = compute_lpc(audio_path=audio_path, order=order)
    plot_lpc_result(res, save=save, show=show, out_dir=out_dir, circle=circle)
    if pz:
        plot_pz_from_a(res["a"], res["fs"], res["order"], res["audio_path"], save=save, show=show, out_dir=out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPC analysis and plotting")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--file', type=str, help="Plot LPC for a single wav file")
    group.add_argument('--folder', type=str, help="Plot LPC for all wav files in a folder")
    parser.add_argument('--order', type=int, default=DEFAULT_ORDER, help="LPC order (default: %(default)s)")
    parser.add_argument('--show', action='store_true', help="Display figures")
    parser.add_argument('--nosave', action='store_true', help="Do not save figures")
    parser.add_argument('--outdir', type=str, default=None, help="Directory to save plots into (will be created)")
    parser.add_argument('--pz', action='store_true', help="Also generate a separate pole plot with unit circle")
    args = parser.parse_args()

    if args.file:
        res = compute_lpc(audio_path=args.file, order=args.order)
        plot_lpc_result(res, save=not args.nosave, show=args.show, out_dir=args.outdir)
        if args.pz:
            plot_pz_from_a(res["a"], res["fs"], res["order"], res["audio_path"], save=not args.nosave, show=args.show, out_dir=args.outdir)
    elif args.folder:
        for fname in os.listdir(args.folder):
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(args.folder, fname)
                print(f"Processing: {fpath}")
                try:
                    result = compute_lpc(audio_path=fpath, order=args.order)
                    plot_lpc_result(result, save=not args.nosave, show=args.show, out_dir=args.outdir)
                    if args.pz:
                        plot_pz_from_a(result["a"], result["fs"], result["order"], result["audio_path"], save=not args.nosave, show=args.show, out_dir=args.outdir)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    else:
        # fallback: plot default AUDIO_PATH with default order
        if _result is not None:
            plot_lpc_result(_result, save=not args.nosave, show=args.show, out_dir=args.outdir)
            if args.pz:
                plot_pz_from_a(_result["a"], _result["fs"], _result["order"], _result["audio_path"], save=not args.nosave, show=args.show, out_dir=args.outdir)
        else:
            result = compute_lpc(audio_path=AUDIO_PATH, order=DEFAULT_ORDER)
            plot_lpc_result(result, save=not args.nosave, show=args.show, out_dir=args.outdir)
            if args.pz:
                plot_pz_from_a(result["a"], result["fs"], result["order"], result["audio_path"], save=not args.nosave, show=args.show, out_dir=args.outdir)
