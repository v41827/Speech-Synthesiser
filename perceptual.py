import os
import glob
import argparse
import librosa
import numpy as np
from pesq import pesq
from scipy.signal import get_window
from scipy.fft import rfft

TARGET_SR = 16000            # resample so PESQ 'wb' works
DEFAULT_REF = "speech/en.wav"  # only used if we can't guess

def safe_pesq(sr, ref, deg, path=""):
    """
    Wrapper around pesq() that never crashes the batch.
    Returns NaN if the underlying PESQ call fails.
    """
    try:
        return pesq(sr, ref, deg, "wb")
    except Exception as e:
        print(f"[pesq-error] {path}: {e}")
        return float("nan")

def lsd(ref, deg, nfft=1024):
    L = max(len(ref), len(deg))
    ref_pad = np.zeros(L, dtype=float)
    deg_pad = np.zeros(L, dtype=float)
    ref_pad[:len(ref)] = ref
    deg_pad[:len(deg)] = deg

    win = get_window("hann", L)
    Xr = np.abs(rfft(ref_pad * win, n=nfft))
    Xd = np.abs(rfft(deg_pad * win, n=nfft))
    eps = 1e-9
    return float(np.sqrt(np.mean((20*np.log10(Xr+eps) - 20*np.log10(Xd+eps))**2)))

def eval_pair(ref_path, deg_path, sr=TARGET_SR):
    ref, _ = librosa.load(ref_path, sr=sr, mono=True)
    deg, _ = librosa.load(deg_path, sr=sr, mono=True)

    # --- make sure PESQ sees signals of the same length (avoids ambiguous truth value in pypesq) ---
    L = min(len(ref), len(deg))
    ref = ref[:L]
    deg = deg[:L]

    pesq_score = safe_pesq(sr, ref, deg, path=deg_path)
    lsd_val = lsd(ref, deg)
    return pesq_score, lsd_val

def guess_ref_from_synth(synth_path):
    """
    synth path:
        synth_50ms_order36/synthesised_had_f_short_order36_50ms_2025...wav
    -> speech/had_f_short.wav
    """
    fname = os.path.basename(synth_path)
    if not fname.startswith("synthesised_") or "_order" not in fname:
        # fallback
        return DEFAULT_REF
    middle = fname[len("synthesised_"):fname.index("_order")]
    # e.g. "had_f_short"
    cand = os.path.join("speech", middle + ".wav")
    if os.path.exists(cand):
        return cand
    # fallback if missing
    return DEFAULT_REF

def main():
    parser = argparse.ArgumentParser("Perceptual eval (PESQ + LSD)")
    parser.add_argument("--ref", default=None,
                        help="reference wav (single-file mode or override)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="evaluate ONE synthesised wav")
    group.add_argument("--scan", action="store_true",
                       help="loop through all synth_* folders and evaluate every .wav")
    parser.add_argument("--pattern", default="synth_*/*.wav",
                        help="glob when using --scan (default: %(default)s)")
    parser.add_argument(
        "--csv",
        nargs="?",
        const="perceptual_results.csv",
        default=None,
        help="optional csv output (use --csv to save to perceptual_results.csv, or --csv myfile.csv)"
    )
    args = parser.parse_args()

    # -------- single file mode --------
    if args.file:
        if args.ref is None:
            # try to guess from filename if user didn't give --ref
            ref_path = guess_ref_from_synth(args.file)
        else:
            ref_path = args.ref

        pesq_score, lsd_val = eval_pair(ref_path, args.file)
        print(f"Reference : {ref_path}")
        print(f"Degraded  : {args.file}")
        print(f"PESQ      : {pesq_score:.3f}")
        print(f"LSD (dB)  : {lsd_val:.3f}")
        return

    # -------- scan / batch mode --------
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[warn] no files for pattern: {args.pattern}")
        return

    rows = [("file", "ref", "pesq", "lsd_db")]
    for f in files:
        # f is something like synth_50ms_order36/synthesised_en_order36_50ms_...wav
        ref_path = args.ref if args.ref is not None else guess_ref_from_synth(f)
        try:
            pesq_score, lsd_val = eval_pair(ref_path, f)
            print(f"{f}  |  ref={ref_path}  |  PESQ={pesq_score:.3f}, LSD={lsd_val:.3f} dB")
            rows.append((f, ref_path, f"{pesq_score:.3f}", f"{lsd_val:.3f}"))
        except Exception as e:
            print(f"[error] {f}: {e}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(",".join(r) + "\n")
        print(f"[info] results saved to {args.csv}")

if __name__ == "__main__":
    main()