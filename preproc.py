import os
import librosa
import numpy as np
import soundfile as sf

def extract_wav(audio_path, start_time=0.03, segment_length=0.1):
    """
    Extract a single segment from a given file.
    - start_time: offset in seconds to start extracting from
    - segment_length: length of the segment in seconds
    Returns: (segment_array, sample_rate)
    """
    audio, sample_rate = librosa.load(audio_path, sr=None)
    start_sample = int(start_time * sample_rate)
    seg_samples = int(segment_length * sample_rate)

    if start_sample >= len(audio):
        return np.array([]), sample_rate

    end = min(start_sample + seg_samples, len(audio))
    segment = audio[start_sample:end]
    return segment, sample_rate

def bulk_extract_wav(root_dir, start_time=0.03, segment_length=0.1, out_root=None):
    """
    Process all .wav files in root_dir, extract one segment per file and save them
    to a sibling folder named "<original_folder>_segments" (unless out_root given).
    Returns: dict mapping original filename -> (saved_path_or_None, sample_rate)
    """
    if out_root is None:
        parent = os.path.dirname(os.path.abspath(root_dir))
        base = os.path.basename(os.path.abspath(root_dir))
        seg_label = int(segment_length * 1000)
        out_root = os.path.join(parent, base + f'_segments_{seg_label}ms')
    os.makedirs(out_root, exist_ok=True)

    all_saved = {}
    for filename in sorted(os.listdir(root_dir)):
        if not filename.lower().endswith('.wav'):
            continue
        file_path = os.path.join(root_dir, filename)
        segment, sample_rate = extract_wav(file_path, start_time, segment_length)

        if segment.size == 0:
            all_saved[filename] = (None, sample_rate)
            continue

        name_no_ext = os.path.splitext(filename)[0]
        out_name = f"{name_no_ext}_segment.wav"
        out_path = os.path.join(out_root, out_name)
        sf.write(out_path, segment, samplerate=sample_rate)
        all_saved[filename] = (out_path, sample_rate)

    return all_saved

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    parser.add_argument("--start", type=float, default=0.03)
    parser.add_argument("--seglen", type=float, default=0.1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = bulk_extract_wav(args.root_dir, start_time=args.start, segment_length=args.seglen, out_root=args.out)
    print(result)
    print(result.get("sample_rate"))