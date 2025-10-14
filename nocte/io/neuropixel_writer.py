"""
Code to generate an ap.bin and ap.meta from data.
Useful to process raw data (e.g. concatenating) and saving it in a
format that is transparent for the rest of the pipeline.

Code is very simplified and does not support full npix format potential.
"""
import logging
import math
from pathlib import Path
import numpy as np

INT16_MIN = -32768
INT16_MAX =  32767
BYTES_PER_INT16 = 2
ADC_DIVISOR = 512.0      # used by your loader: i2v = imAiRangeMax / 512
UV_PER_V = 1_000_000.0   # microvolts per volt


def _as_2d_float(arr: np.ndarray) -> np.ndarray:
    """Return (n_channels, n_samples) float array. Replace non-finite with 0."""
    x = np.asarray(arr)
    assert x.ndim == 2

    x = x.astype(float, copy=False)

    if not np.all(np.isfinite(x)):
        x = np.where(np.isfinite(x), x, 0.0)

    return x

def _microvolts_per_bit(im_ai_range_max_v: float, ap_gain_index: int) -> float:
    """
    Your loader computes: conv_microvolts_per_bit = (imAiRangeMax / 512) * 1e6 / ap_gain.
    We use the same so that writing -> reading returns original amplitudes.
    """
    if not (im_ai_range_max_v > 0 and math.isfinite(im_ai_range_max_v)):
        raise ValueError("im_ai_range_max_v must be positive and finite.")
    if not (ap_gain_index > 0):
        raise ValueError("ap_gain_index must be positive.")
    return (im_ai_range_max_v / ADC_DIVISOR) * UV_PER_V / ap_gain_index

def _float_microvolts_to_int16(data_microvolts: np.ndarray, im_ai_range_max_v: float, ap_gain_index: int) -> np.ndarray:
    """
    Scale float microvolts to int16 using the SAME mapping your loader inverts.
    """
    microvolts_bit = _microvolts_per_bit(im_ai_range_max_v, ap_gain_index)
    scale = 1.0 / microvolts_bit                      # counts per microvolt
    y64 = np.rint(data_microvolts * scale).astype(np.int64)
    y16 = np.clip(y64, INT16_MIN, INT16_MAX).astype(np.int16)
    return y16

# ---------- writers ----------
def write_ap_bin(
    bin_path: Path,
    chunks: list[np.ndarray],
    im_ai_range_max_v: float = 0.6,
    ap_gain_index: int = 500,
):
    """
    Write Neuropixels-style int16 ap.bin (time-major, interleaved).
    Each chunk must be (n_channels, n_samples).
    """
    with open(str(bin_path), "wb") as f:
        for chunk in chunks:
            chunk_float = _as_2d_float(chunk)                               # (C, S) float uV
            chunk_i16 = _float_microvolts_to_int16(chunk_float, im_ai_range_max_v, ap_gain_index)
            # write as C-order time-major: [t0: ch0..chC-1, t1: ch0..] etc.
            # transpose to (S, C) then flatten in C-order
            chunk_i16.T.ravel(order="C").tofile(f)

def write_ap_meta(
    meta_path: Path,
    n_channels: int,
    n_samples: int,
    sampling_rate_hz: float,
    im_ai_range_max_v: float = 0.6,
    ap_gain_index: int = 500,
):
    """
    Minimal SpikeGLX-like .ap.meta (only fields your loader actually uses).
    """
    if not (n_channels > 0 and n_samples >= 0):
        raise ValueError("n_channels must be >0 and n_samples >=0.")
    if not (sampling_rate_hz > 0 and math.isfinite(sampling_rate_hz)):
        raise ValueError("sampling_rate_hz must be positive and finite.")

    file_size_bytes = n_samples * n_channels * BYTES_PER_INT16
    im_ai_range_min_v = -abs(im_ai_range_max_v)

    # snsApLfSy = AP,LF,SY
    sns_ap_lf_sy = f"{n_channels},0,0"
    # Save all channels by index 0..n_channels-1 (your parser handles 'all' too; this is explicit)
    sns_save_subset = "0" if n_channels == 1 else f"0:{n_channels-1}"

    # imroTbl header and one entry per channel:
    # (chan bank refType apGain lfGain something) — match your examples: (.. 0 0 apGain lfGain 0)
    imro_header = f"(0,{n_channels})"
    imro_entries = "".join(f"({ch} 0 0 {ap_gain_index} {ap_gain_index} 0)" for ch in range(n_channels))
    imro_tbl = imro_header + imro_entries

    lines = [
        f"fileSizeBytes={file_size_bytes}",
        f"nSavedChans={n_channels}",
        "typeThis=imec",
        f"imSampRate={sampling_rate_hz}",
        f"imAiRangeMax={im_ai_range_max_v}",
        f"imAiRangeMin={im_ai_range_min_v}",
        f"snsApLfSy={sns_ap_lf_sy}",
        f"snsSaveChanSubset={sns_save_subset}",
        "imStdby=",
        f"~imroTbl={imro_tbl}",
    ]

    with open(str(meta_path), "w") as f:
        f.write("\n".join(lines) + "\n")

def store_neuropixels_data(
    output_folder: str | Path,
    chunks: list[np.ndarray],
    sampling_rate_hz: float,
    im_ai_range_max_v: float = 0.6,
    ap_gain_index: int = 500,
    stem: str = "data",
    overwrite=False,
):
    """
    Write a Neuropixels-style ap.bin and ap.meta file
    into the specified output folder.

    :param output_folder: str or Path
        Directory where the .ap.bin and .ap.meta files will be created.
        The folder is created if it does not exist.

    :param chunks: list of np.ndarray
        List of 2D arrays, each of shape (n_channels, n_samples),
        containing float data in microvolts. All chunks must have the
        same number of channels. Chunks are concatenated along the
        time (sample) axis.

    :param sampling_rate_hz: float
        Sampling rate in Hz. Stored in the .ap.meta file.

    :param im_ai_range_max_v: float, optional
        The maximum analog input range (in volts) for the generated
        recording. Used for amplitude scaling and metadata.

    :param ap_gain_index: int, optional
        Gain index used by the loader. Determines how float microvolt
        values are converted to int16.

    :param stem: str, optional
        Base filename (without extension). The script writes
        <stem>.ap.bin and <stem>.ap.meta.

    :param overwrite: bool, optional
        If False and files already exist, skip writing. If True,
        overwrite existing files.

    :return:
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    bin_path = output_folder / f"{stem}.ap.bin"
    meta_path = output_folder / f"{stem}.ap.meta"

    if (bin_path.exists() or meta_path.exists()) and not overwrite:
        logging.warning(f'Files exist. Skipping')
        return

    assert all(chunk.ndim == 2 for chunk in chunks)

    channel_counts = np.array([chunk.shape[0] for chunk in chunks])
    if not np.allclose(channel_counts, channel_counts[0]):
        raise ValueError(f"Inconsistent channel count across chunks: {list(channel_counts)}")
    n_channels: int = channel_counts[0]

    total_samples: int = sum([chunk.shape[1] for chunk in chunks])

    write_ap_bin(
        bin_path=bin_path,
        chunks=chunks,
        im_ai_range_max_v=im_ai_range_max_v,
        ap_gain_index=ap_gain_index,
    )

    write_ap_meta(
        meta_path=meta_path,
        n_channels=n_channels,
        n_samples=total_samples,
        sampling_rate_hz=sampling_rate_hz,
        im_ai_range_max_v=im_ai_range_max_v,
        ap_gain_index=ap_gain_index,
    )
