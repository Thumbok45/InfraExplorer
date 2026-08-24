"""InfraExplorer DSP engine — local Python port of the browser analyzer.

Requires numpy. The original recording is never written.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Literal

import numpy as np

APP_NAME = "InfraExplorer"
APP_VERSION = "0.2.0"
DSP_VERSION = "0.1"
PROJECT_SCHEMA = 1

DEFAULT_FMIN = 0.1
DEFAULT_FMAX = 30.0
DEFAULT_SPEED = 200.0
DEFAULT_OUTPUT_RATE = 48000
HEADROOM_DB = -3.0
SILENCE_DB = -120.0
INGEST_RATE_CAP = 800

GainMode = Literal["preserve", "conservative", "medium", "aggressive", "manual"]
BandPresetId = Literal["coarse", "log", "nuclear1", "nuclear0p5", "custom"]
SyntheticKind = Literal["garage-night", "garage-short", "calibration", "sweep"]
ProgressFn = Callable[[dict], None]

GAIN_CAPS = {"conservative": 18.0, "medium": 30.0, "aggressive": 48.0}
GAIN_K = {"conservative": 0.38, "medium": 0.62, "aggressive": 0.85}

COARSE_INFRASOUND = [(0.1, 0.5), (0.5, 2.0), (2.0, 10.0), (10.0, 22.0), (22.0, 30.0)]
LOGARITHMIC = [
    (0.1, 0.2),
    (0.2, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 30.0),
]
PRESET_LABELS = {
    "coarse": "Coarse Infrasound",
    "log": "Logarithmic",
    "nuclear1": "1-Hz Nuclear",
    "nuclear0p5": "0.5-Hz Nuclear",
    "custom": "Custom",
}

STAGES = [
    "Validate input",
    "Build low-rate source",
    "Analyze spectrum",
    "Extract filter bank",
    "Measure bands",
    "Build previews",
    "Finalize",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Band:
    lo: float
    hi: float
    name: str
    enabled: bool = True


@dataclass
class BandMeasurement:
    peak_dbfs: float
    rms_dbfs: float
    robust_rms_dbfs: float
    crest_db: float
    active_fraction: float
    relative_db: float
    classification: str
    low_confidence: bool
    suggested_gain: float
    max_auto_gain: float


@dataclass
class ChannelIntegrity:
    peak_dbfs: float
    rms_dbfs: float
    dc_bias: float
    clipped_samples: int
    nan_samples: int
    inf_samples: int
    silent: bool


@dataclass
class SourceInfo:
    kind: str
    name: str
    format: str
    duration: float
    channels: int
    sample_rate: int
    sample_format: str
    bytes: int
    peaks: list[float]
    rms: list[float]
    clipping: bool
    path: str = ""


@dataclass
class SpectrumSurvey:
    freqs: np.ndarray
    mean_dbfs: np.ndarray
    peak_dbfs: np.ndarray


@dataclass
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    fmt: str
    data_offset: int
    data_bytes: int
    frames: int
    is_rf64: bool
    duration: float
    container: str = "WAV"


@dataclass
class ProcessConfig:
    bands: list[Band]
    f_min: float = DEFAULT_FMIN
    f_max: float = DEFAULT_FMAX
    working_rate: int = 200
    speed: float = DEFAULT_SPEED
    microscope_speed: float = 500
    output_rate: int = DEFAULT_OUTPUT_RATE
    gain_mode: GainMode = "medium"
    phase_policy: str = "analytical"
    test_duration: float | None = None
    channel_names: list[str] = field(default_factory=lambda: ["Channel 1", "Channel 2"])


@dataclass
class ProcessResult:
    working_rate: int
    channels: list[np.ndarray]
    band_audio: list[list[np.ndarray]]
    measurements: list[list[BandMeasurement]]
    suggested_gains: list[list[float]]
    spectrum: list[SpectrumSurvey]
    reconstruction_db: list[float]
    integrity: list[ChannelIntegrity]
    log: list[str]


@dataclass
class SyntheticSpec:
    id: str
    title: str
    blurb: str
    duration: float
    sample_rate: int
    channels: int


SYNTHETICS = [
    SyntheticSpec(
        "garage-night",
        "Garage Night",
        "Two-hour stereo piezo session: 19.3 Hz machinery, 3.8 Hz rumble, weak 0.23 Hz, localized 8.4 Hz, occasional thumps.",
        7200,
        200,
        2,
    ),
    SyntheticSpec(
        "garage-short",
        "Garage Night — 10 min",
        "Same spectral mix, first ten minutes. Use this to audition settings before a long run.",
        600,
        200,
        2,
    ),
    SyntheticSpec(
        "calibration",
        "Calibration Tones",
        "Known components at 0.10, 0.50, 3.70, and 19.00 Hz plus an impulse. For DSP regression.",
        120,
        200,
        2,
    ),
    SyntheticSpec(
        "sweep",
        "Log Sweep 0.1–30 Hz",
        "Ninety-second logarithmic sweep, equal amplitude, both channels.",
        90,
        200,
        2,
    ),
]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def dbfs(amp: float) -> float:
    a = abs(float(amp))
    if a <= 1e-20:
        return SILENCE_DB
    return 20.0 * math.log10(a)


def from_dbfs(db: float) -> float:
    return 10.0 ** (db / 20.0)


def rms(signal: np.ndarray, start: int = 0, end: int | None = None) -> float:
    sl = signal[start:end]
    if sl.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(sl, dtype=np.float64))))


def peak_abs(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.max(np.abs(signal)))


def mean(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.mean(signal, dtype=np.float64))


def percentile_abs(signal: np.ndarray, p: float) -> float:
    if signal.size == 0:
        return 0.0
    n = min(signal.size, 200_000)
    step = max(1, signal.size // n)
    sample = np.abs(signal[::step])
    sample.sort()
    idx = min(sample.size - 1, max(0, int(p * (sample.size - 1))))
    return float(sample[idx])


def hann(n: int, periodic: bool = True) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    denom = n if periodic else n - 1
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * math.pi * i) / denom)


def format_hz(f: float) -> str:
    if not math.isfinite(f):
        return "—"
    a = abs(f)
    if a >= 1000:
        return f"{f / 1000:.2f} kHz"
    if a >= 10:
        return f"{f:.1f} Hz"
    return f"{f:.2f} Hz"


def format_band(lo: float, hi: float) -> str:
    def trim(v: float) -> str:
        if v >= 10:
            return f"{v:.0f}"
        if v >= 1:
            return f"{v:.1f}"
        return f"{v:.2f}"

    return f"{trim(lo)}–{trim(hi)} Hz"


def format_duration(sec: float) -> str:
    if not math.isfinite(sec) or sec < 0:
        return "—"
    s_abs = abs(sec)
    h = int(s_abs // 3600)
    m = int((s_abs % 3600) // 60)
    s = int(s_abs % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    cs = int((s_abs % 1) * 100)
    return f"{m}:{s:02d}.{cs:02d}"


def format_db(db: float, digits: int = 1) -> str:
    if not math.isfinite(db):
        return "—"
    n = f"{db:.{digits}f}"
    return f"+{n} dB" if db > 0 else f"{n} dB"


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f" {n / 1024:.1f} KB".strip()
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def safe_filename(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    out = []
    for ch in base:
        if ch in '<>:"/\\|?*':
            out.append("_")
        elif ch.isspace():
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("._")[:80]
    return s or "InfraExplorer"


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def make_band(lo: float, hi: float, name: str | None = None) -> Band:
    return Band(lo=lo, hi=hi, name=name or format_band(lo, hi), enabled=True)


def nuclear_bands(f_min: float, f_max: float, step: float) -> list[tuple[float, float]]:
    edges: list[float] = []
    if f_min < step:
        edges.append(f_min)
        x = step
        while x < f_max - 1e-9:
            edges.append(x)
            x += step
        edges.append(f_max)
    else:
        x = f_min
        while x < f_max - 1e-9:
            edges.append(x)
            x += step
        edges.append(f_max)
    pairs = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo > 1e-9:
            pairs.append((lo, hi))
    return pairs


def bands_from_pairs(pairs: Iterable[tuple[float, float]]) -> list[Band]:
    return [make_band(lo, hi) for lo, hi in pairs]


def preset_bands(
    ident: BandPresetId,
    f_min: float = DEFAULT_FMIN,
    f_max: float = DEFAULT_FMAX,
    custom: list[Band] | None = None,
) -> list[Band]:
    if ident == "custom":
        return custom if custom else bands_from_pairs(COARSE_INFRASOUND)
    if ident == "coarse":
        pairs = [
            (max(lo, f_min), min(hi, f_max))
            for lo, hi in COARSE_INFRASOUND
            if min(hi, f_max) > max(lo, f_min) + 1e-9
        ]
        return bands_from_pairs(pairs)
    if ident == "log":
        pairs = [
            (max(lo, f_min), min(hi, f_max))
            for lo, hi in LOGARITHMIC
            if min(hi, f_max) > max(lo, f_min) + 1e-9
        ]
        return bands_from_pairs(pairs)
    if ident == "nuclear1":
        return bands_from_pairs(nuclear_bands(f_min, f_max, 1.0))
    return bands_from_pairs(nuclear_bands(f_min, f_max, 0.5))


def auto_working_rate(f_max: float) -> int:
    minimum = f_max * 6
    choices = [128, 200, 256, 400, 512, 800, 1024, 1600, 2048]
    for r in choices:
        if r >= minimum:
            return r
    return int(math.ceil(minimum))


def speed_safety(
    f_min: float, f_max: float, speed: float, source_duration: float, output_rate: int
) -> dict:
    output_range = (f_min * speed, f_max * speed)
    nyq = output_rate / 2
    warnings: list[str] = []
    if output_range[0] < 20:
        warnings.append(
            f"Lowest translated frequency ({output_range[0]:.1f} Hz) remains below normal hearing."
        )
    if output_range[1] > nyq * 0.9:
        warnings.append(
            f"Highest translated frequency ({output_range[1]:.0f} Hz) exceeds safe output bandwidth (Nyquist {nyq:.0f} Hz)."
        )
    if source_duration * f_min < 10:
        warnings.append(
            f"Recording contains only {source_duration * f_min:.1f} cycles of {f_min} Hz. Spectral estimates at the low edge will be unreliable."
        )
    return {
        "source_range": (f_min, f_max),
        "speed": speed,
        "output_range": output_range,
        "output_duration": source_duration / speed if speed else 0,
        "output_nyquist": nyq,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def _fourcc(buf: bytes, offset: int) -> str:
    return buf[offset : offset + 4].decode("ascii", errors="replace")


def parse_wav_header(buf: bytes) -> WavInfo:
    if len(buf) < 12:
        raise ValueError("File is too small to be a WAV.")
    riff = _fourcc(buf, 0)
    wave = _fourcc(buf, 8)
    is_rf64 = riff == "RF64"
    if riff not in ("RIFF", "RF64") or wave != "WAVE":
        raise ValueError("Not a WAV / RF64 file.")

    offset = 12
    fmt_code = 1
    channels = 1
    sample_rate = 48000
    bits = 16
    data_offset = -1
    data_bytes = 0
    rf64_data = 0

    view = memoryview(buf)
    while offset + 8 <= len(buf):
        cid = _fourcc(buf, offset)
        size = struct.unpack_from("<I", buf, offset + 4)[0]
        body = offset + 8
        if cid == "ds64" and body + 16 <= len(buf):
            rf64_data = struct.unpack_from("<Q", buf, body + 8)[0]
        elif cid == "fmt " and body + 16 <= len(buf):
            fmt_code = struct.unpack_from("<H", buf, body)[0]
            channels = struct.unpack_from("<H", buf, body + 2)[0]
            sample_rate = struct.unpack_from("<I", buf, body + 4)[0]
            bits = struct.unpack_from("<H", buf, body + 14)[0]
            if fmt_code == 0xFFFE and size >= 40:
                fmt_code = struct.unpack_from("<H", buf, body + 24)[0]
        elif cid == "data":
            data_offset = body
            data_bytes = rf64_data if size == 0xFFFFFFFF else size
            break
        if size % 2 == 1:
            size += 1
        offset = body + size

    if data_offset < 0:
        raise ValueError("WAV has no data chunk.")
    frame_size = (bits // 8) * channels
    if frame_size <= 0:
        raise ValueError("Invalid WAV frame size.")
    frames = data_bytes // frame_size
    if fmt_code == 3:
        kind = "float"
    elif fmt_code == 1:
        kind = "pcm"
    else:
        raise ValueError(f"Unsupported WAV format code {fmt_code}. Use PCM or IEEE float.")
    return WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits,
        fmt=kind,
        data_offset=data_offset,
        data_bytes=data_bytes,
        frames=frames,
        is_rf64=is_rf64,
        duration=frames / sample_rate if sample_rate else 0,
        container="RF64 WAV" if is_rf64 else "WAV",
    )


def read_wav_info(path: str) -> WavInfo:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        buf = f.read(min(size, 65536))
    info = parse_wav_header(buf)
    if info.data_bytes <= 0:
        info.data_bytes = size - info.data_offset
        frame_size = (info.bits_per_sample // 8) * info.channels
        info.frames = info.data_bytes // frame_size
        info.duration = info.frames / info.sample_rate if info.sample_rate else 0
    return info


def _decode_pcm_block(raw: bytes, info: WavInfo) -> np.ndarray:
    """Return float32 interleaved samples."""
    bps = info.bits_per_sample
    n_ch = info.channels
    if info.fmt == "float":
        if bps == 32:
            return np.frombuffer(raw, dtype="<f4").copy()
        if bps == 64:
            return np.frombuffer(raw, dtype="<f8").astype(np.float32)
        raise ValueError(f"Unsupported float bit depth {bps}.")
    if bps == 16:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if bps == 32:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    if bps == 8:
        u = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (u - 128.0) / 128.0
    if bps == 24:
        n = len(raw) // 3
        b = np.frombuffer(raw, dtype=np.uint8)
        if b.size < n * 3:
            n = b.size // 3
            b = b[: n * 3]
        packed = b.reshape(-1, 3)
        val = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (
            packed[:, 2].astype(np.int32) << 16
        )
        val = np.where(val & 0x800000, val | ~0xFFFFFF, val)
        return val.astype(np.float32) / 8388608.0
    raise ValueError(f"Unsupported bit depth {bps}.")
    _ = n_ch
    return np.zeros(0, dtype=np.float32)


def resample_linear(signal: np.ndarray, in_rate: float, out_rate: float) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float64, copy=True)
    if abs(in_rate - out_rate) < 1e-9:
        return signal.astype(np.float64, copy=True)
    out_len = max(1, int(round(signal.size * out_rate / in_rate)))
    src = np.linspace(0, signal.size - 1, out_len)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(signal.size - 1, i0 + 1)
    t = src - i0
    x = signal.astype(np.float64, copy=False)
    return x[i0] + (x[i1] - x[i0]) * t


def _lowpass_decimate(signal: np.ndarray, in_rate: float, out_rate: float) -> np.ndarray:
    """Anti-aliased downsample via successive ×2 averaging, then linear resample."""
    x = signal.astype(np.float64, copy=True)
    rate = float(in_rate)
    while rate / 2 >= out_rate * 0.98 and x.size >= 4:
        if x.size % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
        rate /= 2
    return resample_linear(x, rate, out_rate)


def decode_wav(
    path: str,
    max_seconds: float | None = None,
    ingest_rate: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[np.ndarray], int, WavInfo]:
    info = read_wav_info(path)
    frames = info.frames
    if max_seconds and max_seconds > 0:
        frames = min(frames, int(max_seconds * info.sample_rate))
    frame_size = (info.bits_per_sample // 8) * info.channels
    target = ingest_rate if ingest_rate and ingest_rate < info.sample_rate else None

    channels = [np.zeros(frames, dtype=np.float32) for _ in range(info.channels)]
    chunk = 65536
    written = 0
    with open(path, "rb") as f:
        f.seek(info.data_offset)
        while written < frames:
            take = min(chunk, frames - written)
            raw = f.read(take * frame_size)
            if not raw:
                break
            got = len(raw) // frame_size
            raw = raw[: got * frame_size]
            inter = _decode_pcm_block(raw, info)
            n = min(got, inter.size // info.channels)
            shaped = inter[: n * info.channels].reshape(n, info.channels)
            for ch in range(info.channels):
                channels[ch][written : written + n] = shaped[:, ch]
            written += n
            if on_progress:
                on_progress(written / max(1, frames) * (0.7 if target else 1.0))

    channels = [c[:written] for c in channels]
    rate = info.sample_rate
    if target and target < rate:
        channels = [_lowpass_decimate(c, rate, target).astype(np.float32) for c in channels]
        rate = target
        if on_progress:
            on_progress(1.0)
    return channels, rate, info


def encode_wav32f(channels: list[np.ndarray], sample_rate: int, path: str) -> None:
    if not channels:
        raise ValueError("No channels to encode.")
    n_ch = len(channels)
    frames = int(channels[0].size)
    data_bytes = frames * n_ch * 4
    interleaved = np.empty((frames, n_ch), dtype="<f4")
    for ch in range(n_ch):
        interleaved[:, ch] = channels[ch][:frames].astype(np.float32, copy=False)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        3,
        n_ch,
        sample_rate,
        sample_rate * n_ch * 4,
        n_ch * 4,
        32,
        b"data",
        data_bytes,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(interleaved.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Inclusive decode (FLAC, AIFF, WAV, …)
# ---------------------------------------------------------------------------

LOSSY_EXT = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
WAV_EXT = {".wav", ".rf64", ".w64"}


def sniff_container(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        head = b""
    if head.startswith(b"fLaC"):
        return "FLAC"
    if len(head) >= 12 and head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE":
        return "RF64 WAV" if head[:4] == b"RF64" else "WAV"
    if head.startswith(b"FORM") and (b"AIFF" in head or b"AIFC" in head):
        return "AIFF"
    if head.startswith(b"caff"):
        return "CAF"
    if head.startswith(b"OggS"):
        return "OGG"
    names = {
        ".flac": "FLAC",
        ".wav": "WAV",
        ".rf64": "RF64 WAV",
        ".aiff": "AIFF",
        ".aif": "AIFF",
        ".aifc": "AIFF",
        ".caf": "CAF",
        ".w64": "W64",
        ".wv": "WavPack",
        ".ogg": "OGG",
        ".oga": "OGG",
        ".mp3": "MP3",
        ".m4a": "M4A",
        ".aac": "AAC",
    }
    return names.get(ext, ext.lstrip(".").upper() or "AUDIO")


def _subtype_bits(subtype: str) -> tuple[int, str]:
    table = {
        "PCM_S8": (8, "pcm"),
        "PCM_U8": (8, "pcm"),
        "PCM_16": (16, "pcm"),
        "PCM_24": (24, "pcm"),
        "PCM_32": (32, "pcm"),
        "FLOAT": (32, "float"),
        "DOUBLE": (64, "float"),
    }
    if subtype in table:
        return table[subtype]
    if "24" in subtype:
        return 24, "pcm"
    if "32" in subtype:
        return 32, "pcm"
    if "FLOAT" in subtype.upper():
        return 32, "float"
    return 16, "pcm"


def _make_info(
    sample_rate: int,
    channels: int,
    bits: int,
    fmt: str,
    frames: int,
    container: str,
    is_rf64: bool = False,
) -> WavInfo:
    return WavInfo(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits,
        fmt=fmt,
        data_offset=0,
        data_bytes=frames * channels * max(1, bits // 8),
        frames=frames,
        is_rf64=is_rf64,
        duration=frames / sample_rate if sample_rate else 0.0,
        container=container,
    )


def _apply_ingest(
    channels: list[np.ndarray],
    rate: int,
    ingest_rate: int | None,
) -> tuple[list[np.ndarray], int]:
    if ingest_rate and ingest_rate < rate:
        return [_lowpass_decimate(c, rate, ingest_rate).astype(np.float32) for c in channels], ingest_rate
    return [c.astype(np.float32, copy=False) for c in channels], rate


class _PairDecimator:
    """Successive ×2 averaging with leftover samples so block reads stay continuous."""

    def __init__(self, n_ch: int, in_rate: float, out_rate: float) -> None:
        self.n_ch = n_ch
        self.out_rate = float(out_rate)
        stages = 0
        rate = float(in_rate)
        while rate / 2 >= out_rate * 0.98 and rate > out_rate:
            rate /= 2
            stages += 1
        self.stages = stages
        self.stage_rate = rate
        self.hold: list[list[float | None]] = [[None] * stages for _ in range(n_ch)]
        self.acc: list[list[np.ndarray]] = [[] for _ in range(n_ch)]

    def push(self, planar: list[np.ndarray]) -> None:
        for c in range(self.n_ch):
            x = planar[c].astype(np.float64, copy=False)
            for s in range(self.stages):
                h = self.hold[c][s]
                if h is not None:
                    x = np.concatenate(([h], x))
                if x.size % 2:
                    self.hold[c][s] = float(x[-1])
                    x = x[:-1]
                else:
                    self.hold[c][s] = None
                if x.size == 0:
                    break
                x = 0.5 * (x[0::2] + x[1::2])
            if x.size:
                self.acc[c].append(np.asarray(x, dtype=np.float64))

    def finish(self) -> tuple[list[np.ndarray], int]:
        out: list[np.ndarray] = []
        for c in range(self.n_ch):
            if self.acc[c]:
                x = np.concatenate(self.acc[c])
            else:
                x = np.zeros(0, dtype=np.float64)
            out.append(resample_linear(x, self.stage_rate, self.out_rate).astype(np.float32))
        return out, int(round(self.out_rate))


def decode_soundfile(
    path: str,
    max_seconds: float | None = None,
    ingest_rate: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[np.ndarray], int, WavInfo]:
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sr = int(f.samplerate)
        n_ch = int(f.channels)
        frames = int(len(f))
        if max_seconds and max_seconds > 0:
            frames = min(frames, int(max_seconds * sr))
        bits, kind = _subtype_bits(getattr(f, "subtype", "") or "")
        container = sniff_container(path)
        if container in ("WAV", "AUDIO") and getattr(f, "format", ""):
            container = str(f.format).replace("_", " ")
        info = _make_info(sr, n_ch, bits, kind, frames, container)
        target = ingest_rate if ingest_rate and ingest_rate < sr else None
        block = 65536
        done = 0
        if target:
            dec = _PairDecimator(n_ch, sr, target)
            while done < frames:
                take = min(block, frames - done)
                data = f.read(take, dtype="float32", always_2d=True)
                if data.size == 0:
                    break
                dec.push([data[:, c] for c in range(n_ch)])
                done += data.shape[0]
                if on_progress:
                    on_progress(done / max(1, frames))
            channels, rate = dec.finish()
            if on_progress:
                on_progress(1.0)
            return channels, rate, info

        channels = [np.zeros(frames, dtype=np.float32) for _ in range(n_ch)]
        written = 0
        while written < frames:
            take = min(block, frames - written)
            data = f.read(take, dtype="float32", always_2d=True)
            if data.size == 0:
                break
            n = data.shape[0]
            for c in range(n_ch):
                channels[c][written : written + n] = data[:, c]
            written += n
            if on_progress:
                on_progress(written / max(1, frames))
        return [c[:written] for c in channels], sr, info


def decode_miniaudio(
    path: str,
    max_seconds: float | None = None,
    ingest_rate: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[np.ndarray], int, WavInfo]:
    import miniaudio

    if on_progress:
        on_progress(0.15)
    container = sniff_container(path)
    if container == "FLAC":
        decoded = miniaudio.flac_read_file_f32(path)
    else:
        info = miniaudio.get_file_info(path)
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=info.nchannels,
            sample_rate=info.sample_rate,
        )
    if on_progress:
        on_progress(0.7)
    sr = int(decoded.sample_rate)
    n_ch = int(decoded.nchannels)
    samples = np.frombuffer(decoded.samples, dtype=np.float32)
    if samples.size % max(1, n_ch) != 0 or samples.size == 0:
        samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
    frames = samples.size // max(1, n_ch)
    if max_seconds and max_seconds > 0:
        frames = min(frames, int(max_seconds * sr))
    interleaved = samples[: frames * n_ch].reshape(frames, n_ch)
    channels = [interleaved[:, c].copy() for c in range(n_ch)]
    channels, rate = _apply_ingest(channels, sr, ingest_rate)
    if on_progress:
        on_progress(1.0)
    return channels, rate, _make_info(sr, n_ch, 32, "float", frames, container)


def decode_ffmpeg(
    path: str,
    max_seconds: float | None = None,
    ingest_rate: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[np.ndarray], int, WavInfo]:
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is not on PATH")
    n_ch = 1
    sr = 48000
    bits = 16
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels,sample_rate,bits_per_raw_sample",
                "-of",
                "default=nw=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in probe.stdout.splitlines():
            if line.startswith("channels="):
                n_ch = int(line.split("=", 1)[1] or 1)
            elif line.startswith("sample_rate="):
                sr = int(line.split("=", 1)[1] or 48000)
            elif line.startswith("bits_per_raw_sample="):
                try:
                    bits = int(line.split("=", 1)[1] or 16)
                except ValueError:
                    bits = 16
    out_rate = ingest_rate if ingest_rate and ingest_rate < sr else sr
    cmd = [ffmpeg, "-v", "error", "-i", path]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(n_ch), "-ar", str(out_rate), "pipe:1"]
    if on_progress:
        on_progress(0.2)
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[:400]
        raise ValueError(f"ffmpeg could not decode this file. {err}")
    raw = np.frombuffer(proc.stdout, dtype="<f4")
    frames = raw.size // max(1, n_ch)
    interleaved = raw[: frames * n_ch].reshape(frames, n_ch)
    channels = [interleaved[:, c].copy() for c in range(n_ch)]
    if on_progress:
        on_progress(1.0)
    return channels, int(out_rate), _make_info(sr, n_ch, bits, "pcm", frames, sniff_container(path))


def decode_audio(
    path: str,
    max_seconds: float | None = None,
    ingest_rate: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[list[np.ndarray], int, WavInfo]:
    """Decode WAV/RF64 natively; FLAC and other containers via libsndfile / miniaudio / ffmpeg."""
    container = sniff_container(path)
    ext = os.path.splitext(path)[1].lower()
    errors: list[str] = []

    if container in ("WAV", "RF64 WAV") or ext in WAV_EXT:
        try:
            return decode_wav(path, max_seconds, ingest_rate, on_progress)
        except Exception as exc:
            errors.append(f"WAV parser: {exc}")

    for name, fn in (
        ("FLAC/libsndfile", decode_soundfile),
        ("miniaudio", decode_miniaudio),
        ("ffmpeg", decode_ffmpeg),
    ):
        try:
            return fn(path, max_seconds, ingest_rate, on_progress)
        except ImportError as exc:
            errors.append(f"{name} not installed ({exc})")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    lossy = ext in LOSSY_EXT
    hint = (
        " This looks like a lossy file. Low-level spectral content, phase, and extreme low-frequency information may already be gone. Prefer the original WAV/FLAC."
        if lossy
        else " Use WAV, RF64, FLAC, or AIFF from the recorder when you can."
    )
    raise ValueError("Could not decode this recording. " + " | ".join(errors[-3:]) + hint)


# ---------------------------------------------------------------------------
# Filter bank
# ---------------------------------------------------------------------------


def _raised_cosine(t: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(t, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * x)


def band_responses(fft_size: int, sample_rate: float, bands: list[Band]) -> list[np.ndarray]:
    n_bins = fft_size // 2
    responses = [np.zeros(n_bins + 1, dtype=np.float64) for _ in bands]
    if not bands:
        return responses

    def xover_for(i: int, edge: str) -> float:
        b = bands[i]
        width = b.hi - b.lo
        neighbor = width
        if edge == "lo" and i > 0:
            neighbor = min(neighbor, bands[i - 1].hi - bands[i - 1].lo)
        if edge == "hi" and i < len(bands) - 1:
            neighbor = min(neighbor, bands[i + 1].hi - bands[i + 1].lo)
        bin_hz = sample_rate / fft_size
        return max(bin_hz * 1.5, min(width, neighbor) * 0.18)

    k = np.arange(n_bins + 1, dtype=np.float64)
    freqs = (k * sample_rate) / fft_size
    for i, b in enumerate(bands):
        x_lo = xover_for(i, "lo")
        x_hi = xover_for(i, "hi")
        g = np.zeros_like(freqs)
        inside = (freqs >= b.lo - x_lo) & (freqs <= b.hi + x_hi)
        passband = (freqs >= b.lo + x_lo) & (freqs <= b.hi - x_hi)
        lo_x = freqs < b.lo + x_lo
        g[inside & passband] = 1.0
        mask_lo = inside & lo_x & ~passband
        g[mask_lo] = _raised_cosine((freqs[mask_lo] - (b.lo - x_lo)) / (2 * x_lo))
        mask_hi = inside & ~lo_x & ~passband
        g[mask_hi] = 1.0 - _raised_cosine((freqs[mask_hi] - (b.hi - x_hi)) / (2 * x_hi))
        responses[i] = g
    return responses


def extract_bands(
    signal: np.ndarray,
    sample_rate: float,
    bands: list[Band],
    on_progress: Callable[[float], None] | None = None,
) -> list[np.ndarray]:
    n = int(signal.size)
    outputs = [np.zeros(n, dtype=np.float64) for _ in bands]
    if n == 0 or not bands:
        return outputs

    fft_size = min(
        8192,
        max(4096, 1 << math.ceil(math.log2(max(256, math.floor(sample_rate * 16))))),
    )
    hop = fft_size // 2
    window = hann(fft_size)
    responses = band_responses(fft_size, sample_rate, bands)
    pad = fft_size
    total = n + 2 * pad
    n_frames = math.ceil((total - fft_size) / hop) + 1
    x = signal.astype(np.float64, copy=False)

    for frame in range(n_frames):
        origin = frame * hop - pad
        src_start = max(0, origin)
        src_end = min(n, origin + fft_size)
        frame_buf = np.zeros(fft_size, dtype=np.float64)
        if src_end > src_start:
            dst = src_start - origin
            frame_buf[dst : dst + (src_end - src_start)] = x[src_start:src_end]
        spec = np.fft.rfft(frame_buf * window)
        for b, resp in enumerate(responses):
            time = np.fft.irfft(spec * resp, n=fft_size)
            if src_end > src_start:
                dst = src_start - origin
                outputs[b][src_start:src_end] += time[dst : dst + (src_end - src_start)]
        if on_progress and (frame & 15) == 0:
            on_progress(frame / n_frames)
    if on_progress:
        on_progress(1.0)
    return outputs


def downsample_to(signal: np.ndarray, in_rate: float, out_rate: float) -> np.ndarray:
    if signal.size == 0:
        return signal.astype(np.float64, copy=True)
    if abs(in_rate - out_rate) < 1e-6:
        return signal.astype(np.float64, copy=True)
    if out_rate > in_rate:
        return resample_linear(signal, in_rate, out_rate)
    return _lowpass_decimate(signal, in_rate, out_rate)


def time_compress(
    signal: np.ndarray, working_rate: float, speed: float, output_rate: float
) -> np.ndarray:
    return resample_linear(signal, working_rate * speed, output_rate)


def reconstruction_error_db(original: np.ndarray, bands: list[np.ndarray]) -> float:
    if original.size == 0:
        return 0.0
    summed = np.zeros_like(original, dtype=np.float64)
    for b in bands:
        summed += b[: original.size]
    err = float(np.sum((summed - original) ** 2))
    ref = float(np.sum(original.astype(np.float64) ** 2))
    if ref <= 1e-30:
        return 0.0
    return 10.0 * math.log10(err / ref)


# ---------------------------------------------------------------------------
# Gain / mix
# ---------------------------------------------------------------------------


def measure_band(signal: np.ndarray, strongest_rms: float, noise_floor_dbfs: float) -> BandMeasurement:
    peak = peak_abs(signal)
    r = rms(signal)
    robust = percentile_abs(signal, 0.9)
    peak_d = dbfs(peak)
    rms_d = dbfs(r)
    robust_d = dbfs(robust)
    crest = peak_d - rms_d
    relative = dbfs(r) - dbfs(max(strongest_rms, 1e-20))

    win = max(32, signal.size // 200)
    active = 0
    windows = 0
    thresh = max(r * 0.25, from_dbfs(noise_floor_dbfs + 6))
    i = 0
    while i + win <= signal.size:
        windows += 1
        if rms(signal, i, i + win) > thresh:
            active += 1
        i += win
    active_fraction = 0.0 if windows == 0 else active / windows
    low_conf = rms_d < noise_floor_dbfs + 8 or relative < -60 or active_fraction < 0.02
    if relative >= -6:
        classification = "very-strong"
    elif relative >= -14:
        classification = "strong"
    elif relative >= -26:
        classification = "moderate"
    elif relative >= -40:
        classification = "weak"
    else:
        classification = "very-weak"
    return BandMeasurement(
        peak_dbfs=peak_d,
        rms_dbfs=rms_d,
        robust_rms_dbfs=robust_d,
        crest_db=crest,
        active_fraction=active_fraction,
        relative_db=relative,
        classification=classification,
        low_confidence=low_conf,
        suggested_gain=0.0,
        max_auto_gain=0.0,
    )


def apply_gain_mode(measurements: list[BandMeasurement], mode: GainMode) -> list[BandMeasurement]:
    out = []
    for m in measurements:
        if mode in ("preserve", "manual"):
            m.suggested_gain = 0.0
            m.max_auto_gain = 0.0 if mode == "preserve" else 48.0
            out.append(m)
            continue
        cap = GAIN_CAPS[mode]
        k = GAIN_K[mode]
        g = min(cap, max(0.0, -m.relative_db * k))
        max_auto = min(cap, cap * 0.55 + 6) if m.low_confidence else cap
        if m.low_confidence:
            g = min(g, max_auto)
        if m.classification == "very-weak" and m.active_fraction < 0.01:
            g = min(g, 12.0)
        m.suggested_gain = g
        m.max_auto_gain = max_auto
        out.append(m)
    return out


def mix_bands(
    bands: list[np.ndarray],
    gains_db: list[float],
    mute: list[bool],
    solo: list[bool],
) -> np.ndarray:
    n = bands[0].size if bands else 0
    out = np.zeros(n, dtype=np.float64)
    any_solo = any(solo)
    for b, src in enumerate(bands):
        if b < len(mute) and mute[b]:
            continue
        if any_solo and not (b < len(solo) and solo[b]):
            continue
        g = from_dbfs(gains_db[b] if b < len(gains_db) else 0.0)
        out += src[:n] * g
    return out


def master_trim(signal: np.ndarray, headroom_db: float = HEADROOM_DB) -> tuple[float, float]:
    peak = peak_abs(signal)
    peak_d = dbfs(peak)
    if peak_d <= headroom_db:
        return 0.0, peak_d
    trim = headroom_db - peak_d
    signal *= from_dbfs(trim)
    return trim, headroom_db


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


def survey_spectrum(signal: np.ndarray, sample_rate: float, fft_size: int = 4096) -> SpectrumSurvey:
    n_bins = fft_size // 2
    freqs = (np.arange(n_bins, dtype=np.float64) * sample_rate) / fft_size
    mean_e = np.zeros(n_bins, dtype=np.float64)
    peak_e = np.zeros(n_bins, dtype=np.float64)
    window = hann(fft_size)
    hop = fft_size // 2
    frames = 0
    x = signal.astype(np.float64, copy=False)
    origin = 0
    while origin + fft_size <= x.size:
        spec = np.fft.rfft(x[origin : origin + fft_size] * window)
        mag = np.abs(spec[:n_bins]) * (2.0 / fft_size)
        e = mag * mag
        mean_e += e
        peak_e = np.maximum(peak_e, e)
        frames += 1
        origin += hop
    denom = max(1, frames)
    mean_db = np.array([dbfs(math.sqrt(v / denom)) for v in mean_e], dtype=np.float64)
    peak_db = np.array([dbfs(math.sqrt(v)) for v in peak_e], dtype=np.float64)
    return SpectrumSurvey(freqs=freqs, mean_dbfs=mean_db, peak_dbfs=peak_db)


def spectrogram(
    signal: np.ndarray,
    sample_rate: float,
    f_min: float,
    f_max: float,
    width: int,
    height: int,
) -> np.ndarray:
    fft_size = 1024
    window = hann(fft_size)
    hop = max(1, (signal.size - fft_size) // max(1, width - 1))
    img = np.zeros((height, width), dtype=np.float64)
    n_bins = fft_size // 2
    log_min = math.log(max(f_min, sample_rate / fft_size))
    log_max = math.log(max(f_max, log_min + 1e-9))
    x = signal.astype(np.float64, copy=False)
    for col in range(width):
        origin = min(max(0, col * hop), max(0, x.size - fft_size))
        if x.size >= fft_size:
            spec = np.fft.rfft(x[origin : origin + fft_size] * window)
        else:
            spec = np.zeros(n_bins + 1)
        mag = np.abs(spec)
        for row in range(height):
            frac = 1 - row / max(1, height - 1)
            f = math.exp(log_min + frac * (log_max - log_min))
            k = (f * fft_size) / sample_rate
            k0 = int(min(n_bins - 2, max(0, math.floor(k))))
            t = k - k0
            m = mag[k0] + (mag[k0 + 1] - mag[k0]) * t
            img[row, col] = dbfs((2 * m) / fft_size)
    return img


# ---------------------------------------------------------------------------
# Synthetic
# ---------------------------------------------------------------------------


def _add_sine(dst: np.ndarray, sr: float, hz: float, amp: float, phase: float = 0.0) -> None:
    i = np.arange(dst.size, dtype=np.float64)
    dst += amp * np.sin((2 * math.pi * hz / sr) * i + phase)


def _add_impulse(dst: np.ndarray, sr: float, at_sec: float, amp: float, tau_sec: float) -> None:
    start = int(at_sec * sr)
    if start < 0 or start >= dst.size:
        return
    tau = tau_sec * sr
    for i in range(start, dst.size):
        env = math.exp(-(i - start) / tau)
        if env < 1e-5:
            break
        first = 1.0 if (i - start) == 0 else -math.exp(-(i - start) / (tau * 0.15))
        dst[i] += amp * env * first


def _add_burst(dst: np.ndarray, sr: float, hz: float, amp: float, t0: float, dur: float) -> None:
    a = int(t0 * sr)
    b = min(dst.size, int((t0 + dur) * sr))
    fade = max(1, int(0.4 * sr))
    w = 2 * math.pi * hz / sr
    for i in range(a, b):
        rel = i - a
        from_end = b - 1 - i
        env = 1.0
        if rel < fade:
            env = rel / fade
        if from_end < fade:
            env = min(env, from_end / fade)
        dst[i] += amp * env * math.sin(w * i)


def _noise(n: int, amp: float, seed: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    s = seed & 0xFFFFFFFF
    lp = 0.0
    for i in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        w = (s / 0xFFFFFFFF) * 2 - 1
        lp = lp * 0.995 + w * 0.005
        out[i] = lp * amp * 40
    return out


def generate_synthetic(kind: SyntheticKind) -> tuple[list[np.ndarray], int, SyntheticSpec]:
    spec = next(s for s in SYNTHETICS if s.id == kind)
    sr = spec.sample_rate
    n = int(spec.duration * sr)
    ch1 = np.zeros(n, dtype=np.float64)
    ch2 = np.zeros(n, dtype=np.float64)

    if kind == "calibration":
        _add_sine(ch1, sr, 0.1, 0.08, 0)
        _add_sine(ch2, sr, 0.1, 0.08, 0.1)
        _add_sine(ch1, sr, 0.5, 0.08, 0.2)
        _add_sine(ch2, sr, 0.5, 0.08, 0.3)
        _add_sine(ch1, sr, 3.7, 0.25, 0)
        _add_sine(ch2, sr, 3.7, 0.22, 0.4)
        _add_sine(ch1, sr, 19.0, 0.45, 0)
        _add_sine(ch2, sr, 19.0, 0.4, 0.15)
        _add_impulse(ch1, sr, 20, 0.9, 0.8)
        _add_impulse(ch2, sr, 20.04, 0.55, 0.8)
        ch1 += _noise(n, 0.0004, 1)
        ch2 += _noise(n, 0.0004, 2)
    elif kind == "sweep":
        f0, f1 = 0.1, 30.0
        t = np.arange(n, dtype=np.float64) / sr
        frac = np.arange(n, dtype=np.float64) / max(1, n - 1)
        phase = (2 * math.pi * (f0 * t * ((f1 / f0) ** frac - 1)) / math.log(f1 / f0))
        ch1[:] = 0.2 * np.sin(phase)
        ch2[:] = 0.2 * np.sin(phase + 0.08)
    else:
        t = np.arange(n, dtype=np.float64) / sr
        am19 = 1 + 0.12 * np.sin(2 * math.pi * 0.003 * t)
        ch1 += 0.12 * am19 * np.sin(2 * math.pi * 19.32 * t)
        ch2 += 0.09 * am19 * np.sin(2 * math.pi * 19.32 * t + 0.22)
        ch1 += 0.04 * np.sin(2 * math.pi * 14.91 * t + 0.4)
        ch2 += 0.036 * np.sin(2 * math.pi * 14.91 * t + 0.51)
        ch1 += 0.035 * np.sin(2 * math.pi * 3.82 * t)
        ch2 += 0.018 * np.sin(2 * math.pi * 3.82 * t + 0.3)
        ch2 += 0.03 * np.sin(2 * math.pi * 8.41 * t + 1.1)
        ch1 += 0.008 * np.sin(2 * math.pi * 0.23 * t)
        ch2 += 0.005 * np.sin(2 * math.pi * 0.23 * t + 0.6)
        ch1 += 0.006 * np.sin(2 * math.pi * 0.47 * t + 0.2)
        ch2 += 0.0055 * np.sin(2 * math.pi * 0.47 * t + 0.9)
        ch2 += 0.002 * np.sin(2 * math.pi * 0.008 * t)
        ch1 += _noise(n, 0.0005, 7)
        ch2 += _noise(n, 0.00045, 9)
        dur = spec.duration
        t_burst = 90.0
        while t_burst < dur - 30:
            _add_burst(ch1, sr, 2.71, 0.05, t_burst, 8)
            _add_burst(ch2, sr, 2.71, 0.02, t_burst + 0.08, 8)
            t_burst += 180 + ((t_burst * 13) % 70)
        for th in (120, 400, 900, 1400, 2100, 3100, 4300, 5100, 6200, 6900):
            if th < dur - 2:
                _add_impulse(ch1, sr, th, 0.7, 1.2)
                _add_impulse(ch2, sr, th + 0.03, 0.35, 1.2)

    return [ch1.astype(np.float32), ch2.astype(np.float32)], sr, spec


def source_from_synthetic(
    kind: SyntheticKind, channels: list[np.ndarray], sample_rate: int
) -> SourceInfo:
    spec = next(s for s in SYNTHETICS if s.id == kind)
    peaks = [dbfs(peak_abs(c)) for c in channels]
    rmses = [dbfs(rms(c)) for c in channels]
    return SourceInfo(
        kind="synthetic",
        name=spec.title,
        format="SYNTH",
        duration=channels[0].size / sample_rate,
        channels=len(channels),
        sample_rate=sample_rate,
        sample_format="32-bit float",
        bytes=int(sum(c.nbytes for c in channels)),
        peaks=peaks,
        rms=rmses,
        clipping=any(p >= -0.1 for p in peaks),
    )


def source_from_file(
    path: str, channels: list[np.ndarray], sample_rate: int, info: WavInfo
) -> SourceInfo:
    peaks = [dbfs(peak_abs(c)) for c in channels]
    rmses = [dbfs(rms(c)) for c in channels]
    return SourceInfo(
        kind="file",
        name=os.path.basename(path),
        format=info.container or ("RF64 WAV" if info.is_rf64 else "WAV"),
        duration=channels[0].size / sample_rate if sample_rate else 0,
        channels=len(channels),
        sample_rate=sample_rate,
        sample_format=f"{info.bits_per_sample}-bit {info.fmt}",
        bytes=os.path.getsize(path),
        peaks=peaks,
        rms=rmses,
        clipping=any(p >= -0.05 for p in peaks),
        path=path,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _integrity(signal: np.ndarray) -> ChannelIntegrity:
    finite = np.isfinite(signal)
    nans = int(np.isnan(signal).sum())
    infs = int(np.isinf(signal).sum())
    clipped = int((np.abs(signal[finite]) >= 0.999).sum()) if finite.any() else 0
    peak = peak_abs(signal)
    r = rms(signal)
    return ChannelIntegrity(
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(r),
        dc_bias=mean(signal),
        clipped_samples=clipped,
        nan_samples=nans,
        inf_samples=infs,
        silent=r < 1e-8,
    )


def process_channels(
    channels: list[np.ndarray],
    input_rate: float,
    config: ProcessConfig,
    on_progress: ProgressFn | None = None,
) -> ProcessResult:
    started = datetime.now(timezone.utc)
    log: list[str] = []

    def stamp(msg: str) -> None:
        log.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}")

    def report(stage_index: int, fraction: float, detail: str) -> None:
        if on_progress:
            on_progress(
                {
                    "stage": STAGES[stage_index],
                    "stage_index": stage_index,
                    "stage_count": len(STAGES),
                    "fraction": (stage_index + fraction) / len(STAGES),
                    "detail": detail,
                    "elapsed_ms": (datetime.now(timezone.utc) - started).total_seconds() * 1000,
                }
            )

    report(0, 0, "Checking input")
    if not channels:
        raise ValueError("No channels to process.")
    lengths = {c.size for c in channels}
    if len(lengths) != 1:
        raise ValueError("Channels do not contain the same sample count. Synchronization is broken.")
    nyq = input_rate / 2
    if config.f_max > nyq:
        raise ValueError(
            f"The input Nyquist frequency is {nyq} Hz. Frequencies above {nyq} Hz were never recorded."
        )
    stamp(f"Input {len(channels)} ch × {channels[0].size} samples @ {input_rate} Hz")

    working = [c.astype(np.float64, copy=False) for c in channels]
    working_rate = config.working_rate
    report(1, 0, f"Working rate {working_rate} Hz")
    if abs(input_rate - working_rate) > 1e-6:
        nxt = []
        for i, ch in enumerate(working):
            report(1, i / len(working), f"Downsampling channel {i + 1}")
            nxt.append(downsample_to(ch, input_rate, working_rate))
        working = nxt
    stamp(f"Working representation {working[0].size} samples @ {working_rate} Hz")

    if config.test_duration and config.test_duration > 0:
        keep = min(working[0].size, int(config.test_duration * working_rate))
        working = [ch[:keep] for ch in working]
        stamp(f"Test mode: first {config.test_duration}s ({keep} samples)")

    integ = [_integrity(ch) for ch in working]
    for i, g in enumerate(integ):
        stamp(
            f"Channel {i + 1} peak {g.peak_dbfs:.1f} dBFS RMS {g.rms_dbfs:.1f} dBFS DC {g.dc_bias:.2e}"
        )
        if g.clipped_samples:
            stamp(f"Channel {i + 1} clipping samples: {g.clipped_samples}")
        if g.silent:
            stamp(f"Channel {i + 1} appears silent")

    live = [b for b in config.bands if b.enabled and b.hi > b.lo]
    if not live:
        raise ValueError("No enabled bands.")

    report(2, 0, "Surveying spectrum")
    spectrum = []
    for i, ch in enumerate(working):
        spectrum.append(survey_spectrum(ch, working_rate))
        report(2, (i + 1) / len(working), f"Spectrum channel {i + 1}")

    band_audio: list[list[np.ndarray]] = []
    reconstruction_db: list[float] = []
    report(3, 0, "Extracting filter bank")
    for i, ch in enumerate(working):
        extracted = extract_bands(
            ch,
            working_rate,
            live,
            lambda f, i=i: report(3, (i + f) / len(working), f"Channel {i + 1} bands"),
        )
        band_audio.append(extracted)
        full = extract_bands(
            ch,
            working_rate,
            [Band(config.f_min, config.f_max, "range", True)],
        )
        reconstruction_db.append(reconstruction_error_db(full[0], extracted))
        stamp(f"Channel {i + 1} reconstruction residual {reconstruction_db[-1]:.1f} dB")

    report(4, 0, "Measuring bands")
    measurements: list[list[BandMeasurement]] = []
    for ch_bands in band_audio:
        strengths = [rms(s) for s in ch_bands]
        strongest = max(strengths + [1e-12])
        noise_floor = min(dbfs(rms(s)) for s in ch_bands) - 6
        raw = [measure_band(s, strongest, noise_floor) for s in ch_bands]
        mode: GainMode = "medium" if config.gain_mode == "manual" else config.gain_mode
        measurements.append(apply_gain_mode(raw, mode))

    suggested = [[m.suggested_gain for m in ms] for ms in measurements]
    stamp(f"Gain mode {config.gain_mode}")
    report(5, 1, "Preview caches ready")
    report(6, 1, "Done")
    stamp("Processing complete")

    return ProcessResult(
        working_rate=int(working_rate),
        channels=working,
        band_audio=band_audio,
        measurements=measurements,
        suggested_gains=suggested,
        spectrum=spectrum,
        reconstruction_db=reconstruction_db,
        integrity=integ,
        log=log,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _csv(s: str) -> str:
    if any(c in s for c in '",\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def analysis_csv(
    source: SourceInfo,
    config: ProcessConfig,
    measurements: list[list[BandMeasurement]],
    bands: list[Band],
) -> str:
    lines = [
        "channel,channel_name,band,lo_hz,hi_hz,peak_dbfs,rms_dbfs,robust_rms_dbfs,crest_db,relative_db,active_fraction,classification,low_confidence,suggested_gain_db"
    ]
    for ci, ms in enumerate(measurements):
        for bi, m in enumerate(ms):
            b = bands[bi]
            lines.append(
                ",".join(
                    [
                        str(ci + 1),
                        _csv(config.channel_names[ci] if ci < len(config.channel_names) else f"Channel {ci + 1}"),
                        _csv(b.name),
                        str(b.lo),
                        str(b.hi),
                        f"{m.peak_dbfs:.3f}",
                        f"{m.rms_dbfs:.3f}",
                        f"{m.robust_rms_dbfs:.3f}",
                        f"{m.crest_db:.3f}",
                        f"{m.relative_db:.3f}",
                        f"{m.active_fraction:.4f}",
                        m.classification,
                        "1" if m.low_confidence else "0",
                        f"{m.suggested_gain:.2f}",
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def processing_log_text(
    source: SourceInfo,
    config: ProcessConfig,
    integrity: list[ChannelIntegrity],
    measurements: list[list[BandMeasurement]],
    bands: list[Band],
    reconstruction_db: list[float],
    gains: list[list[float]],
    log: list[str],
) -> str:
    lines = [
        f"{APP_NAME} {APP_VERSION}  DSP {DSP_VERSION}",
        f"Project source: {source.name}",
        f"Input: {source.format}  {source.channels} ch  {source.sample_rate} Hz  {source.sample_format}",
        f"Duration: {format_duration(source.duration)}",
        f"Internal analysis rate: {config.working_rate} Hz",
        f"Main speed: {config.speed}×",
        f"Source range: {format_hz(config.f_min)} – {format_hz(config.f_max)}",
        f"Translated range: {format_hz(config.f_min * config.speed)} – {format_hz(config.f_max * config.speed)}",
        f"Phase policy: {config.phase_policy}",
        f"Gain mode: {config.gain_mode}",
        "",
    ]
    for i, g in enumerate(integrity):
        name = config.channel_names[i] if i < len(config.channel_names) else f"Channel {i + 1}"
        lines += [
            f"INPUT CHECK  {name}",
            f"  Peak {format_db(g.peak_dbfs)}   RMS {format_db(g.rms_dbfs)}   DC {g.dc_bias:.2e}",
            f"  Clipped samples: {g.clipped_samples}   NaN: {g.nan_samples}   Inf: {g.inf_samples}   Silent: {'yes' if g.silent else 'no'}",
            f"  Reconstruction residual: {format_db(reconstruction_db[i] if i < len(reconstruction_db) else 0)}",
            "",
            "  Band                 RMS      Relative   Applied gain   Class",
        ]
        for bi, m in enumerate(measurements[i] if i < len(measurements) else []):
            b = bands[bi]
            gain = gains[i][bi] if i < len(gains) and bi < len(gains[i]) else m.suggested_gain
            flag = "  LOW CONFIDENCE" if m.low_confidence else ""
            lines.append(
                f"  {format_band(b.lo, b.hi):<18} {format_db(m.rms_dbfs):>9}  {format_db(m.relative_db):>9}  {format_db(gain):>12}   {m.classification}{flag}"
            )
        lines.append("")
    lines.append("SESSION LOG")
    for row in log:
        lines.append(f"  {row}")
    lines += [
        "",
        "Preserve first. Measure second. Translate third. Enhance only for listening.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _peak_near(signal: np.ndarray, sample_rate: float, target_hz: float, window_hz: float) -> tuple[float, float]:
    fft_size = 1 << math.ceil(math.log2(min(signal.size, 16384)))
    w = hann(min(fft_size, signal.size))
    buf = np.zeros(fft_size, dtype=np.float64)
    n = min(w.size, signal.size)
    buf[:n] = signal[:n] * w[:n]
    spec = np.fft.rfft(buf)
    k0 = max(1, int(((target_hz - window_hz) * fft_size) / sample_rate))
    k1 = min(fft_size // 2 - 1, int(math.ceil(((target_hz + window_hz) * fft_size) / sample_rate)))
    mag = np.abs(spec)
    sl = mag[k0 : k1 + 1]
    if sl.size == 0:
        return target_hz, SILENCE_DB
    best_k = int(k0 + np.argmax(sl))
    best = float(mag[best_k])
    return (best_k * sample_rate) / fft_size, dbfs((2 * best) / fft_size)


def run_self_test() -> list[dict]:
    channels, sample_rate, _ = generate_synthetic("calibration")
    bands = preset_bands("coarse", 0.1, 30)
    config = ProcessConfig(
        bands=bands,
        f_min=0.1,
        f_max=30,
        working_rate=sample_rate,
        speed=200,
        output_rate=48000,
        gain_mode="preserve",
        channel_names=["Mic 1", "Mic 2"],
    )
    result = process_channels(channels, sample_rate, config)
    cases = []

    def add(name: str, passed: bool, detail: str) -> None:
        cases.append({"name": name, "pass": passed, "detail": detail})

    add(
        "Channel lengths match",
        result.channels[0].size == result.channels[1].size,
        f"{result.channels[0].size} vs {result.channels[1].size}",
    )
    recon = result.reconstruction_db[0]
    add("Filter-bank reconstruction residual < −20 dB", recon < -20, f"{recon:.1f} dB")

    compressed = time_compress(result.channels[0], result.working_rate, 200, 48000)
    for name, hz in [
        ("0.10 Hz → 20 Hz", 20),
        ("0.50 Hz → 100 Hz", 100),
        ("3.70 Hz → 740 Hz", 740),
        ("19.00 Hz → 3800 Hz", 3800),
    ]:
        found_hz, _ = _peak_near(compressed, 48000, hz, max(8, hz * 0.04))
        err = abs(found_hz - hz) / hz
        add(name, err < 0.08, f"peak {found_hz:.1f} Hz ({err * 100:.1f}% error)")

    peak = dbfs(peak_abs(compressed))
    add("Compressed output is finite and non-silent", math.isfinite(peak) and peak > -60, f"peak {peak:.1f} dBFS")

    rel = result.measurements[0]
    strongest = max(rel, key=lambda m: m.rms_dbfs)
    weakest = min(rel, key=lambda m: m.rms_dbfs)
    add(
        "19–22 Hz band is stronger than 0.1–0.5 Hz",
        strongest.rms_dbfs > weakest.rms_dbfs + 6,
        f"strongest {strongest.rms_dbfs:.1f} vs weakest {weakest.rms_dbfs:.1f} dBFS",
    )
    add(
        "No NaN / Inf in working audio",
        all(g.nan_samples == 0 and g.inf_samples == 0 for g in result.integrity),
        "; ".join(f"ch{i + 1} nan={g.nan_samples} inf={g.inf_samples}" for i, g in enumerate(result.integrity)),
    )
    return cases


if __name__ == "__main__":
    print(f"{APP_NAME} {APP_VERSION} DSP {DSP_VERSION} self-test")
    for case in run_self_test():
        mark = "PASS" if case["pass"] else "FAIL"
        print(f"  [{mark}] {case['name']}  —  {case['detail']}")
    print(json.dumps({"ok": True}, indent=2))
