"""Pure-python audio duration probing and validation for STT pipelines.

No external dependencies -- WAV via stdlib `wave`, MP3 via frame/Xing parsing,
M4A/MP4 via atom walking. Returns None for formats we don't understand
(webm, ogg, flac, opaque blobs); validation still passes those through to whisper.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Optional, Tuple, Union

AudioSource = Union[bytes, str, "os.PathLike[str]", BinaryIO]

WHISPER_MAX_BYTES = 25 * 1024 * 1024
WHISPER_MAX_DURATION_S = 1500.0
WHISPER_FORMATS = ("wav", "mp3", "m4a", "mp4", "mpeg", "mpga", "webm", "flac", "ogg")


# --- format detection ---


def _read_all(source: AudioSource) -> Tuple[bytes, Optional[str]]:
    """Return (bytes, filename_hint)."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        return path.read_bytes(), path.name
    pos = source.tell() if hasattr(source, "tell") else None
    data = source.read()
    if pos is not None and hasattr(source, "seek"):
        source.seek(pos)
    name = getattr(source, "name", None)
    if isinstance(name, str):
        name = os.path.basename(name)
    else:
        name = None
    return data, name


def detect_format(data: bytes, filename: Optional[str] = None) -> Optional[str]:
    """Detect by magic bytes; fall back to extension hint."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        # ISO BMFF -- treat M4A and MP4 the same
        return "m4a"
    if len(data) >= 4 and data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    if len(data) >= 4 and data[:4] == b"fLaC":
        return "flac"
    if len(data) >= 4 and data[:4] == b"OggS":
        return "ogg"
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"

    # extension fallback
    if filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext in WHISPER_FORMATS:
            return "m4a" if ext == "mp4" else ext
    return None


# --- per-format duration probes ---


def _wav_duration(data: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate if rate else None
    except wave.Error:
        return None


# MPEG audio frame sample-rate / bitrate tables.
_MPEG_VERSION = {0: 2.5, 2: 2, 3: 1}  # bits 19-20
_MPEG_LAYER = {1: 3, 2: 2, 3: 1}      # bits 17-18

# bitrate kbps tables: [version][layer-1][index 1..14]
_BITRATE_V1 = {
    1: [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    2: [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    3: [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
}
_BITRATE_V2 = {
    1: [32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    2: [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    3: [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}
_SAMPLE_RATE = {
    1: [44100, 48000, 32000],
    2: [22050, 24000, 16000],
    2.5: [11025, 12000, 8000],
}
_SAMPLES_PER_FRAME = {
    (1, 1): 384,
    (1, 2): 1152,
    (1, 3): 1152,
    (2, 1): 384,
    (2, 2): 1152,
    (2, 3): 576,
    (2.5, 1): 384,
    (2.5, 2): 1152,
    (2.5, 3): 576,
}


def _id3v2_size(data: bytes) -> int:
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    # synchsafe 4-byte int at bytes 6..10
    b = data[6:10]
    size = (b[0] << 21) | (b[1] << 14) | (b[2] << 7) | b[3]
    return 10 + size  # 10-byte header + content


def _parse_mp3_frame_header(b: bytes) -> Optional[dict]:
    if len(b) < 4 or b[0] != 0xFF or (b[1] & 0xE0) != 0xE0:
        return None
    version_bits = (b[1] >> 3) & 0x03
    layer_bits = (b[1] >> 1) & 0x03
    bitrate_idx = (b[2] >> 4) & 0x0F
    rate_idx = (b[2] >> 2) & 0x03
    padding = (b[2] >> 1) & 0x01
    channel_mode = (b[3] >> 6) & 0x03

    if version_bits not in _MPEG_VERSION or layer_bits not in _MPEG_LAYER:
        return None
    if bitrate_idx in (0, 0x0F) or rate_idx == 0x03:
        return None

    version = _MPEG_VERSION[version_bits]
    layer = _MPEG_LAYER[layer_bits]
    bitrate_table = _BITRATE_V1 if version == 1 else _BITRATE_V2
    bitrate_kbps = bitrate_table[layer][bitrate_idx - 1]
    sample_rate = _SAMPLE_RATE[version][rate_idx]
    samples = _SAMPLES_PER_FRAME[(version, layer)]

    if layer == 1:
        frame_len = (12 * bitrate_kbps * 1000 // sample_rate + padding) * 4
    else:
        frame_len = 144 * bitrate_kbps * 1000 // sample_rate + padding

    return {
        "version": version,
        "layer": layer,
        "bitrate_kbps": bitrate_kbps,
        "sample_rate": sample_rate,
        "samples_per_frame": samples,
        "frame_len": frame_len,
        "channel_mode": channel_mode,
    }


def _mp3_duration(data: bytes) -> Optional[float]:
    offset = _id3v2_size(data)
    # find first valid frame
    while offset + 4 <= len(data):
        if data[offset] == 0xFF and (data[offset + 1] & 0xE0) == 0xE0:
            hdr = _parse_mp3_frame_header(data[offset:offset + 4])
            if hdr:
                break
        offset += 1
    else:
        return None

    # Xing/Info or VBRI tag for VBR
    if hdr["version"] == 1:
        side_info_len = 17 if hdr["channel_mode"] == 3 else 32
    else:
        side_info_len = 9 if hdr["channel_mode"] == 3 else 17
    tag_offset = offset + 4 + side_info_len

    if tag_offset + 12 <= len(data):
        tag = data[tag_offset:tag_offset + 4]
        if tag in (b"Xing", b"Info"):
            flags = struct.unpack(">I", data[tag_offset + 4:tag_offset + 8])[0]
            if flags & 0x01:
                frames = struct.unpack(">I", data[tag_offset + 8:tag_offset + 12])[0]
                return frames * hdr["samples_per_frame"] / hdr["sample_rate"]
        elif tag == b"VBRI":
            frames = struct.unpack(">I", data[tag_offset + 14:tag_offset + 18])[0]
            return frames * hdr["samples_per_frame"] / hdr["sample_rate"]

    # CBR fallback: (file_size - id3) * 8 / bitrate
    audio_bytes = len(data) - _id3v2_size(data)
    return audio_bytes * 8.0 / (hdr["bitrate_kbps"] * 1000)


def _mp4_duration(data: bytes) -> Optional[float]:
    """Walk top-level atoms to moov/mvhd."""
    pos = 0
    while pos + 8 <= len(data):
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        atype = data[pos + 4:pos + 8]
        header_size = 8
        if size == 1 and pos + 16 <= len(data):
            size = struct.unpack(">Q", data[pos + 8:pos + 16])[0]
            header_size = 16
        elif size == 0:
            size = len(data) - pos
        if atype == b"moov":
            return _find_mvhd(data[pos + header_size:pos + size])
        if size <= 0:
            return None
        pos += size
    return None


def _find_mvhd(moov_body: bytes) -> Optional[float]:
    pos = 0
    while pos + 8 <= len(moov_body):
        size = struct.unpack(">I", moov_body[pos:pos + 4])[0]
        atype = moov_body[pos + 4:pos + 8]
        if size == 0:
            size = len(moov_body) - pos
        if atype == b"mvhd" and size >= 32:
            version = moov_body[pos + 8]
            if version == 0:
                # u32 timescale at +20, u32 duration at +24
                timescale = struct.unpack(">I", moov_body[pos + 20:pos + 24])[0]
                duration = struct.unpack(">I", moov_body[pos + 24:pos + 28])[0]
            else:
                # u32 timescale at +28, u64 duration at +32
                timescale = struct.unpack(">I", moov_body[pos + 28:pos + 32])[0]
                duration = struct.unpack(">Q", moov_body[pos + 32:pos + 40])[0]
            return duration / timescale if timescale else None
        if size <= 0:
            return None
        pos += size
    return None


_PROBES = {
    "wav": _wav_duration,
    "mp3": _mp3_duration,
    "m4a": _mp4_duration,
    "mp4": _mp4_duration,
}


def probe_audio_duration(source: AudioSource) -> Optional[float]:
    """Best-effort duration in seconds. Returns None for unknown / unsupported formats."""
    data, name = _read_all(source)
    fmt = detect_format(data, name)
    probe = _PROBES.get(fmt)
    if probe is None:
        return None
    return probe(data)


# --- validation ---


@dataclass
class AudioValidation:
    size_bytes: int
    detected_format: Optional[str]
    duration_s: Optional[float]
    warnings: list = field(default_factory=list)


def validate_audio_for_transcription(
    source: AudioSource,
    *,
    max_size_bytes: int = WHISPER_MAX_BYTES,
    max_duration_s: float = WHISPER_MAX_DURATION_S,
    allowed_formats: Iterable[str] = WHISPER_FORMATS,
) -> AudioValidation:
    """Validate an audio payload against whisper's hard limits.

    Raises ValueError for hard violations (empty / oversize / disallowed format /
    overlong probed duration). Returns AudioValidation with soft warnings otherwise.
    """
    data, name = _read_all(source)
    size = len(data)
    if size == 0:
        raise ValueError("audio is empty")
    if size > max_size_bytes:
        raise ValueError(
            f"audio size {size} bytes exceeds limit {max_size_bytes} bytes "
            f"(~{size / 1024 / 1024:.1f}MB > ~{max_size_bytes / 1024 / 1024:.0f}MB)"
        )

    fmt = detect_format(data, name)
    allowed = set(allowed_formats)
    if fmt is None:
        raise ValueError(
            f"could not detect audio format from magic bytes or filename {name!r}; "
            f"allowed formats: {sorted(allowed)}"
        )
    if fmt not in allowed:
        raise ValueError(f"detected format {fmt!r} not in allowed: {sorted(allowed)}")

    warnings: list = []
    duration = probe_audio_duration(data)
    if duration is None:
        warnings.append(f"duration probe not available for format {fmt!r}")
    elif duration > max_duration_s:
        raise ValueError(
            f"audio duration {duration:.1f}s exceeds limit {max_duration_s:.0f}s"
        )

    return AudioValidation(
        size_bytes=size,
        detected_format=fmt,
        duration_s=duration,
        warnings=warnings,
    )
