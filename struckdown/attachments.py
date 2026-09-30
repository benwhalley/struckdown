"""Images in struckdown prompts.

An image is passed in the context, wrapped by :func:`attach`, and goes where its
variable is rendered::

    complete("Read this sheet: {{ sheet }} [[extract:answers]]",
             {"sheet": attach("scan.jpg")})

How it gets there. Rendering an :class:`Attachment` (``struckdown_finalize``
calls :func:`render_value`) writes a placeholder into the text and records the
image in a per-call scope. The placeholder carries a random per-call nonce, so
text arriving from elsewhere -- a context string, a model's output -- can't
forge one; :func:`protect` escapes anything that looks like a placeholder but
lacks the nonce. After rendering, :func:`canonicalise` drops the nonce, leaving
``⟦sd-img:<sha256>:<max side>:<detail>:<metadata flag>⟧``. Messages therefore
stay plain strings, and the response cache keys on the image's content and the
settings it is sent with. Just before a model call, :func:`expand` swaps each
placeholder for the image itself (resized, re-encoded, metadata stripped) as a
pydantic-ai ``BinaryContent`` in the same position.

Pillow is needed only when an image is actually used: ``pip install
'struckdown[vision]'``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import re
import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_MAX_SIDE = 2048  # px, long side: gpt-5.6-sol stops using more detail here
DEFAULT_MAX_IMAGES = 50  # per request: Azure OpenAI's limit
DEFAULT_DETAIL = "auto"  # full resolution on current OpenAI models; "high" is lower
DETAILS = ("auto", "low", "high", "original")

_NONCE_RE = re.compile(r"⟦sd-img:([0-9a-f]{16}):([0-9a-f]{64}):(\d+):([a-z]+):([01])⟧")
PLACEHOLDER_RE = re.compile(r"⟦sd-img:([0-9a-f]{64}):(\d+):([a-z]+):([01])⟧")
_ANY_RE = re.compile(r"⟦sd-img", re.IGNORECASE)


class StruckdownAttachmentError(ValueError):
    """An image can't be used as asked: too many, wrong place, unsupported model..."""


@dataclass(frozen=True, eq=False)
class Attachment:
    """An image to send to the model. Build one with :func:`attach`."""

    source: Union[Path, bytes] = field(repr=False)
    name: Optional[str]
    sha256: str
    detail: Optional[str] = None
    metadata: bool = False

    def __str__(self) -> str:
        # str() happens when a filter such as |join stringifies the image
        return placeholder_for(self)

    def read(self) -> bytes:
        return self.source.read_bytes() if isinstance(self.source, Path) else self.source


class AttachmentList(list):
    """Several attachments; renders as each image in order."""

    def __str__(self) -> str:
        return "".join(placeholder_for(a) for a in self)


def _pil():
    """Pillow's Image module, with HEIC support if pillow-heif is installed."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Images need Pillow: pip install 'struckdown[vision]'"
        ) from e
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    return Image


def _check_detail(detail: Optional[str]) -> Optional[str]:
    if detail is not None and detail not in DETAILS:
        raise ValueError(f"detail must be one of {', '.join(DETAILS)}; got {detail!r}")
    return detail


def attach(source: Any, *, detail: Optional[str] = None, metadata: bool = False):
    """Wrap an image for use in a template's context.

    ``source`` is a path (``str`` or ``Path``), image bytes, a Pillow ``Image``,
    an existing :class:`Attachment`, or a list of any of these (which gives an
    :class:`AttachmentList`). A string is always a path here; a string in the
    context without ``attach()`` stays text.

    ``detail`` overrides the call's ``image_detail`` for this image.
    ``metadata=True`` sends the file's metadata (date, GPS, camera, original
    size) as a line of text straight after the image. Metadata is always
    removed from the image itself, which models don't read anyway.
    """
    _check_detail(detail)
    if isinstance(source, (list, tuple)):
        return AttachmentList(attach(s, detail=detail, metadata=metadata) for s in source)
    if isinstance(source, Attachment):
        return dataclasses.replace(
            source, detail=detail or source.detail, metadata=metadata or source.metadata
        )

    Image = _pil()
    if isinstance(source, Image.Image):
        buf = io.BytesIO()
        source.save(buf, "PNG")
        data = buf.getvalue()
        return Attachment(data, None, hashlib.sha256(data).hexdigest(), detail, metadata)
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        _verify_image(io.BytesIO(data), "the bytes given")
        return Attachment(data, None, hashlib.sha256(data).hexdigest(), detail, metadata)
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        _verify_image(path, path.name)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return Attachment(path, path.name, h.hexdigest(), detail, metadata)
    raise TypeError(
        f"attach() takes a path, bytes, a Pillow Image or a list of them; got {type(source).__name__}"
    )


def _verify_image(fp, label: str) -> None:
    Image = _pil()
    try:
        with Image.open(fp) as im:
            im.verify()
    except Exception as e:
        hint = ""
        if isinstance(fp, Path) and fp.suffix.lower() in (".heic", ".heif"):
            hint = " (HEIC needs pillow-heif: pip install 'struckdown[vision]')"
        raise StruckdownAttachmentError(f"{label} is not an image Pillow can read{hint}: {e}") from e


# --- per-call scope --------------------------------------------------------------


@dataclass
class _Scope:
    max_side: int = DEFAULT_MAX_SIDE
    max_images: int = DEFAULT_MAX_IMAGES
    detail: str = DEFAULT_DETAIL
    nonce: str = field(default_factory=lambda: secrets.token_hex(8))
    images: dict = field(default_factory=dict)  # sha256 -> Attachment
    encoded: dict = field(default_factory=dict)  # (sha256, side, metadata) -> (bytes, mime, text)


_scope: ContextVar[Optional[_Scope]] = ContextVar("struckdown_images", default=None)


def new_scope(
    image_max_side: Optional[int] = None,
    max_images: Optional[int] = None,
    image_detail: Optional[str] = None,
):
    """Start a fresh image scope for one ``complete()`` call; returns the reset token."""
    return _scope.set(
        _Scope(
            max_side=image_max_side or DEFAULT_MAX_SIDE,
            max_images=max_images or DEFAULT_MAX_IMAGES,
            detail=_check_detail(image_detail) or DEFAULT_DETAIL,
        )
    )


def reset_scope(token) -> None:
    _scope.reset(token)


def current_scope() -> _Scope:
    scope = _scope.get()
    if scope is None:
        scope = _Scope()
        _scope.set(scope)
    return scope


# --- rendering ---------------------------------------------------------------------


def placeholder_for(att: Attachment) -> str:
    scope = current_scope()
    scope.images[att.sha256] = att
    detail = att.detail or scope.detail
    return f"⟦sd-img:{scope.nonce}:{att.sha256}:{scope.max_side}:{detail}:{int(att.metadata)}⟧"


def _is_pil_image(value: Any) -> bool:
    return type(value).__module__.startswith("PIL.") and hasattr(value, "getexif")


def render_value(value: Any) -> Optional[str]:
    """Placeholder text if ``value`` is an image or a sequence of images, else None."""
    if isinstance(value, Attachment):
        return placeholder_for(value)
    if _is_pil_image(value):
        return placeholder_for(attach(value))
    if isinstance(value, (list, tuple)) or _is_iterator(value):
        items = list(value) if not isinstance(value, (list, tuple)) else value
        images = [v for v in items if isinstance(v, Attachment) or _is_pil_image(v)]
        if not images:
            return None if isinstance(value, (list, tuple)) else str(items)
        if len(images) != len(items):
            raise StruckdownAttachmentError(
                "a list mixing images with other values can't be rendered; "
                "loop over it and render each item"
            )
        return "".join(render_value(v) for v in items)
    return None


def _is_iterator(value: Any) -> bool:
    return hasattr(value, "__next__") and not isinstance(value, (str, bytes))


def protect(text: str) -> str:
    """Escape placeholder-shaped text that didn't come from this call's images.

    A placeholder carrying this call's nonce is kept. One whose nonce matches
    only ignoring case was changed by a filter (``|upper``) and raises. Anything
    else is escaped with a zero-width space, as struckdown escapes its own syntax.
    """
    if "⟦" not in text:
        return text
    nonce = current_scope().nonce
    out, last = [], 0
    for m in _ANY_RE.finditer(text):
        genuine = _NONCE_RE.match(text, m.start())
        if genuine and genuine.group(1) == nonce:
            continue
        if nonce in text[m.start(): m.start() + 60].lower() and nonce not in text[m.start(): m.start() + 60]:
            raise StruckdownAttachmentError(
                "a filter changed an image placeholder (e.g. |upper); render images directly"
            )
        out.append(text[last: m.start()] + "⟦​")
        last = m.start() + 1
    out.append(text[last:])
    return "".join(out)


def canonicalise(text: str) -> str:
    """Drop this call's nonce from its placeholders, so messages are stable text."""
    if "⟦sd-img:" not in text:
        return text
    nonce = current_scope().nonce

    def _canon(m):
        return f"⟦sd-img:{m.group(2)}:{m.group(3)}:{m.group(4)}:{m.group(5)}⟧" if m.group(1) == nonce else m.group(0)

    return _NONCE_RE.sub(_canon, text)


def contains_image(text: Optional[str]) -> bool:
    return bool(text) and bool(_ANY_RE.search(text))


def forbid_images(text: Optional[str], where: str) -> None:
    if contains_image(text):
        raise StruckdownAttachmentError(
            f"images can't go in {where}; put the image in the first user turn instead"
        )


def readable(text: str) -> str:
    """Placeholders as ``[image: name]`` for display."""
    if "⟦sd-img:" not in text:
        return text
    images = current_scope().images

    def _describe(m):
        att = images.get(m.group(1))
        return f"[image: {att.name or m.group(1)[:8]}]" if att else f"[image {m.group(1)[:8]}]"

    return PLACEHOLDER_RE.sub(_describe, text)


# --- sending ------------------------------------------------------------------------


def count_images(messages: Iterable[dict]) -> int:
    return sum(len(PLACEHOLDER_RE.findall(m.get("content") or "")) for m in messages
               if isinstance(m.get("content"), str))


def check_request(messages: list[dict], model_name: Optional[str], credentials=None) -> None:
    """Refuse a request before it is sent: images in a system message, too many
    images, or a model known not to accept them."""
    for m in messages:
        if m.get("role") == "system":
            forbid_images(m.get("content"), "a system prompt")
    n = count_images(messages)
    if not n:
        return
    limit = current_scope().max_images
    if n > limit:
        raise StruckdownAttachmentError(
            f"{n} images in this request (counting earlier turns); the limit is {limit} (max_images=)"
        )
    from .capabilities import supports_images

    ok, source = supports_images(model_name, credentials)
    if ok is False:
        raise StruckdownAttachmentError(f"{model_name} does not accept images, according to {source}")


def expand(text: str) -> Union[str, list]:
    """A user message's text with its placeholders swapped for the images."""
    if "⟦sd-img:" not in text:
        return text
    from pydantic_ai import BinaryContent

    scope = current_scope()
    parts: list = []
    last = 0
    for m in PLACEHOLDER_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last: m.start()])
        sha, side, detail, meta = m.group(1), int(m.group(2)), m.group(3), m.group(4) == "1"
        att = scope.images.get(sha)
        if att is None:
            raise StruckdownAttachmentError(
                f"the image {sha[:8]} for this prompt is no longer loaded; pass it in the context again"
            )
        data, mime, meta_text = encode(att, side, meta)
        parts.append(BinaryContent(data=data, media_type=mime, identifier=m.group(0),
                                   vendor_metadata={"detail": detail}))
        if meta_text:
            parts.append(meta_text)
        last = m.end()
    if last < len(text):
        parts.append(text[last:])
    return parts


def encode(att: Attachment, max_side: int, metadata: bool) -> tuple[bytes, str, Optional[str]]:
    """The bytes actually sent: upright, at most ``max_side`` px, no metadata.

    Memoised per call. Lossless sources (PNG, GIF, ...) stay PNG; photos become
    JPEG at quality 90. Transparency is flattened onto white; CMYK, 16-bit and
    palette images become RGB or greyscale; an animation or multi-image HEIC
    contributes its first frame.
    """
    scope = current_scope()
    key = (att.sha256, max_side, metadata)
    if key in scope.encoded:
        return scope.encoded[key]
    Image = _pil()
    from PIL import ImageOps

    im = Image.open(io.BytesIO(att.read()))
    fmt = (im.format or "").upper()
    meta_text = _metadata_text(im, att.name) if metadata else None
    im.seek(0)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        im = flat
    elif im.mode.startswith("I"):
        im = im.point(lambda v: v / 256).convert("L")
    elif im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    im = im.copy()
    im.info = {}  # nothing from the original file survives: no EXIF, XMP or comments
    buf = io.BytesIO()
    if fmt in ("PNG", "GIF", "BMP", "TIFF", "WEBP") and fmt:
        im.save(buf, "PNG", optimize=True)
        mime = "image/png"
    else:
        im.save(buf, "JPEG", quality=90)
        mime = "image/jpeg"
    scope.encoded[key] = (buf.getvalue(), mime, meta_text)
    return scope.encoded[key]


def _metadata_text(im, name: Optional[str]) -> str:
    """The readable parts of an image's EXIF, as one line of JSON."""
    exif = im.getexif()
    details = exif.get_ifd(0x8769)  # Exif sub-IFD
    gps = exif.get_ifd(0x8825)
    fields: dict[str, Any] = {}
    taken = details.get(0x9003) or exif.get(0x0132)  # DateTimeOriginal, DateTime
    if taken:
        fields["taken"] = str(taken).strip()
    camera = " ".join(str(exif.get(t, "")).strip() for t in (0x010F, 0x0110)).strip()  # Make, Model
    if camera:
        fields["camera"] = camera
    if details.get(0xA434):  # LensModel
        fields["lens"] = str(details[0xA434]).strip()
    if gps.get(2) and gps.get(4):
        try:
            fields["gps"] = {"lat": _degrees(gps[2], gps.get(1)), "lon": _degrees(gps[4], gps.get(3))}
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if exif.get(0x0112) not in (None, 1):
        fields["orientation"] = int(exif[0x0112])
    fields["size"] = f"{im.size[0]}x{im.size[1]}"
    return f"[Metadata for {name or 'this image'}: {json.dumps(fields)}]"


def _degrees(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    value = d + m / 60 + s / 3600
    return round(-value if str(ref).upper() in ("S", "W") else value, 6)
