"""Image uploads for a route: held in memory, checked, never written to disk.

Django spills an upload bigger than ``FILE_UPLOAD_MAX_MEMORY_SIZE`` to a
temporary file. :class:`MemoryOnlyUploadHandler` replaces that for the API
view: every file stays in memory, and one over the route's size cap is
dropped mid-stream and reported, rather than buffered and refused later.

:func:`read_image` then opens the bytes with Pillow to confirm they are an
image of an allowed format. Struckdown re-encodes the image and strips its
metadata before it goes to the provider (``struckdown.attachments``), so what
is sent is pixels, not whatever else the file carried.
"""

from __future__ import annotations

import io

from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.uploadhandler import FileUploadHandler, SkipFile

# MPO is the JPEG variant iPhones write
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "GIF"}
# a small file can declare a huge image that takes gigabytes to decode; the
# header gives the size without decoding, so refuse before anything does
MAX_PIXELS = 40_000_000


class MemoryOnlyUploadHandler(FileUploadHandler):
    def __init__(self, request=None, *, max_bytes: int, max_files: int):
        super().__init__(request)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.files_seen = 0
        self.refused: dict[str, str] = {}

    def new_file(self, field_name, *args, **kwargs):
        super().new_file(field_name, *args, **kwargs)
        self.files_seen += 1
        if self.files_seen > self.max_files:
            self.refused[field_name] = f"more than {self.max_files} images"
            raise SkipFile
        self.buffer = io.BytesIO()

    def receive_data_chunk(self, raw_data, start):
        if start + len(raw_data) > self.max_bytes:
            self.refused[self.field_name] = f"over {self.max_bytes:,} bytes"
            raise SkipFile
        self.buffer.write(raw_data)

    def file_complete(self, file_size):
        self.buffer.seek(0)
        return InMemoryUploadedFile(
            file=self.buffer,
            field_name=self.field_name,
            name=self.file_name,
            content_type=self.content_type,
            size=file_size,
            charset=self.charset,
            content_type_extra=self.content_type_extra,
        )


class NotAnImage(ValueError):
    pass


def read_image(upload) -> bytes:
    """The upload's bytes, if Pillow reads them as an allowed image format."""
    try:
        from PIL import Image
    except ImportError as exc:  # struckdown[vision] not installed
        raise NotAnImage("image support is not installed on this server") from exc
    data = upload.read()
    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt = image.format
            width, height = image.size
            frames = getattr(image, "n_frames", 1)
            if width * height > MAX_PIXELS:
                raise NotAnImage(f"too many pixels ({width}x{height}; at most {MAX_PIXELS:,})")
            if frames > 1 and fmt != "MPO":
                raise NotAnImage("animated images are not accepted")
            image.verify()
    except NotAnImage:
        raise
    except Exception as exc:
        raise NotAnImage("not a readable image") from exc
    if fmt not in ALLOWED_FORMATS:
        raise NotAnImage(f"{fmt} images are not accepted ({', '.join(sorted(ALLOWED_FORMATS))})")
    return data
