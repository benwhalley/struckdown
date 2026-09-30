---
layout: default
title: Images
parent: How-To Guides
nav_order: 3
---

# Send images to a model

Wrap an image with `attach()` and pass it in the context. It goes wherever its
variable is rendered, in order with the text around it:

```python
from struckdown import complete, attach, LLM, LLMCredentials

result = complete(
    "Here is an answer sheet: {{ sheet }}\n"
    "What SRN is written in the boxes? [[srn]]",
    {"sheet": attach("scans/IMG_7553.HEIC")},
    model=LLM(model_name="gpt-5.6-sol"),
    credentials=LLMCredentials(api_key="...", base_url="https://litellm.example"),
)
result["srn"]
```

Images need Pillow: `pip install 'struckdown[vision]'` (which also adds HEIC
support).

## What `attach()` takes

- a path (`str` or `Path`), image bytes, or a Pillow `Image`;
- a list of any of these, which gives an `AttachmentList`.

`attach()` is required: a string in the context is always text, never read as
a path. A Pillow `Image` can go in the context directly. A missing file or a
file that isn't an image raises when you call `attach()`.

## Where images go

An image variable is an ordinary context variable. Its value decides how it
renders: a string as text, an attachment as that image, a list of attachments
as each image in order. Indexing, loops, macros and filters are plain Jinja:

```
Compare these: {{ photos }}

Front: {{ photos[0] }}  Back: {{ photos[1] }}

{% for p in photos %}Photo {{ loop.index }}: {{ p }}
{% endfor %}
Which photo is blurred? [[pick:blurred|1,2,3]]
```

The model receives one user message whose content is the text and images in
that order, so it can tell "image two" from "image one".

Images can only go in user turns. An image in a `<system>` block raises
`StruckdownAttachmentError`: put it in the first user turn instead.

## What is sent

Each image is re-encoded before sending:

- turned upright using its EXIF orientation;
- resized to at most `image_max_side` px on the long side (default 2048);
- saved as PNG if the original was lossless (PNG, GIF, ...), otherwise as JPEG
  at quality 90; transparency is flattened onto white, and an animation or
  multi-image HEIC contributes its first frame;
- **stripped of all metadata**: EXIF, including GPS location, date and camera.

Models don't read embedded metadata. If you want the model to know when or
where a photo was taken, ask for it as text:

```python
attach("IMG_1.jpg", metadata=True)
```

sends a line straight after the image, such as
`[Metadata for IMG_1.jpg: {"taken": "2026:09:18 10:02:11", "gps": {"lat": 50.37, "lon": -4.14}, "camera": "Apple iPhone 15"}]`.

## Options

| Argument to `complete()` | Default | Meaning |
|---|---|---|
| `image_max_side` | 2048 | long side in px after resizing |
| `max_images` | 50 | most images in one request, counting earlier turns |
| `image_detail` | `"auto"` | detail setting passed to the provider with each image |

`attach(path, detail="low")` overrides the detail setting for one image.

The defaults were measured on `gpt-5.6-sol` via LiteLLM and Azure: the model
uses no more than 2048 px (input tokens stop rising there), Azure refuses a
51st image in a request, and `"auto"` gives full resolution, while `"high"` is a
cheaper, lower-resolution tier that missed fine text in testing.

## Things to know

- **Later slots resend earlier images.** Each slot resends the conversation so
  far, so an image before the first slot is sent, and billed, again for every
  later slot. A `<checkpoint>` drops earlier turns and their images; render the
  variable again after it if you still need the image.
- **A missing variable is silent.** If the context has no `sheet`, `{{ sheet }}`
  renders as nothing and the model is asked about an image it never sees. Use
  `strict_undefined=True` for image templates.
- **Caching keys on content.** The same image under another filename is a
  cache hit; a changed image is a miss.
- **Unsupported models are caught early.** Before sending, struckdown asks
  whether the model accepts images: the LiteLLM proxy you're calling (its
  `/model/info`), then LiteLLM's public model registry, then OpenRouter's model
  list. A model known not to accept images raises; an unknown one is sent
  anyway. `STRUCKDOWN_CHECK_IMAGE_SUPPORT=0` skips the lookup.

## From the command line

```bash
sd chat "Describe {{ photo }} [[description]]" --attach photo=IMG_1.jpg
sd chat "Compare {{ photos }} [[sharper]]" --attach "photos=scans/*.jpg"
sd batch scans/*.jpg -p read_sheet.sd --as-image -o results.xlsx
```

`--attach` takes `name=path`; a glob gives a list. `sd chat` also accepts
`--image-max-side`, `--max-images` and `--image-detail`. With `--as-image`,
`sd batch` makes `{{ input }}` (and `{{ image }}`) each image itself, rather
than text read from it.
