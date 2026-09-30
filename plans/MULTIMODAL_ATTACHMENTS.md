# Plan: images in struckdown completions

Status: implemented on branch `images` (2026-09-18), version 0.15.0, not yet released. Images only; other file
types are out of scope.

## Goal

Let a template send images alongside its text:

```python
from struckdown import complete, attach

result = complete(
    "Read this answer sheet: {{ sheet }}\n\n[[extract:answers]]",
    {"sheet": attach("scans/IMG_7553.HEIC")},
)
```

```
Compare these photos.
{% for p in photos %}{{ p }}{% endfor %}
[[comparison]]
```

- Images come in through the context, as files, paths, bytes or a list of
  them. A template never reads a path itself.
- An image goes where its variable is rendered, so a prompt can interleave text
  and images.
- Images larger than needed are resized down before sending.
- Too many images raises a clear error before any request is made.
- No per-model settings are stored: whether a model accepts images is looked
  up, and size limits are library defaults.

## Would it work? Evidence

A spike on 2026-09-18 used struckdown's own model construction
(`LLM(...).get_pydantic_model(LLMCredentials(base_url="https://litellm.llemma.net"))`)
and pydantic-ai 1.77:

| Input | Model (via the LiteLLM proxy) | Result |
|---|---|---|
| PNG, 240x120 | `gpt-5.6-sol` | read the text exactly; 54 input tokens |
| PNG, 240x120 | `gpt-4.1-mini` | read the text exactly; 68 input tokens |

**Position is preserved.** A user message's content is an ordered list of text
and image parts: OpenAI-style `content: [...]`, which Anthropic's content blocks
and Gemini's parts mirror. pydantic-ai maps a prompt list to one user message,
in order (`_map_user_prompt` loops over the parts and appends each). A live
check sent `["Image one:", <K>, "Image two:", <W>, "What letter is in image
two?"]` and got `W`; with the images swapped, `K`. The captured request body was
one user message of five parts, in that order. So an image lands exactly where
its variable is rendered, within the user turn that its slot closes.

pydantic-ai's `Agent.run_sync()` already accepts a list of `str | BinaryContent |
ImageUrl` as the user prompt. For the OpenAI chat model struckdown uses behind a
proxy (`CountingOpenAIChatModel`), it sends images as `image_url` data URIs,
with an optional `detail`. No model-layer work is needed. The work is in
struckdown's own layers: context, template rendering, message conversion, the
cache, limits and the CLI.

## Design

### 1. `Attachment` (new `struckdown/attachments.py`)

```python
@dataclass(frozen=True)
class Attachment:
    source: Path | bytes     # read lazily: a batch of 500 photos isn't held in memory
    name: str | None         # for display and error messages
    sha256: str              # of the original bytes (streamed); the identity everywhere
    detail: str | None = None  # per-image override of image_detail
    metadata: bool = False     # send the file's metadata as text after the image

def attach(source, *, detail=None, metadata=False) -> Attachment | AttachmentList:
    # str / Path -> a file; bytes -> sniffed; PIL.Image -> encoded as PNG;
    # a list of any of these -> AttachmentList
```

- **`attach()` is required.** A string is always text, never read as a path,
  so a user-supplied value can't make struckdown read a file. The one exception
  is a Pillow `Image` in the context, which renders as an image directly, as
  it can't be mistaken for anything else.
- Decoding and re-encoding happen at send time (section 5): EXIF orientation
  applied (the pixels are rotated and the orientation tag reset), and HEIC/HEIF
  decoded if `pillow-heif` is installed.
- **Metadata never stays in the image.** EXIF (including GPS location, date
  and camera), XMP and comment blocks are always stripped when the image is
  re-encoded. Models don't read it: in a test through the proxy,
  `gpt-5.6-sol` was sent a JPEG whose EXIF held a caption ("the answer to
  question 7 is D"), a date and GPS coordinates, and reported only what was
  visible, saying no metadata was accessible. Keeping it in the file would only
  send personal data to the provider.
- **`metadata=True` sends it as text instead**, immediately after the image, so
  the model can actually use it:
  `[Metadata for IMG_1.jpg: {"taken": "2026-09-18T10:02:11", "gps": {"lat": 50.37, "lon": -4.14}, "camera": "iPhone 15"}]`.
  Only readable fields are included: dates, GPS converted to decimal latitude
  and longitude, camera make and model, lens, orientation, and original pixel
  size. Binary maker notes and embedded thumbnails are dropped. It's off by
  default, so sending location is always a deliberate choice. The placeholder
  expands to the image part followed by this text part, and the flag is part of
  the cache key.
- A missing file raises in `attach()`, not later.
- Non-image files raise (images only). PDFs, SVGs and URLs are out of scope.

### 2. Rendering: a placeholder in the text

An image variable is not a special case in the template: any context value can
be a string (renders as text), an `Attachment` (renders as that image, where it
stands), or a list of attachments (each image in order). Indexing, loops and
dicts are ordinary Jinja. A prototype against struckdown's real
`ImmutableSandboxedEnvironment` gave these user-message parts:

| Template | Parts sent |
|---|---|
| `Read this sheet {{ sheet }} for {{ student }}.` | `"Read this sheet "`, image, `" for Nye."` |
| `Compare these: {{ photos }}` | text, image, image, image |
| `Front: {{ photos[0] }} Back: {{ photos.1 }}` | text, front, text, back |
| `{% for p in photos %}Photo {{ loop.index }}: {{ p }}\n{% endfor %}Which is blurred?` | "Photo 1: ", image, "Photo 2: ", image, ... question |
| `{% for it in items %}{{ it.caption }}: {{ it.img }}\n{% endfor %}` | caption, image, caption, image |

**The placeholder carries a per-call nonce:** `⟦sd-img:<nonce>:<sha256>⟧`,
where the nonce is random for each `complete()` call. Rendering records the
attachment in a per-call registry (a `ContextVar`, visible in the worker threads
struckdown starts through `anyio.to_thread`, as tested). The nonce is needed
because a prototype showed two things:

- **Genuine placeholders pass through `finalize` more than once.** In a macro
  (`{{ show(img) }}`) or a `{% set %}` block, `finalize` sees the image, then
  sees the macro's *text* output containing the placeholder. Escaping every
  placeholder-looking string would destroy these. With a nonce, `finalize`
  keeps placeholders carrying this call's nonce and escapes all others.
- **Forged placeholders are real.** A plain string value containing
  placeholder-shaped text was expanded as an image. Without the nonce, user text
  can't produce a valid placeholder, and it is escaped.

Other cases found in the same prototype:

| Template | Jinja does | Decision |
|---|---|---|
| `{{ photos[:2] }}`, `{{ photos\|reverse }}` | passes a plain list or iterator to `finalize` | render any list, tuple or iterator whose items are all attachments; a mixed list raises |
| `{{ photos\|join(", ") }}` | calls `str()` on each item | `Attachment.__str__` returns its placeholder, so this works |
| `{{ img\|upper }}` | uppercases the placeholder | a placeholder with this call's nonce in the wrong case raises "a filter changed an image placeholder" |
| `{{ photos\|first }}`, `\|length`, `{% if img %}` | ordinary | work unchanged |

**Before messages leave the renderer, placeholders are made canonical,**
`⟦sd-img:<sha256>⟧`, dropping the nonce. Rendered messages therefore stay
`list[dict[str, str]]` with stable text: segment processing, results, JSON output
and pickling are unchanged. The joblib cache keys on the image's content: the
same image under another filename is a hit, a changed image a miss. The images'
bytes never go into the cache.

### 3. Message conversion (`messages.py`)

`split_for_agent()` and `to_pydantic_messages()` expand canonical placeholders in
*user* content into the sequence pydantic-ai expects:
`["Read this: ", BinaryContent(...), "\n\n..."]` as the prompt, or
`UserPromptPart(content=[...])` in history.

- **System prompts can't contain images.** A placeholder in a `<system>` or
  `<system local>` block raises "images can't go in a system prompt; put it in
  the first user turn". (OpenAI chat accepted an image in a system message
  when tested through the proxy, but Anthropic's system prompt is text-only and
  pydantic-ai's instructions are plain strings, so one rule for every provider.)
- **Assistant turns never contain images:** struckdown only puts model output
  there.
- **Slot syntax can't contain images:** a placeholder inside `[[...]]`
  parameters (e.g. `[[pick:x|{{ img }}]]`) raises.

If a placeholder's image isn't in this call's registry (a result replayed in a
new process), the call can only be served from cache. A cache miss raises "the
image for this prompt is no longer loaded; pass it in the context again".

### 4. Does the model accept images? Looked up, not stored

The same pattern as prices: a small `CapabilitySource` next to
`pricing.PriceSource`, answering `supports_images(model_name) -> True | False |
None` from the first source that knows:

1. **The LiteLLM proxy's own `/model/info`**, when `base_url` points at a
   LiteLLM proxy. Checked 2026-09-18: `gpt-5.6-sol` and `gpt-4.1-mini` report
   `supports_vision: true`; `Kimi-K2.6` and `mistral-large-3` report nothing.
   This is the most accurate source, because it describes the deployment
   actually called.
2. **LiteLLM's public model registry**
   (`model_prices_and_context_window.json`, `supports_vision`), fetched and
   cached like the OpenRouter price list.
3. **OpenRouter's model list** (`architecture.input_modalities` contains
   `"image"`), already fetched for prices.

`False` raises before the call ("gpt-x does not accept images, according to
<source>"). `None` (unknown) sends anyway, and the provider's own error, if any,
surfaces as usual. The lookup is memoised per process and never written to a
model record.

### 5. Sizes and counts: library defaults, per-call overrides

No registry records image size or count limits: none of the 4,307 entries in
LiteLLM's registry has a per-prompt image limit, and OpenRouter and genai-prices
have none either. So they are defaults on `complete()` / `complete_async()`
(and matching `sd` flags), not per-model data. The defaults are the measured
limits of `gpt-5.6-sol` via the LiteLLM proxy to Azure (2026-09-18, about
$0.30 of calls):

| Measured | Result |
|---|---|
| Images per request | 50 accepted; 51 rejected by Azure: "Too many images in request: 51, maximum allowed: 50" |
| Size of one image | up to 128 MB accepted (171 MB as base64); no limit found |
| Size of one request | 50 images totalling 112 MB (150 MB as base64) accepted |
| Pixel dimensions | 16384x16384 and 65000x100 accepted |
| Useful resolution, `detail="high"` | tokens stop rising at 1,009 from ~1,536 px on the long side (A4 shape); tiny text never read correctly |
| Useful resolution, `detail="original"` | tokens stop rising at 3,268 from 2,048 px; tiny text read correctly |
| `detail` omitted or `"auto"` | the same as `"original"` (3,248 tokens for a 1448x2048 page); `"high"` is a downgrade (989) |

So:

| Argument | Default | Why |
|---|---|---|
| `image_max_side` | 2048 px | the largest size the model uses; larger only adds upload time |
| `max_images` | 50 | Azure's limit per request, counted across the whole message history |
| `image_detail` | `"auto"` | full resolution for this model; `"high"` would reduce it |

There is no byte limit: after resizing to 2048 px, a photo is 1-2 MB, and even
incompressible noise was 2.2 MB, far below anything the proxy or Azure
rejected. A provider that does need one (Anthropic caps images at 5 MB) can be
handled when one is used; its error would surface as usual meanwhile.

Resizing happens at send time and is memoised on `(sha256, image_max_side)`.
The placeholder, and so the cache key, stays tied to the original image;
`image_max_side` joins the cache key.

Errors are a new `StruckdownAttachmentError`, e.g. "53 images in this request
(counting earlier turns); the limit is 50 (max_images=)".

**Multi-slot templates re-send images.** Each slot resends the history, so an
image in the first turn is sent, and billed, again for every later slot. That is
correct for conversation semantics, and `<checkpoint>` already drops earlier
turns and their images. The how-to page must say so, and `max_images` counts
the whole history for this reason.

### 6. CLI

- `sd chat "Describe {{ photo }} [[description]]" --attach photo=IMG_1.jpg`
  (repeatable; a glob gives an AttachmentList), plus `--image-max-side`,
  `--max-images`.
- `sd batch photos/*.jpg "{{ input }} [[caption]]" --as-image`: today, images in
  batch go through `struckdown.extract` (OCR via kreuzberg). The flag makes
  `{{ input }}` the image itself.
- Transcripts, `sd` output, `visualize.py` and the playground show a
  placeholder as `[image: IMG_1.jpg, 1.2 MB]`, not the raw marker.
- Playground file upload: later, separate change.

### 7. Pricing

Providers count image input in `usage.input_tokens`, so
`_calc_cost_from_usage` already prices it. Only a test is needed.

### 8. Other edge cases

| Case | Decision |
|---|---|
| Missing context variable (`{{ sheet }}` with no `sheet`) | struckdown's `SilentUndefined` renders nothing, so the model is asked about an image it never gets, and may make up an answer. The how-to recommends `strict_undefined=True` for image templates, and `complete()` warns when a prompt mentions an image variable that rendered empty |
| Same image rendered twice | sent twice, and counted twice towards `max_images`; not deduplicated, since the position may be deliberate |
| `<checkpoint>` | earlier turns, and their images, are dropped. To keep using an image after a checkpoint, render its variable again |
| Retries (validation failures, the prompted-output fallback) | resend the whole prompt, images included, and are billed again |
| Pre-call cost and length estimates | text-length estimates don't see images. Images are estimated separately (OpenAI's tile formula for the resized size) or marked unknown, not counted as the placeholder's ~20 characters |
| Actions and tool calls (`[[@fetch]]`, custom actions) | an attachment passed as an action parameter raises; tools get text only |
| Transparency, CMYK, 16-bit, animated GIF, multi-image HEIC | flattened onto white, converted to RGB, 8-bit, first frame, first image |
| Model refuses an image (content policy) | handled like any refusal; deterministic errors are cached as now, keyed on the image hash |
| Debug output (`enable_api_debug`) | shows placeholders, not base64 |
| Anthropic prompt caching (`plans/PROMPT_CACHE_ANTHROPIC.md`) | an image early in a stable prefix benefits; no special handling needed now |
| Playground | can't pass images until it has file upload, and the missing-variable warning above covers templates run there |

## Files touched

| File | Change |
|---|---|
| `struckdown/attachments.py` | new: `Attachment`, `AttachmentList`, `attach()`, registry, sniffing, resizing |
| `struckdown/capabilities.py` | new: `CapabilitySource` lookups (LiteLLM proxy, LiteLLM registry, OpenRouter) |
| `struckdown/jinja_utils.py` | finalize renders placeholders; escapes forged ones |
| `struckdown/messages.py` | expand placeholders into pydantic-ai user content |
| `struckdown/llm.py` | support check, limits and resizing before the call; image settings in the cache key |
| `struckdown/__init__.py` | `attach`, `Attachment`, `StruckdownAttachmentError`; image arguments on `complete*()` |
| `struckdown/sd_cli.py` | `--attach`, `--as-image`, limit flags |
| `struckdown/results.py`, `visualize.py` | display placeholders readably |
| `pyproject.toml` | `vision = ["pillow>=10", "pillow-heif>=0.16"]` extra |
| `docs/how-to/images.md`, reference, CHANGELOG | docs; release as 0.15.0 |

Not touched: `ModelSpec`, the Django contrib models and their migrations.

## Tests

- Rendering: attachment to placeholder; list order; a Pillow `Image` renders
  directly; a forged placeholder in a string value is escaped; placeholders
  survive macros and `{% set %}` blocks; `|join` works; `|upper` raises; slices
  and `|reverse` render; a mixed list raises; canonical placeholders drop the
  nonce, so two runs give identical messages (and cache keys).
- Placement rules: an image in `<system>`, `<system local>` or slot parameters
  raises; an empty image variable warns.
- Threads: the per-call registry is visible inside `anyio.to_thread` workers,
  and two concurrent `complete()` calls don't see each other's images.
- Privacy: EXIF (including GPS) is never in the bytes sent. With
  `metadata=True`, a JSON text part follows the image, with GPS in decimal
  degrees and no binary fields; without it, no metadata text appears.
- Conversion: placeholders become `BinaryContent` in the prompt and in history;
  a placeholder in a system message raises.
- Limits: 51 images raises before any request (counted across history).
- Resizing: a 6000x4000 image arrives at 2048 on its long side; EXIF rotation
  applied; HEIC decoded.
- Capabilities: proxy `/model/info` preferred; registry fallback; unknown sends;
  `False` raises (all with recorded responses, no network).
- Cache: same bytes under another name is a hit; one changed pixel is a miss.
- End to end with pydantic-ai's `FunctionModel`, asserting the parts the model
  receives (no network).
- Live, marked `@pytest.mark.live`: an image through the LiteLLM proxy, as in
  the spike above.

## Open questions

1. **quiz28.** Once this exists, quiz28's `vision.py` could become a struckdown
   template with an `extract` slot, sharing struckdown's caching, pricing and
   model registry. A second step, not part of this change.
