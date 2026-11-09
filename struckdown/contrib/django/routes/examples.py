"""Copy-paste examples for a route: a whole HTML page with a form, and a Python script.

Both are built from the route's input schema, so the form has a field per
input of the right kind (a textarea for text, a number box, a select for an
enum, a file picker for an image). ``ORIGIN`` stands in for the site's
address, which the admin page fills in from the browser.
"""

from __future__ import annotations

import json
from html import escape

from . import schema as schemas

ORIGIN = "__ORIGIN__"

_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:42rem;margin:2rem auto;padding:0 1rem;"
    "line-height:1.5}label{display:block;margin-top:1rem;font-weight:600}"
    "textarea,input,select{font:inherit;width:100%;box-sizing:border-box}"
    "textarea{min-height:8rem}input[type=checkbox]{width:auto}"
    "button{margin-top:1rem;font:inherit;padding:.4rem 1.2rem}"
    "pre{background:#f4f4f4;padding:1rem;white-space:pre-wrap}"
)


def _tidy(code: str) -> str:
    """No blank lines and no trailing newline: shown in a box and copied as is."""
    return "\n".join(line.rstrip() for line in code.splitlines() if line.strip())


def _field(name: str, prop: dict, required: bool) -> str:
    n = escape(name)
    req = " required" if required else ""
    label = f'  <label for="{n}">{n}</label>\n'
    if prop.get("format") == "image":
        return label + f'  <input type="file" id="{n}" name="{n}" accept="image/*"{req}>\n'
    if "enum" in prop:
        options = "".join(
            f'<option value="{escape(str(v))}">{escape(str(v))}</option>' for v in prop["enum"]
        )
        return label + f'  <select id="{n}" name="{n}"{req}>{options}</select>\n'
    kind = prop.get("type")
    if kind in ("integer", "number"):
        step = "1" if kind == "integer" else "any"
        return label + f'  <input type="number" step="{step}" id="{n}" name="{n}"{req}>\n'
    if kind == "boolean":
        return (
            f'  <label><input type="checkbox" id="{n}" name="{n}" value="true"> {n}</label>\n'
        )
    max_length = prop.get("maxLength")
    limit = f' maxlength="{max_length}"' if max_length else ""
    return label + f'  <textarea id="{n}" name="{n}"{limit}{req}></textarea>\n'


def html_page(route, url: str, token: str) -> str:
    """A complete page: a form for the route's inputs and the reply below it."""
    input_schema = route.effective_input_schema()
    required = set(input_schema.get("required") or [])
    fields = "".join(
        _field(name, prop, name in required)
        for name, prop in (input_schema.get("properties") or {}).items()
    )
    title = escape(route.name)
    return _tidy(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_STYLE}</style>
</head>
<body>
<h1>{title}</h1>
<form id="form">
{fields}  <button type="submit">Send</button>
</form>
<pre id="result"></pre>
<script>
const URL = "{url}";
const TOKEN = "{token}";

document.getElementById("form").addEventListener("submit", async (event) => {{
  event.preventDefault();
  const result = document.getElementById("result");
  result.textContent = "Working...";
  try {{
    const response = await fetch(URL, {{
      method: "POST",
      headers: {{"Authorization": "Bearer " + TOKEN}},
      body: new FormData(event.target),
    }});
    const reply = await response.json();
    // reply.data has one key per [[slot]]; reply.cost is what the call cost
    result.textContent = JSON.stringify(response.ok ? reply.data : reply, null, 2);
  }} catch (error) {{
    result.textContent = "Could not reach the server: " + error;
  }}
}});
</script>
</body>
</html>""")


def python_script(route, url: str, token: str) -> str:
    input_schema = route.effective_input_schema()
    images = schemas.image_fields(input_schema)
    text = {n: "..." for n in (input_schema.get("properties") or {}) if n not in images}
    if images:
        files = ", ".join(f'"{n}": open("{n}.jpg", "rb")' for n in sorted(images))
        payload = f"    data={json.dumps(text)},\n    files={{{files}}},\n"
    else:
        payload = f"    json={json.dumps(text)},\n"
    return _tidy(
        "import requests\n"
        "response = requests.post(\n"
        f'    "{url}",\n'
        f'    headers={{"Authorization": "Bearer {token}"}},\n'
        f"{payload}"
        "    timeout=120,\n"
        ")\n"
        "response.raise_for_status()\n"
        "reply = response.json()\n"
        'print(reply["data"])  # one key per [[slot]]\n'
        'print(reply["cost"])\n'
    )
