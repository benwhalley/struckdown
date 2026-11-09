// Route admin: the struckdown editor on the prompt, "Generate from the prompt"
// beside each schema, and the host in the endpoint and example.
(function () {
    function ready(fn) {
        if (document.readyState !== "loading") fn();
        else document.addEventListener("DOMContentLoaded", fn);
    }

    function csrf() {
        var el = document.querySelector("[name=csrfmiddlewaretoken]");
        return el ? el.value : "";
    }

    ready(function () {
        document.querySelectorAll(".sd-origin").forEach(function (el) {
            el.textContent = window.location.origin;
        });

        // the highlighting editor, if the project serves it; the textarea
        // stays the form field and is kept in step on every change
        var prompt = document.querySelector("textarea[data-sd-editor]");
        if (prompt && window.StruckdownEditor) {
            var mount = document.createElement("div");
            mount.className = "sd-editor-mount";
            prompt.parentNode.insertBefore(mount, prompt);
            // not the hidden attribute: admin themes' display rules override it
            prompt.style.display = "none";
            window.StruckdownEditor.create(mount, prompt.value, {
                onChange: function (value) { prompt.value = value; },
            });
        }

        document.addEventListener("click", function (ev) {
            var btn = ev.target.closest && ev.target.closest(".sd-schema-generate");
            if (!btn) return;
            ev.preventDefault();
            var area = btn.parentNode.previousElementSibling;
            var status = btn.parentNode.querySelector(".sd-schema-status");
            if (area.value.trim() && !window.confirm("Replace the schema with one generated from the prompt?")) return;
            var data = new FormData();
            data.append("kind", btn.dataset.kind);
            data.append("prompt", prompt ? prompt.value : "");
            var tokens = document.querySelector("[name=max_input_tokens]");
            data.append("max_input_tokens", tokens ? tokens.value : "");
            status.textContent = "Generating...";
            fetch(btn.dataset.url, {
                method: "POST",
                body: data,
                credentials: "same-origin",
                headers: { "X-CSRFToken": csrf() },
            })
                .then(function (r) { return r.json(); })
                .then(function (j) {
                    if (j.error) { status.textContent = j.error; return; }
                    area.value = JSON.stringify(j.schema, null, 2);
                    status.textContent = "Generated. Edit it as needed, then save.";
                })
                .catch(function () { status.textContent = "Could not generate."; });
        });
    });
})();
