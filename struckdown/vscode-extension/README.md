# Struckdown VSCode Extension

Syntax highlighting for Struckdown prompt files (`.sd`).

## Install

```bash
cd /Users/benwhalley/dev/struckdown/vscode-extension
./install.sh
```

Then reload VSCode: **Cmd/Ctrl+Shift+P** → "Reload Window"

## Activate Theme

**Cmd/Ctrl+Shift+P** → "Color Theme" → Select:
- **Struckdown Dark (Atom One Style)**
- **Struckdown Light (Atom One Style)**

Both themes include:
- `[[slots]]` → Yellow background, black bold text
- `{{vars}}` → Green background, black bold text
- `<checkpoint>` → Red bold
- `<system>` → Cyan bold
- Full markdown support

## Don't Want to Change Themes?

Add struckdown colors to your current theme. Open settings (**Cmd/Ctrl+Shift+P** → "Preferences: Open User Settings (JSON)") and add:

```json
{
  "editor.tokenColorCustomizations": {
    "textMateRules": [
      {
        "scope": [
          "variable.other.placeholder.struckdown",
          "punctuation.definition.placeholder.begin.struckdown",
          "punctuation.definition.placeholder.end.struckdown",
          "storage.type.struckdown",
          "punctuation.separator.type.struckdown",
          "keyword.operator.multiplier.struckdown"
        ],
        "settings": {
          "background": "#FFFF00",
          "foreground": "#000000",
          "fontStyle": "bold"
        }
      },
      {
        "scope": [
          "variable.other.template.struckdown",
          "punctuation.definition.template.begin.struckdown",
          "punctuation.definition.template.end.struckdown"
        ],
        "settings": {
          "background": "#00FF00",
          "foreground": "#000000",
          "fontStyle": "bold"
        }
      }
    ]
  }
}
```

**Note:** Background colors work in most themes. If they don't show up, change `"background"` to use `"foreground"` with bright colors instead.


