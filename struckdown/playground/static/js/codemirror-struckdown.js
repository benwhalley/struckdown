/**
 * Struckdown syntax highlighting editor
 *
 * Uses a custom overlay-based approach with contenteditable div
 * for real-time syntax highlighting without external dependencies.
 *
 * Highlights:
 * - [[slot]] completions (green)
 * - [[@action]] actions (purple)
 * - [[@break]] break action (red)
 * - {{variable}} template vars (pink)
 * - {% tag %} jinja tags (blue)
 * - <system>...</system> blocks (grey)
 * - <header>...</header> blocks (yellow)
 * - <checkpoint> markers (gold)
 * - <!-- comments --> (muted)
 */

// Regex patterns for syntax elements
const PATTERNS = [
    // Break action [[@break...]] - red (must be before other actions)
    { pattern: /\[\[@break[^\]]*\]\]/g, class: 'sd-break' },
    // Action slots [[@...]] - purple (must be before regular slots)
    { pattern: /\[\[@[^\]]*\]\]/g, class: 'sd-action' },
    // Regular slots [[...]] - green
    { pattern: /\[\[[^\]]*\]\]/g, class: 'sd-slot' },
    // Template vars {{...}} - pink
    { pattern: /\{\{[^}]*\}\}/g, class: 'sd-template-var' },
    // Jinja tags {% ... %}
    { pattern: /\{%[^%]*%\}/g, class: 'sd-jinja-tag' },
    // System tags
    { pattern: /<system[^>]*>/gi, class: 'sd-system-tag' },
    { pattern: /<\/system>/gi, class: 'sd-system-tag' },
    // Header tags
    { pattern: /<header[^>]*>/gi, class: 'sd-header-tag' },
    { pattern: /<\/header>/gi, class: 'sd-header-tag' },
    // Checkpoint
    { pattern: /<checkpoint\s*\/?>/gi, class: 'sd-checkpoint' },
    { pattern: /<\/checkpoint>/gi, class: 'sd-checkpoint' },
    // Comments
    { pattern: /<!--[\s\S]*?-->/g, class: 'sd-comment' },
];

// Special pattern for markdown headings (needs line-by-line processing)
const HEADING_PATTERN = /^(#{1,6})\s+(.+)$/gm;

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// Check if cursor is inside a {{var}} and return the variable name
function getVariableAtPosition(text, position) {
    const before = text.slice(0, position);
    const after = text.slice(position);

    const openMatch = before.match(/\{\{([^}]*)$/);
    const closeMatch = after.match(/^([^}]*)\}\}/);

    if (openMatch && closeMatch) {
        const varName = (openMatch[1] + closeMatch[1]).trim();
        return varName || null;
    }
    return null;
}

function highlightSyntax(text) {
    // Collect all matches with their positions
    const matches = [];

    for (const {pattern, class: className} of PATTERNS) {
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(text)) !== null) {
            matches.push({
                start: match.index,
                end: match.index + match[0].length,
                text: match[0],
                className,
                isHeading: false
            });
        }
    }

    // Handle markdown headings specially (with separate marker and text styling)
    HEADING_PATTERN.lastIndex = 0;
    let headingMatch;
    while ((headingMatch = HEADING_PATTERN.exec(text)) !== null) {
        const marker = headingMatch[1];
        const headingText = headingMatch[2];
        matches.push({
            start: headingMatch.index,
            end: headingMatch.index + headingMatch[0].length,
            marker: marker,
            headingText: headingText,
            isHeading: true
        });
    }

    // Sort by start position
    matches.sort((a, b) => a.start - b.start);

    // Remove overlapping matches (keep first)
    const filtered = [];
    let lastEnd = 0;
    for (const match of matches) {
        if (match.start >= lastEnd) {
            filtered.push(match);
            lastEnd = match.end;
        }
    }

    // Build highlighted HTML
    let result = '';
    let pos = 0;

    for (const match of filtered) {
        // Add text before match
        if (match.start > pos) {
            result += escapeHtml(text.slice(pos, match.start));
        }
        // Add highlighted match
        if (match.isHeading) {
            result += `<span class="sd-heading"><span class="sd-heading-marker">${escapeHtml(match.marker)}</span> <span class="sd-heading-text">${escapeHtml(match.headingText)}</span></span>`;
        } else {
            result += `<span class="${match.className}">${escapeHtml(match.text)}</span>`;
        }
        pos = match.end;
    }

    // Add remaining text
    if (pos < text.length) {
        result += escapeHtml(text.slice(pos));
    }

    return result;
}

/**
 * Create a syntax-highlighted editor
 *
 * Uses a two-layer approach:
 * - Background layer: syntax-highlighted preview (non-editable)
 * - Foreground layer: transparent textarea for actual editing
 */
function createStruckdownEditor(container, initialContent = '', options = {}) {
    const {onChange, onSave} = options;

    // Create wrapper
    const wrapper = document.createElement('div');
    wrapper.className = 'sd-editor-wrapper';
    wrapper.style.cssText = `
        position: relative;
        width: 100%;
        height: 100%;
        font-family: 'SF Mono', 'Menlo', 'Monaco', 'Consolas', monospace;
        font-size: 16px;
        line-height: 1.5;
    `;

    // Create highlight layer (background)
    const highlight = document.createElement('pre');
    highlight.className = 'sd-editor-highlight';
    highlight.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        margin: 0;
        padding: 10px;
        overflow: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        background: white;
        color: #333;
        border: none;
        pointer-events: none;
        font: inherit;
        line-height: inherit;
        box-sizing: border-box;
    `;

    // Create textarea (foreground, transparent text)
    const textarea = document.createElement('textarea');
    textarea.className = 'sd-editor-textarea';
    textarea.style.cssText = `
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        margin: 0;
        padding: 10px;
        border: none;
        resize: none;
        background: transparent;
        color: transparent;
        caret-color: #333;
        font: inherit;
        line-height: inherit;
        white-space: pre-wrap;
        word-wrap: break-word;
        overflow: auto;
        outline: none;
        box-sizing: border-box;
    `;
    textarea.spellcheck = false;
    textarea.value = initialContent;

    wrapper.appendChild(highlight);
    wrapper.appendChild(textarea);
    container.appendChild(wrapper);

    // Sync scroll positions
    textarea.addEventListener('scroll', () => {
        highlight.scrollTop = textarea.scrollTop;
        highlight.scrollLeft = textarea.scrollLeft;
    });

    // Update highlighting on input
    function updateHighlight() {
        highlight.innerHTML = highlightSyntax(textarea.value);
    }

    textarea.addEventListener('input', () => {
        updateHighlight();
        if (onChange) {
            onChange(textarea.value);
        }
    });

    // Handle keyboard shortcuts
    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            if (onSave) {
                onSave();
            }
        }
        // Handle Tab key for indentation
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 4;
            updateHighlight();
            if (onChange) {
                onChange(textarea.value);
            }
        }
    });

    // Double-click on {{var}} to open inputs panel and focus the field
    textarea.addEventListener('dblclick', (e) => {
        const position = textarea.selectionStart;
        const varName = getVariableAtPosition(textarea.value, position);

        if (varName) {
            // Open inputs panel
            const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(
                document.getElementById('inputs-offcanvas')
            );
            offcanvas.show();

            // Focus the corresponding input field after panel opens
            setTimeout(() => {
                const input = document.querySelector(`.input-field[name="${varName}"]`);
                if (input) {
                    input.focus();
                    input.select();
                }
            }, 300);
        }
    });

    // Initial highlight
    updateHighlight();

    // Return editor object
    return {
        _isCodeMirror: true,  // For compatibility with getSyntax()
        state: {
            doc: {
                toString: () => textarea.value
            }
        },
        textarea,
        highlight,
        wrapper,
        getValue: () => textarea.value,
        setValue: (value) => {
            textarea.value = value;
            updateHighlight();
        },
        focus: () => textarea.focus()
    };
}

// Export for use in playground
window.StruckdownEditor = {
    create: createStruckdownEditor
};
