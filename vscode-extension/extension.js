const vscode = require('vscode');

// color schemes based on Travel palette
// using rgba for whole-line backgrounds so cursor highlight shows through
const COLORS = {
    light: {
        checkpoint: { bg: 'rgba(252, 220, 220, 0.7)', fg: '#000000', bold: true },
        system: { bg: 'rgba(248, 248, 248, 0.6)' },
        systemTagLine: { bg: 'rgba(232, 232, 232, 0.7)' },
        systemTag: { bg: null, fg: '#000000', bold: true },
        slot: { bg: '#D4EDDA', fg: '#155724' },           // green for [[slot]]
        action: { bg: '#EDE7F6', fg: '#6A1B9A' },         // purple for [[@action]]
        template: { bg: '#FBDCE8', fg: '#9B2C5A' },       // pink for {{var}}
        break: { bg: '#FCDCDC', fg: '#F23030' },
        include: { bg: '#D6F5F7', fg: '#000000', bold: false }
    },
    dark: {
        checkpoint: { bg: 'rgba(200, 50, 50, 0.7)', fg: '#FFFFFF', bold: true },
        system: { bg: 'rgba(35, 35, 35, 0.6)' },
        systemTagLine: { bg: 'rgba(51, 51, 51, 0.7)' },
        systemTag: { bg: null, fg: '#FFFFFF', bold: true },
        slot: { bg: '#1E5128', fg: '#98FB98' },           // green for [[slot]]
        action: { bg: '#4A148C', fg: '#CE93D8' },         // purple for [[@action]]
        template: { bg: '#5E2750', fg: '#FFB6C1' },       // pink for {{var}}
        break: { bg: '#F23030', fg: '#FFFFFF' },
        include: { bg: '#23B7D9', fg: '#000000', bold: false }
    }
};

function activate(context) {
    let decorations = {};

    function createDecorations() {
        // dispose old decorations
        Object.values(decorations).forEach(d => d && d.dispose());

        const isDark = vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.Dark ||
                       vscode.window.activeColorTheme.kind === vscode.ColorThemeKind.HighContrastDark;
        const colors = isDark ? COLORS.dark : COLORS.light;

        decorations = {
            checkpoint: vscode.window.createTextEditorDecorationType({
                isWholeLine: true,
                backgroundColor: colors.checkpoint.bg,
                color: colors.checkpoint.fg,
                fontWeight: colors.checkpoint.bold ? 'bold' : 'normal'
            }),
            system: vscode.window.createTextEditorDecorationType({
                isWholeLine: true,
                backgroundColor: colors.system.bg
            }),
            systemTagLine: vscode.window.createTextEditorDecorationType({
                isWholeLine: true,
                backgroundColor: colors.systemTagLine.bg
            }),
            systemTag: vscode.window.createTextEditorDecorationType({
                ...(colors.systemTag.bg && { backgroundColor: colors.systemTag.bg }),
                color: colors.systemTag.fg,
                fontWeight: colors.systemTag.bold ? 'bold' : 'normal'
            }),
            slot: vscode.window.createTextEditorDecorationType({
                backgroundColor: colors.slot.bg,
                color: colors.slot.fg,
                borderRadius: '3px',
                fontWeight: 'bold'
            }),
            action: vscode.window.createTextEditorDecorationType({
                backgroundColor: colors.action.bg,
                color: colors.action.fg,
                borderRadius: '3px',
                fontWeight: 'bold'
            }),
            template: vscode.window.createTextEditorDecorationType({
                backgroundColor: colors.template.bg,
                color: colors.template.fg,
                borderRadius: '3px',
                fontWeight: 'bold'
            }),
            break: vscode.window.createTextEditorDecorationType({
                backgroundColor: colors.break.bg,
                color: colors.break.fg,
                borderRadius: '3px'
            }),
            include: vscode.window.createTextEditorDecorationType({
                backgroundColor: colors.include.bg,
                color: colors.include.fg,
                fontWeight: colors.include.bold ? 'bold' : 'normal',
                borderRadius: '3px'
            })
        };
    }

    createDecorations();

    // recreate decorations when theme changes
    vscode.window.onDidChangeActiveColorTheme(() => {
        createDecorations();
        if (vscode.window.activeTextEditor) {
            updateDecorations(vscode.window.activeTextEditor);
        }
    }, null, context.subscriptions);

    function updateDecorations(editor) {
        if (!editor || editor.document.languageId !== 'struckdown') return;

        const text = editor.document.getText();
        const checkpointRanges = [];
        const systemRanges = [];
        const systemTagLineRanges = [];
        const systemTagRanges = [];
        const slotRanges = [];
        const actionRanges = [];
        const templateRanges = [];
        const breakRanges = [];
        const includeRanges = [];
        const lines = text.split('\n');

        let inSystem = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const lineRange = new vscode.Range(i, 0, i, line.length);

            const hasOpenTag = /<system/.test(line);
            const hasCloseTag = /<\/system>/.test(line);

            // track system blocks (multi-line, whole line)
            if (hasOpenTag) {
                inSystem = true;
            }

            if (inSystem) {
                // lines with tags get darker bg, content lines get lighter bg
                if (hasOpenTag || hasCloseTag) {
                    systemTagLineRanges.push({ range: lineRange });
                } else {
                    systemRanges.push({ range: lineRange });
                }
            }

            if (hasCloseTag) {
                inSystem = false;
            }

            // find <system...> opening tags (inline highlight for bold text)
            const systemOpenRegex = /<system[^>]*>/g;
            let match;
            while ((match = systemOpenRegex.exec(line)) !== null) {
                const start = new vscode.Position(i, match.index);
                const end = new vscode.Position(i, match.index + match[0].length);
                systemTagRanges.push({ range: new vscode.Range(start, end) });
            }

            // find </system> closing tags (inline highlight for bold text)
            const systemCloseRegex = /<\/system>/g;
            while ((match = systemCloseRegex.exec(line)) !== null) {
                const start = new vscode.Position(i, match.index);
                const end = new vscode.Position(i, match.index + match[0].length);
                systemTagRanges.push({ range: new vscode.Range(start, end) });
            }

            // checkpoint - whole line
            if (/<checkpoint/.test(line)) {
                checkpointRanges.push({ range: lineRange });
            }

            // find <include .../> tags (inline highlight)
            const includeRegex = /<include\s+[^>]*\/>/g;
            while ((match = includeRegex.exec(line)) !== null) {
                const start = new vscode.Position(i, match.index);
                const end = new vscode.Position(i, match.index + match[0].length);
                includeRanges.push({ range: new vscode.Range(start, end) });
            }

            // find {{...}} template variables (pink)
            const templateRegex = /\{\{[^}]+\}\}/g;
            while ((match = templateRegex.exec(line)) !== null) {
                const start = new vscode.Position(i, match.index);
                const end = new vscode.Position(i, match.index + match[0].length);
                templateRanges.push({ range: new vscode.Range(start, end) });
            }

            // find all [[...]] and categorize them
            const bracketRegex = /\[\[([^\]]+)\]\]/g;
            while ((match = bracketRegex.exec(line)) !== null) {
                const start = new vscode.Position(i, match.index);
                const end = new vscode.Position(i, match.index + match[0].length);
                const content = match[1];

                if (content.startsWith('@break')) {
                    breakRanges.push({ range: new vscode.Range(start, end) });
                } else if (content.startsWith('@')) {
                    // action - blue
                    actionRanges.push({ range: new vscode.Range(start, end) });
                } else {
                    // slot - green
                    slotRanges.push({ range: new vscode.Range(start, end) });
                }
            }
        }

        editor.setDecorations(decorations.system, systemRanges);
        editor.setDecorations(decorations.systemTagLine, systemTagLineRanges);
        editor.setDecorations(decorations.systemTag, systemTagRanges);
        editor.setDecorations(decorations.checkpoint, checkpointRanges);
        editor.setDecorations(decorations.slot, slotRanges);
        editor.setDecorations(decorations.action, actionRanges);
        editor.setDecorations(decorations.template, templateRanges);
        editor.setDecorations(decorations.break, breakRanges);
        editor.setDecorations(decorations.include, includeRanges);
    }

    // update on editor change
    vscode.window.onDidChangeActiveTextEditor(updateDecorations, null, context.subscriptions);
    vscode.workspace.onDidChangeTextDocument(e => {
        const editor = vscode.window.activeTextEditor;
        if (editor && e.document === editor.document) {
            updateDecorations(editor);
        }
    }, null, context.subscriptions);

    // initial update
    if (vscode.window.activeTextEditor) {
        updateDecorations(vscode.window.activeTextEditor);
    }
}

function deactivate() {}

module.exports = { activate, deactivate };
