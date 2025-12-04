// Struckdown Playground Editor JavaScript

let editor = null;
let analyseTimeout = null;
let currentInputs = [];
let sessionId = null;

// File watching and dirty state
let lastKnownMtime = null;
let lastSavedContent = null;
let isDirty = false;
let fileWatchInterval = null;
let pendingConflict = null;

// Session management - persist inputs per session
function getSessionId() {
    // Check URL hash first
    let hash = window.location.hash.slice(1);
    if (hash && hash.startsWith('s=')) {
        return hash.slice(2);
    }

    // Generate new session ID
    return 'sd_' + Math.random().toString(36).substring(2, 10);
}

function initSession() {
    sessionId = getSessionId();

    // Update URL hash if not already set
    if (!window.location.hash.includes('s=' + sessionId)) {
        window.location.hash = 's=' + sessionId;
    }
}

function getStorageKey() {
    return 'struckdown_inputs_' + sessionId;
}

function getPinnedSlotsKey() {
    return 'struckdown_pinned_' + sessionId;
}

function savePinnedSlots() {
    const pinnedSlots = [];
    document.querySelectorAll('.output-card.pinned').forEach(card => {
        const slot = card.dataset.slot;
        if (slot) pinnedSlots.push(slot);
    });
    localStorage.setItem(getPinnedSlotsKey(), JSON.stringify(pinnedSlots));
}

function loadPinnedSlots() {
    const stored = localStorage.getItem(getPinnedSlotsKey());
    if (stored) {
        try {
            return JSON.parse(stored);
        } catch (e) {
            return [];
        }
    }
    return [];
}

// Dirty state management
function setDirty(dirty) {
    isDirty = dirty;
    updateDirtyIndicator();
}

function updateDirtyIndicator() {
    const indicator = document.getElementById('dirty-indicator');
    if (indicator) {
        indicator.style.display = isDirty ? 'inline-block' : 'none';
    }
}

function markEditorDirty() {
    if (!isDirty) {
        setDirty(true);
    }
}

// File watching
function startFileWatching() {
    // Only in local mode
    if (document.getElementById('remote-mode').value === 'true') {
        console.log('File watching disabled (remote mode)');
        return;
    }

    console.log('Starting file watching...');

    // Poll every 2 seconds
    fileWatchInterval = setInterval(function() {
        checkFileStatus(false);
    }, 2000);

    // Get initial file state
    checkFileStatus(true);
}

function stopFileWatching() {
    if (fileWatchInterval) {
        clearInterval(fileWatchInterval);
        fileWatchInterval = null;
    }
}

function checkFileStatus(isInitial) {
    fetch('/api/file-status')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.log('File status error:', data.error);
                return;
            }

            if (isInitial === true) {
                // Initial load - just store the mtime and content
                lastKnownMtime = data.mtime;
                lastSavedContent = data.content;
                console.log('File watching initialized, mtime:', lastKnownMtime);
                return;
            }

            // Check if file changed externally
            if (lastKnownMtime !== null && data.mtime !== lastKnownMtime) {
                console.log('File changed externally, old mtime:', lastKnownMtime, 'new mtime:', data.mtime);
                handleExternalFileChange(data);
            } else {
                // Uncomment for verbose logging:
                // console.log('File check: no change, mtime:', data.mtime);
            }
        })
        .catch(err => {
            console.log('File status fetch error:', err);
        });
}

function handleExternalFileChange(data) {
    const currentContent = getSyntax();

    // If we have unsaved local changes
    if (isDirty && currentContent !== data.content) {
        // Show conflict modal
        pendingConflict = data;
        showConflictModal();
    } else {
        // No local changes or content is the same - just reload
        reloadFromFile(data);
    }
}

function reloadFromFile(data) {
    lastKnownMtime = data.mtime;
    lastSavedContent = data.content;

    // Update editor content
    if (editor._isCodeMirror && editor.setValue) {
        editor.setValue(data.content);
    } else if (editor.value !== undefined) {
        editor.value = data.content;
    }

    setDirty(false);
    analyseTemplate();
}

function showConflictModal() {
    const modal = new bootstrap.Modal(document.getElementById('conflict-modal'));
    modal.show();
}

function handleConflictKeepLocal() {
    // User wants to keep local changes - just update mtime so we don't warn again
    if (pendingConflict) {
        lastKnownMtime = pendingConflict.mtime;
    }
    pendingConflict = null;
    bootstrap.Modal.getInstance(document.getElementById('conflict-modal')).hide();
}

function handleConflictLoadExternal() {
    // User wants to discard local changes and load external version
    if (pendingConflict) {
        reloadFromFile(pendingConflict);
    }
    pendingConflict = null;
    bootstrap.Modal.getInstance(document.getElementById('conflict-modal')).hide();
}

function saveInputsToStorage() {
    const values = getInputValues();
    const model = document.getElementById('model-input').value;
    const data = { inputs: values, model: model };
    localStorage.setItem(getStorageKey(), JSON.stringify(data));
}

function loadInputsFromStorage() {
    const stored = localStorage.getItem(getStorageKey());
    if (stored) {
        try {
            return JSON.parse(stored);
        } catch (e) {
            return null;
        }
    }
    return null;
}

function restoreInputs() {
    const stored = loadInputsFromStorage();
    if (!stored) return;

    // Restore model
    if (stored.model) {
        document.getElementById('model-input').value = stored.model;
    }

    // Restore input field values
    if (stored.inputs) {
        Object.entries(stored.inputs).forEach(([name, value]) => {
            const field = document.querySelector(`.input-field[name="${name}"]`);
            if (field) {
                field.value = value;
            }
        });
    }
}

// Initialize editor (CodeMirror if available, textarea fallback)
function initEditor() {
    const container = document.getElementById('editor-container');
    const source = document.getElementById('syntax-source');
    const initialContent = source ? source.value : '';

    // Try to use CodeMirror if available
    if (window.StruckdownEditor) {
        try {
            editor = window.StruckdownEditor.create(container, initialContent, {
                onChange: function(content) {
                    clearTimeout(analyseTimeout);
                    analyseTimeout = setTimeout(analyseTemplate, 500);
                    markEditorDirty();
                },
                onSave: saveOnly
            });
            editor._isCodeMirror = true;

            // Initial analysis
            if (initialContent) {
                analyseTemplate();
            }
            return;
        } catch (e) {
            console.warn('CodeMirror init failed, falling back to textarea:', e);
        }
    }

    // Fallback: plain textarea
    const textarea = document.createElement('textarea');
    textarea.id = 'editor-textarea';
    textarea.className = 'form-control h-100 font-monospace';
    textarea.style.cssText = 'resize: none; border: none; border-radius: 0; font-size: 14px; line-height: 1.5;';
    textarea.value = initialContent;
    textarea.spellcheck = false;

    container.appendChild(textarea);
    editor = textarea;

    // Debounced analysis on input
    textarea.addEventListener('input', function() {
        clearTimeout(analyseTimeout);
        analyseTimeout = setTimeout(analyseTemplate, 500);
        markEditorDirty();
    });

    // Initial analysis
    if (initialContent) {
        analyseTemplate();
    }

    // Keyboard shortcut: Ctrl+S / Cmd+S to save
    textarea.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveAndRun();
        }
    });
}

// Get current syntax from editor
function getSyntax() {
    if (!editor) return '';
    // CodeMirror 6 uses state.doc.toString()
    if (editor._isCodeMirror) {
        return editor.state.doc.toString();
    }
    // Textarea fallback
    return editor.value;
}

// Analyse template for inputs and validation
function analyseTemplate() {
    const syntax = getSyntax();

    fetch('/api/analyse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({syntax: syntax})
    })
    .then(response => response.json())
    .then(data => {
        // Handle validation errors
        if (!data.valid && data.error) {
            showError(`Line ${data.error.line}: ${data.error.message}`);
        } else {
            hideError();
        }

        // Update inputs panel if inputs changed
        const newInputs = data.inputs_required || [];
        if (JSON.stringify(newInputs) !== JSON.stringify(currentInputs)) {
            currentInputs = newInputs;
            updateInputsPanel(newInputs);
        }
    })
    .catch(err => {
        console.error('Analysis error:', err);
    });
}

// Update inputs panel with new fields
function updateInputsPanel(inputsRequired) {
    const container = document.getElementById('inputs-container');

    // Merge current DOM values with stored values
    const currentValues = getInputValues();
    const stored = loadInputsFromStorage();
    const mergedValues = { ...(stored?.inputs || {}), ...currentValues };

    fetch('/partials/inputs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            inputs_required: inputsRequired,
            current_values: mergedValues
        })
    })
    .then(response => response.text())
    .then(html => {
        container.innerHTML = html;
        // Restore any stored values and add change listeners
        restoreInputs();
        addInputChangeListeners();
    })
    .catch(err => {
        console.error('Error updating inputs:', err);
    });
}

// Add change listeners to input fields to auto-save
function addInputChangeListeners() {
    document.querySelectorAll('.input-field').forEach(field => {
        field.addEventListener('input', debounce(saveInputsToStorage, 500));
    });
}

// Debounce helper
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Get current input values
function getInputValues() {
    const values = {};
    document.querySelectorAll('.input-field').forEach(field => {
        values[field.name] = field.value;
    });
    return values;
}

// Show error banner
function showError(message) {
    const banner = document.getElementById('error-banner');
    const messageEl = document.getElementById('error-message');
    messageEl.textContent = message;
    banner.style.display = 'flex';
    banner.classList.add('show');
}

// Hide error banner
function hideError() {
    const banner = document.getElementById('error-banner');
    banner.classList.remove('show');
    banner.style.display = 'none';
}

// Save only (without running)
function saveOnly() {
    const syntax = getSyntax();
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    if (remoteMode) {
        // In remote mode, just clear dirty state (nothing to save to disk)
        setDirty(false);
        setStatus('success', 'Saved');
        return;
    }

    setStatus('running', 'Saving...');

    fetch('/api/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({syntax: syntax})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (data.mtime) {
                lastKnownMtime = data.mtime;
            }
            lastSavedContent = syntax;
            setDirty(false);
            setStatus('success', 'Saved');
        } else {
            setStatus('error', 'Save failed');
        }
    })
    .catch(err => {
        console.error('Save error:', err);
        setStatus('error', 'Save failed: ' + err.message);
    });
}

// Save and run the prompt
function saveAndRun() {
    const syntax = getSyntax();
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    setStatus('running', 'Running...');
    disableSaveButton(true);

    // Save first (local mode only)
    const savePromise = remoteMode ?
        Promise.resolve() :
        fetch('/api/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({syntax: syntax})
        }).then(response => response.json()).then(data => {
            if (data.success) {
                // Update mtime and clear dirty state after successful save
                if (data.mtime) {
                    lastKnownMtime = data.mtime;
                }
                lastSavedContent = syntax;
                setDirty(false);
            }
        });

    savePromise
    .then(() => {
        if (mode === 'batch') {
            return runBatch(syntax);
        } else if (mode === 'file') {
            return runFile(syntax);
        } else {
            return runSingle(syntax);
        }
    })
    .catch(err => {
        console.error('Error:', err);
        setStatus('error', 'Error: ' + err.message);
        disableSaveButton(false);
    });
}

// Run single mode
function runSingle(syntax) {
    const inputs = getInputValues();
    const model = document.getElementById('model-input').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    const body = {
        syntax: syntax,
        inputs: inputs,
        model: model || null
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetch('/api/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .then(data => {
        disableSaveButton(false);

        if (data.error) {
            setStatus('error', 'Error: ' + data.error);
            renderSingleOutputs(data);
        } else {
            setStatus('success', 'Complete');
            renderSingleOutputs(data);
            updateCostDisplay(data.cost);
        }
    });
}

// Render single mode outputs
function renderSingleOutputs(data) {
    // Get slots in order
    fetch('/api/analyse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({syntax: getSyntax()})
    })
    .then(response => response.json())
    .then(analysisData => {
        fetch('/partials/outputs', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                outputs: data.outputs,
                error: data.error,
                cost: data.cost,
                slots_defined: analysisData.slots_defined
            })
        })
        .then(response => response.text())
        .then(html => {
            document.getElementById('outputs-container').innerHTML = html;
            initOutputInteractions();
        });
    });
}

// Initialize output card interactions
function initOutputInteractions() {
    // Restore pinned state from localStorage
    const pinnedSlots = loadPinnedSlots();
    if (pinnedSlots.length > 0) {
        document.querySelectorAll('.output-card').forEach(card => {
            const slot = card.dataset.slot;
            if (slot && pinnedSlots.includes(slot)) {
                card.classList.add('pinned');
                const button = card.querySelector('.pin-btn');
                if (button) {
                    button.classList.add('active');
                    const icon = button.querySelector('i');
                    if (icon) {
                        icon.classList.remove('bi-pin');
                        icon.classList.add('bi-pin-fill');
                    }
                }
            }
        });
        sortOutputCards();
    }
}

// Toggle pin state on a card
function togglePin(button) {
    const card = button.closest('.output-card');
    const isPinned = card.classList.toggle('pinned');

    // Update button appearance
    const icon = button.querySelector('i');
    if (isPinned) {
        icon.classList.remove('bi-pin');
        icon.classList.add('bi-pin-fill');
        button.classList.add('active');
    } else {
        icon.classList.remove('bi-pin-fill');
        icon.classList.add('bi-pin');
        button.classList.remove('active');
    }

    sortOutputCards();
    savePinnedSlots();
}

// Sort output cards (pinned first)
function sortOutputCards() {
    const container = document.getElementById('outputs-list');
    if (!container) return;

    const cards = Array.from(container.querySelectorAll('.output-card'));
    cards.sort((a, b) => {
        const aPinned = a.classList.contains('pinned');
        const bPinned = b.classList.contains('pinned');
        if (aPinned && !bPinned) return -1;
        if (!aPinned && bPinned) return 1;
        return 0;
    });

    cards.forEach(card => container.appendChild(card));
}

// Copy output to clipboard
function copyOutput(slotName) {
    const el = document.getElementById('output-' + slotName);
    if (el) {
        navigator.clipboard.writeText(el.textContent.trim());
    }
}

// Update cost display
function updateCostDisplay(cost) {
    const display = document.getElementById('cost-display');
    if (!cost) {
        display.style.display = 'none';
        return;
    }

    display.style.display = 'block';
    document.getElementById('cost-tokens').textContent =
        (cost.total_tokens || cost.input_tokens + cost.output_tokens || 0);
    document.getElementById('cost-amount').textContent =
        cost.total_cost ? '$' + cost.total_cost.toFixed(4) : '';
}

// Set status text
function setStatus(type, message) {
    const el = document.getElementById('status-text');
    el.textContent = message;
    el.className = 'status-' + type;
}

// Disable/enable save button
function disableSaveButton(disabled) {
    const btn = document.getElementById('save-run-btn');
    btn.disabled = disabled;
    if (disabled) {
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Running...';
    } else {
        btn.innerHTML = '<i class="bi bi-play-fill"></i> Save & Run';
    }
}

// Handle batch file upload
function handleBatchFileUpload(event) {
    const file = event.target.files[0];
    console.log('handleBatchFileUpload called, file:', file?.name);
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setStatus('running', 'Uploading file...');

    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Upload response:', data);
        if (data.error) {
            setStatus('error', 'Upload error: ' + data.error);
            return;
        }

        document.getElementById('current-file-id').value = data.file_id;
        console.log('Set current-file-id to:', data.file_id);
        document.getElementById('batch-filename').textContent = data.filename;
        document.getElementById('batch-row-count').textContent = data.row_count;
        document.getElementById('batch-file-info').style.display = 'block';

        setStatus('success', 'File uploaded: ' + data.row_count + ' rows');
    })
    .catch(err => {
        console.error('Upload error:', err);
        setStatus('error', 'Upload failed: ' + err.message);
    });
}

// Clear batch file
function clearBatchFile() {
    document.getElementById('current-file-id').value = '';
    document.getElementById('batch-file').value = '';
    document.getElementById('batch-file-info').style.display = 'none';
}

// Handle source file upload (file mode)
function handleSourceFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setStatus('running', 'Uploading file...');

    fetch('/api/upload-source', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            setStatus('error', 'Upload error: ' + data.error);
            return;
        }

        document.getElementById('current-source-file-id').value = data.file_id;
        document.getElementById('source-filename').textContent = data.filename;
        document.getElementById('source-file-size').textContent = formatFileSize(data.size);
        document.getElementById('source-file-info').style.display = 'block';

        setStatus('success', 'File uploaded: ' + data.filename);
    })
    .catch(err => {
        setStatus('error', 'Upload failed: ' + err.message);
    });
}

// Clear source file
function clearSourceFile() {
    document.getElementById('current-source-file-id').value = '';
    document.getElementById('source-file').value = '';
    document.getElementById('source-file-info').style.display = 'none';
}

// Format file size for display
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Run file mode
function runFile(syntax) {
    const fileId = document.getElementById('current-source-file-id').value;
    if (!fileId) {
        setStatus('error', 'No file uploaded');
        disableSaveButton(false);
        return Promise.reject(new Error('No file uploaded'));
    }

    const model = document.getElementById('model-input').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    const body = {
        syntax: syntax,
        file_id: fileId,
        model: model || null
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetch('/api/run-file', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .then(data => {
        disableSaveButton(false);

        if (data.error) {
            setStatus('error', 'Error: ' + data.error);
            renderSingleOutputs(data);
        } else {
            setStatus('success', 'Complete');
            renderSingleOutputs(data);
            updateCostDisplay(data.cost);
        }
    });
}

// Run batch mode
function runBatch(syntax) {
    const fileId = document.getElementById('current-file-id').value;
    console.log('runBatch called, fileId:', fileId);
    if (!fileId) {
        console.error('No file ID found in #current-file-id');
        setStatus('error', 'No file uploaded');
        disableSaveButton(false);
        return Promise.reject(new Error('No file uploaded'));
    }

    const model = document.getElementById('model-input').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    const body = {
        syntax: syntax,
        file_id: fileId,
        model: model || null
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetch('/api/run-batch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            setStatus('error', 'Error: ' + data.error);
            disableSaveButton(false);
            return;
        }

        document.getElementById('current-task-id').value = data.task_id;
        initBatchStream(data.task_id);
    });
}
