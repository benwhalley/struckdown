// Struckdown Playground Editor JavaScript

let editor = null;
let analyseTimeout = null;
let autosaveTimeout = null;
let currentInputs = [];
let currentSlots = [];
let sessionId = null;

// Autosave interval (ms) - only in remote mode
const AUTOSAVE_INTERVAL = 60000;

// CSRF protection
function getCsrfToken() {
    let token = localStorage.getItem('struckdown_csrf_token');
    if (!token) {
        token = crypto.randomUUID();
        localStorage.setItem('struckdown_csrf_token', token);
    }
    return token;
}

function fetchWithCsrf(url, options = {}) {
    options.headers = options.headers || {};
    options.headers['X-CSRF-Token'] = getCsrfToken();
    return fetch(url, options);
}

// File watching and dirty state
let lastKnownMtime = null;
let lastSavedContent = null;
let isDirty = false;
let fileWatchInterval = null;
let pendingConflict = null;

// Multi-file support
let currentFilePath = null;
let fileBrowserModal = null;
let pendingFileSwitchPath = null;

// Session management - persist inputs per session
function getHashParams() {
    const hash = window.location.hash.slice(1);
    const params = {};
    hash.split('&').forEach(part => {
        const [key, value] = part.split('=');
        if (key && value) {
            params[key] = decodeURIComponent(value);
        }
    });
    return params;
}

function setHashParams(params) {
    const parts = [];
    for (const [key, value] of Object.entries(params)) {
        if (value) {
            parts.push(key + '=' + encodeURIComponent(value));
        }
    }
    window.location.hash = parts.join('&');
}

function getSessionId() {
    const params = getHashParams();
    if (params.s) {
        return params.s;
    }
    // Generate new session ID
    return 'sd_' + Math.random().toString(36).substring(2, 10);
}

function initSession() {
    sessionId = getSessionId();
    updateHashWithCurrentState();
}

function updateHashWithCurrentState() {
    const params = getHashParams();
    params.s = sessionId;
    if (currentFilePath) {
        params.f = currentFilePath;
    }
    setHashParams(params);
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

// Warn before leaving page with unsaved changes
window.addEventListener('beforeunload', function(e) {
    if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
        return '';
    }
});

function markEditorDirty() {
    if (!isDirty) {
        setDirty(true);
    }
    // Schedule autosave in remote mode
    scheduleAutosave();
}

// Schedule autosave (debounced, remote mode only)
function scheduleAutosave() {
    const remoteMode = document.getElementById('remote-mode').value === 'true';
    if (!remoteMode) return;

    // Clear any pending autosave
    if (autosaveTimeout) {
        clearTimeout(autosaveTimeout);
    }

    // Schedule new autosave
    autosaveTimeout = setTimeout(function() {
        if (isDirty) {
            autosave();
        }
    }, AUTOSAVE_INTERVAL);
}

// Perform autosave (remote mode only)
function autosave() {
    const remoteMode = document.getElementById('remote-mode').value === 'true';
    if (!remoteMode || !isDirty) return;

    savePromptAndUpdateUrl().then(() => {
        setDirty(false);
        // Don't show status for autosave to avoid distraction
        console.log('Autosaved');
    }).catch(err => {
        console.warn('Autosave failed:', err.message);
    });
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
    // Skip if no file is open
    if (!currentFilePath) {
        return;
    }

    // Use the file-specific endpoint
    fetch('/api/files/' + encodeURIComponent(currentFilePath))
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
                    analyseTimeout = setTimeout(analyseTemplate, 4000);
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
        analyseTimeout = setTimeout(analyseTemplate, 4000);
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

    fetchWithCsrf('/api/analyse', {
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

        // Check if inputs or uses_history changed
        const newInputs = data.inputs_required || [];
        const newUsesHistory = data.uses_history || false;
        const inputsChanged = JSON.stringify(newInputs) !== JSON.stringify(currentInputs);
        const historyChanged = newUsesHistory !== usesHistory;

        if (inputsChanged || historyChanged) {
            currentInputs = newInputs;
            updateInputsPanel(newInputs, newUsesHistory);
        }

        // Store slots for incremental rendering
        currentSlots = data.slots_defined || [];
    })
    .catch(err => {
        console.error('Analysis error:', err);
    });
}

// Update inputs panel with new fields
function updateInputsPanel(inputsRequired, usesHistoryFlag = false) {
    const container = document.getElementById('inputs-container');

    // Merge current DOM values with stored values
    const currentValues = getInputValues();
    const stored = loadInputsFromStorage();
    const mergedValues = { ...(stored?.inputs || {}), ...currentValues };

    // Preserve chat history seed if it exists
    const seedEl = document.getElementById('chat-history-seed');
    if (seedEl) {
        mergedValues['_chat_history_seed'] = seedEl.value;
    }

    // Update global uses_history flag
    usesHistory = usesHistoryFlag;

    fetchWithCsrf('/partials/inputs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            inputs_required: inputsRequired,
            current_values: mergedValues,
            uses_history: usesHistoryFlag
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
        // Clear missing highlight when user starts typing
        field.addEventListener('input', function() {
            const container = this.closest('div');
            if (container && container.classList.contains('input-missing')) {
                container.classList.remove('input-missing');
                const indicator = container.querySelector('.required-indicator');
                if (indicator) indicator.remove();
            }
        });
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
        const name = field.name;
        let value = field.value;

        // Skip empty strings - don't include in context so Jinja treats as undefined
        if (value === '' || value.trim() === '') {
            return;
        }

        // Check if JSON toggle is checked for this field
        const jsonToggle = document.getElementById(`json-${name}`);
        if (jsonToggle && jsonToggle.checked) {
            try {
                value = JSON.parse(value);
            } catch (e) {
                // If JSON parse fails, keep as string
                console.warn(`Failed to parse JSON for ${name}:`, e.message);
            }
        }

        values[name] = value;
    });
    return values;
}

// Get strict mode setting
function getStrictMode() {
    const toggle = document.getElementById('strict-mode-toggle');
    return toggle ? toggle.checked : false;
}

// Get list of required inputs that have empty values
function getMissingInputs() {
    const values = getInputValues();
    // Since getInputValues now skips empty strings, we just check if the key exists
    return currentInputs.filter(name => values[name] === undefined);
}

// Highlight missing inputs in the inputs panel
function highlightMissingInputs(missingVars) {
    // First clear all existing highlights
    clearMissingInputHighlights();

    // Add highlight to missing inputs
    missingVars.forEach(name => {
        const field = document.querySelector(`.input-field[name="${name}"]`);
        if (field) {
            const container = field.closest('div');
            if (container) {
                container.classList.add('input-missing');
                // Find the label and add required indicator if not already present
                const label = container.querySelector('label');
                if (label && !label.querySelector('.required-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'required-indicator';
                    indicator.textContent = ' * required';
                    label.appendChild(indicator);
                }
            }
        }
    });
}

// Clear all missing input highlights
function clearMissingInputHighlights() {
    document.querySelectorAll('.input-missing').forEach(el => {
        el.classList.remove('input-missing');
    });
    document.querySelectorAll('.required-indicator').forEach(el => {
        el.remove();
    });
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
        // In remote mode, save to backend and update URL
        setStatus('running', 'Saving...');
        savePromptAndUpdateUrl().then((result) => {
            setDirty(false);
            if (result && result.warning) {
                setStatus('warning', result.warning);
            } else {
                setStatus('success', 'Saved');
            }
        }).catch(err => {
            setStatus('error', 'Save failed: ' + err.message);
        });
        return;
    }

    // Check if we have a file to save to
    if (!currentFilePath) {
        setStatus('error', 'No file open');
        return;
    }

    setStatus('running', 'Saving...');

    // Use the new file-specific save endpoint
    fetchWithCsrf('/api/files/' + encodeURIComponent(currentFilePath), {
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
            setStatus('error', 'Save failed: ' + (data.error || 'Unknown error'));
        }
    })
    .catch(err => {
        console.error('Save error:', err);
        setStatus('error', 'Save failed: ' + err.message);
    });
}

// Save prompt to backend and update URL (remote mode only)
// Called after successful run or save to create a shareable link
function savePromptAndUpdateUrl() {
    const remoteMode = document.getElementById('remote-mode').value === 'true';
    if (!remoteMode) return Promise.resolve();

    const syntax = getSyntax();
    if (!syntax || !syntax.trim()) {
        return Promise.reject(new Error('Empty prompt'));
    }

    return fetchWithCsrf('/api/save-prompt', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({syntax: syntax})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        if (data.prompt_id) {
            // Update URL without reloading
            const newUrl = '/p/' + data.prompt_id;
            history.pushState({promptId: data.prompt_id}, '', newUrl);
            // Update hidden field
            const hiddenField = document.getElementById('current-prompt-id');
            if (hiddenField) {
                hiddenField.value = data.prompt_id;
            }
        }
        // Return both prompt_id and warning (if any)
        return { prompt_id: data.prompt_id, warning: data.warning };
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
    let savePromise;
    if (remoteMode) {
        savePromise = Promise.resolve();
    } else if (currentFilePath) {
        // Use file-specific save endpoint
        savePromise = fetchWithCsrf('/api/files/' + encodeURIComponent(currentFilePath), {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({syntax: syntax})
        }).then(response => response.json()).then(data => {
            if (data.success) {
                if (data.mtime) {
                    lastKnownMtime = data.mtime;
                }
                lastSavedContent = syntax;
                setDirty(false);
            }
        });
    } else {
        // No file to save - skip save step
        savePromise = Promise.resolve();
    }

    savePromise
    .then(() => {
        if (mode === 'batch') {
            return runBatch(syntax);
        } else if (mode === 'file') {
            return runFile(syntax);
        } else {
            // Use incremental mode by default for single execution
            // This shows slot results as they complete
            return runSingleIncremental(syntax);
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

    // Check for missing required inputs
    const missing = getMissingInputs();
    if (missing.length > 0) {
        // Highlight missing inputs
        highlightMissingInputs(missing);

        // Open inputs panel
        const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(
            document.getElementById('inputs-offcanvas')
        );
        offcanvas.show();

        // Show error message
        setStatus('error', 'Missing inputs: ' + missing.join(', '));
        disableSaveButton(false);
        return Promise.resolve();
    }

    const body = {
        syntax: syntax,
        inputs: inputs,
        model: model || null,
        strict_undefined: getStrictMode()
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetchWithCsrf('/api/run', {
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
            // Save prompt and update URL (remote mode only)
            savePromptAndUpdateUrl().catch(err => {
                console.warn('Failed to save prompt URL:', err.message);
            });
        }
    });
}

// Run single mode with incremental (streaming) results
function runSingleIncremental(syntax) {
    const inputs = getInputValues();
    const model = document.getElementById('model-input').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    // Check for missing required inputs
    const missing = getMissingInputs();
    if (missing.length > 0) {
        highlightMissingInputs(missing);
        const offcanvas = bootstrap.Offcanvas.getOrCreateInstance(
            document.getElementById('inputs-offcanvas')
        );
        offcanvas.show();
        setStatus('error', 'Missing inputs: ' + missing.join(', '));
        disableSaveButton(false);
        return Promise.resolve();
    }

    const body = {
        syntax: syntax,
        inputs: inputs,
        model: model || null,
        strict_undefined: getStrictMode(),
        session_id: sessionId  // For evidence loading
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    // Track incremental results
    const incrementalResults = {
        outputs: {},
        cost: null,
        error: null,
        slotsCompleted: []
    };

    // Clear previous outputs and show placeholders for expected slots
    const container = document.getElementById('outputs-container');
    container.innerHTML = '';

    if (currentSlots.length > 0) {
        currentSlots.forEach(slotKey => {
            createPendingSlotCard(slotKey);
        });
    } else {
        container.innerHTML = `
            <p class="text-muted text-center mt-5">
                <i class="bi bi-hourglass-split"></i> Processing...
            </p>
        `;
    }

    // Show initial pending state
    setStatus('running', 'Processing...');

    // Use POST to start the stream, passing data as query params is not ideal for large payloads
    // So we'll use the existing CSRF token approach
    return fetch('/api/run-incremental', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken()
        },
        body: JSON.stringify(body)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Failed to start incremental run');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function processStream() {
            return reader.read().then(({done, value}) => {
                if (done) {
                    disableSaveButton(false);
                    return;
                }

                buffer += decoder.decode(value, {stream: true});

                // SSE events are separated by double newlines
                // Only process complete events, keep partial data in buffer
                const events = buffer.split('\n\n');

                // Keep the last part in buffer (may be incomplete)
                buffer = events.pop() || '';

                for (const eventBlock of events) {
                    if (!eventBlock.trim()) continue;

                    let eventType = null;
                    let eventData = null;

                    for (const line of eventBlock.split('\n')) {
                        if (line.startsWith('event: ')) {
                            eventType = line.substring(7).trim();
                        } else if (line.startsWith('data: ')) {
                            eventData = line.substring(6);
                        }
                    }

                    if (eventType && eventData) {
                        try {
                            const data = JSON.parse(eventData);
                            handleIncrementalEvent(eventType, data, incrementalResults);
                        } catch (e) {
                            console.error('Failed to parse event data:', eventData, e);
                        }
                    }
                }

                return processStream();
            });
        }

        return processStream();
    }).catch(error => {
        console.error('Incremental run error:', error);
        setStatus('error', 'Error: ' + error.message);
        disableSaveButton(false);
    });
}

// Handle incremental SSE events
function handleIncrementalEvent(eventType, data, results) {
    switch (eventType) {
        case 'slot_completed':
            // Store the result
            results.outputs[data.slot_key] = data.result.output;
            results.slotsCompleted.push(data.slot_key);
            // Update status
            setStatus('running', `Completed: ${data.slot_key} (${results.slotsCompleted.length} slots)`);
            // Update outputs display incrementally
            renderIncrementalSlot(data.slot_key, data.result);
            break;

        case 'checkpoint':
            setStatus('running', `Checkpoint ${data.segment_index + 1} reached`);
            break;

        case 'complete':
            // Final result
            setStatus('success', 'Complete');
            if (data.result && data.result.results) {
                // Convert results to outputs format
                const outputs = {};
                for (const [key, segResult] of Object.entries(data.result.results)) {
                    outputs[key] = segResult.output;
                }
                const cost = {
                    total_cost: data.result.total_cost,
                    prompt_tokens: data.result.prompt_tokens,
                    completion_tokens: data.result.completion_tokens
                };
                updateCostDisplay(cost);
            }
            // Mark any slots that weren't executed as skipped
            markSkippedSlots(results.slotsCompleted);
            disableSaveButton(false);
            // Save prompt and update URL (remote mode only)
            savePromptAndUpdateUrl().catch(err => {
                console.warn('Failed to save prompt URL:', err.message);
            });
            break;

        case 'error':
            setStatus('error', 'Error: ' + data.error_message);
            // Mark remaining slots as skipped on error
            markSkippedSlots(results.slotsCompleted);
            disableSaveButton(false);
            break;
    }
}

// Create a pending placeholder card for a slot
function createPendingSlotCard(slotKey) {
    const container = document.getElementById('outputs-container');
    const cardHtml = `
        <div class="output-card card mb-2 output-card-pending" data-slot="${slotKey}">
            <div class="card-header py-2 d-flex justify-content-between align-items-center">
                <div class="d-flex align-items-center">
                    <button type="button" class="btn btn-sm pin-btn me-2"
                            onclick="togglePin(this)" title="Pin to top">
                        <i class="bi bi-pin"></i>
                    </button>
                    <code class="slot-name">${slotKey}</code>
                </div>
                <span class="spinner-border spinner-border-sm text-muted" role="status"></span>
            </div>
            <div class="card-body py-2">
                <div class="output-value text-muted" id="output-${slotKey}">
                    <i>Waiting...</i>
                </div>
            </div>
        </div>
    `;
    container.insertAdjacentHTML('beforeend', cardHtml);
}

// Mark remaining pending cards as skipped
function markSkippedSlots(completedSlots) {
    document.querySelectorAll('.output-card-pending').forEach(card => {
        const slotKey = card.dataset.slot;
        if (!completedSlots.includes(slotKey)) {
            card.classList.remove('output-card-pending');
            card.classList.add('output-card-skipped');
            const spinner = card.querySelector('.spinner-border');
            if (spinner) {
                spinner.outerHTML = '<span class="badge bg-secondary">Skipped</span>';
            }
            const valueEl = card.querySelector('.output-value');
            if (valueEl) {
                valueEl.innerHTML = '<i class="text-muted">Conditional not executed</i>';
            }
        }
    });
}

// Render a single slot result incrementally
function renderIncrementalSlot(slotKey, result) {
    // Find or create the output card for this slot
    let card = document.querySelector(`.output-card[data-slot="${slotKey}"]`);

    // Clear the "Processing..." placeholder on first slot
    const container = document.getElementById('outputs-container');
    const placeholder = container.querySelector('p.text-muted');
    if (placeholder) {
        placeholder.remove();
    }

    if (!card) {
        // Create a new card matching the template structure
        const cardHtml = `
            <div class="output-card card mb-2" data-slot="${slotKey}">
                <div class="card-header py-2 d-flex justify-content-between align-items-center">
                    <div class="d-flex align-items-center">
                        <button type="button" class="btn btn-sm pin-btn me-2"
                                onclick="togglePin(this)" title="Pin to top">
                            <i class="bi bi-pin"></i>
                        </button>
                        <code class="slot-name">${slotKey}</code>
                    </div>
                    <button class="btn btn-link btn-sm text-muted p-0" onclick="copyOutput('${slotKey}')"
                            title="Copy to clipboard">
                        <i class="bi bi-clipboard"></i>
                    </button>
                </div>
                <div class="card-body py-2">
                    <div class="output-value" id="output-${slotKey}">${escapeHtml(formatOutput(result.output))}</div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', cardHtml);
    } else {
        // Update existing card (may be a pending placeholder)
        card.classList.remove('output-card-pending');

        // Replace spinner with copy button
        const spinner = card.querySelector('.spinner-border');
        if (spinner) {
            spinner.outerHTML = `
                <button class="btn btn-link btn-sm text-muted p-0" onclick="copyOutput('${slotKey}')"
                        title="Copy to clipboard">
                    <i class="bi bi-clipboard"></i>
                </button>
            `;
        }

        // Update value
        const valueEl = card.querySelector('.output-value');
        if (valueEl) {
            valueEl.classList.remove('text-muted');
            valueEl.innerHTML = escapeHtml(formatOutput(result.output));
        }
    }
}

// Format output value for display
function formatOutput(value) {
    if (value === null || value === undefined) {
        return 'null';
    }
    if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
    }
    return String(value);
}

// Escape HTML for safe insertion
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Render single mode outputs
function renderSingleOutputs(data) {
    // Get slots in order
    fetchWithCsrf('/api/analyse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({syntax: getSyntax()})
    })
    .then(response => response.json())
    .then(analysisData => {
        fetchWithCsrf('/partials/outputs', {
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

// Disable/enable run button
function disableSaveButton(disabled) {
    const btn = document.getElementById('run-btn');
    if (!btn) return;
    btn.disabled = disabled;
    if (disabled) {
        btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Running...';
    } else {
        btn.innerHTML = '<i class="bi bi-play-fill"></i> Run';
    }
}

// Handle batch file upload (supports multiple files)
function handleBatchFileUpload(event) {
    const files = event.target.files;
    console.log('handleBatchFileUpload called, files:', files.length);
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append('file', file);
    }

    const uploadMsg = files.length > 1
        ? `Uploading ${files.length} files...`
        : 'Uploading file...';
    setStatus('running', uploadMsg);

    fetchWithCsrf('/api/upload', {
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

        let statusMsg = 'Uploaded: ' + data.row_count + ' rows';
        if (data.warning) {
            statusMsg += ' (Warning: ' + data.warning + ')';
        }
        setStatus('success', statusMsg);
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

    fetchWithCsrf('/api/upload-source', {
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

// ============================================
// Evidence Files (for @evidence action)
// ============================================

// Handle evidence file upload
function handleEvidenceUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append('file', file);
    }
    formData.append('session_id', sessionId);

    setStatus('running', 'Uploading evidence files...');

    fetchWithCsrf('/api/upload-evidence', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Clear the file input
        event.target.value = '';

        if (data.error) {
            setStatus('error', 'Evidence upload error: ' + data.error);
            return;
        }

        // Show success with chunk count
        const fileNames = data.files.map(f => f.filename).join(', ');
        const totalChunks = data.files.reduce((sum, f) => sum + f.chunk_count, 0);
        setStatus('success', `Uploaded: ${fileNames} (${totalChunks} chunks)`);

        if (data.warnings) {
            console.warn('Evidence upload warnings:', data.warnings);
        }

        // Refresh the evidence list
        loadEvidenceList();
    })
    .catch(err => {
        setStatus('error', 'Evidence upload failed: ' + err.message);
    });
}

// Load and render evidence file list
function loadEvidenceList() {
    fetch('/api/evidence?session_id=' + encodeURIComponent(sessionId))
        .then(response => response.json())
        .then(data => {
            renderEvidenceList(data.files || []);
        })
        .catch(err => {
            console.error('Failed to load evidence list:', err);
        });
}

// Render evidence file list
function renderEvidenceList(files) {
    const container = document.getElementById('evidence-list');
    const countBadge = document.getElementById('evidence-count');

    countBadge.textContent = files.length;

    if (files.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = files.map(f => `
        <div class="d-flex justify-content-between align-items-center mb-1">
            <span class="small">
                <i class="bi bi-file-text"></i> ${escapeHtml(f.filename)}
                <span class="text-muted">(${f.chunk_count} chunks)</span>
            </span>
            <button class="btn btn-link btn-sm text-danger p-0"
                    onclick="deleteEvidence('${f.file_id}')"
                    title="Remove">
                <i class="bi bi-x"></i>
            </button>
        </div>
    `).join('');
}

// Delete an evidence file
function deleteEvidence(fileId) {
    fetchWithCsrf('/api/evidence/' + encodeURIComponent(fileId) + '?session_id=' + encodeURIComponent(sessionId), {
        method: 'DELETE'
    })
    .then(response => response.json())
    .then(data => {
        if (data.deleted) {
            loadEvidenceList();
        }
    })
    .catch(err => {
        console.error('Failed to delete evidence:', err);
    });
}

// Escape HTML for safe rendering
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
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
        model: model || null,
        strict_undefined: getStrictMode()
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetchWithCsrf('/api/run-file', {
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
            // Save prompt and update URL (remote mode only)
            savePromptAndUpdateUrl().catch(err => {
                console.warn('Failed to save prompt URL:', err.message);
            });
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
        model: model || null,
        strict_undefined: getStrictMode()
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    return fetchWithCsrf('/api/run-batch', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })
    .then(response => {
        console.log('run-batch response status:', response.status);
        return response.json();
    })
    .then(data => {
        console.log('run-batch response data:', data);
        if (data.error) {
            setStatus('error', 'Error: ' + data.error);
            disableSaveButton(false);
            return;
        }

        document.getElementById('current-task-id').value = data.task_id;
        // Small delay to ensure task is fully initialised on server
        setTimeout(() => initBatchStream(data.task_id), 100);
    })
    .catch(err => {
        console.error('run-batch fetch error:', err);
        setStatus('error', 'Error: ' + err.message);
        disableSaveButton(false);
    });
}

// ============================================
// Multi-file support functions
// ============================================

function initFileBrowser() {
    // Initialize current file path from hidden input (server-provided initial file)
    const pathInput = document.getElementById('current-file-path');
    const initialFilePath = pathInput && pathInput.value ? pathInput.value : null;
    currentFilePath = initialFilePath;

    // Initialize Bootstrap modal
    const modalEl = document.getElementById('file-browser-modal');
    if (modalEl) {
        fileBrowserModal = new bootstrap.Modal(modalEl);
    }

    // Check if URL hash specifies a different file to load
    const params = getHashParams();
    if (params.f && params.f !== initialFilePath) {
        // Hash has a file path - load it after editor is ready
        setTimeout(() => {
            switchToFile(params.f);
        }, 100);
    } else {
        // Update hash with current file
        updateHashWithCurrentState();
    }
}

function showFileBrowser() {
    if (!fileBrowserModal) {
        initFileBrowser();
    }
    loadFileList();
    fileBrowserModal.show();
}

function loadFileList() {
    const fileListEl = document.getElementById('file-list');
    fileListEl.innerHTML = '<div class="text-muted text-center py-3"><span class="spinner-border spinner-border-sm"></span> Loading...</div>';

    fetch('/api/files')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                fileListEl.innerHTML = '<div class="text-danger text-center py-3">' + data.error + '</div>';
                return;
            }

            if (!data.files || data.files.length === 0) {
                fileListEl.innerHTML = '<div class="text-muted text-center py-3">No .sd files found</div>';
                return;
            }

            // Render file list
            fileListEl.innerHTML = data.files.map(file => {
                const isActive = file.path === currentFilePath;
                return `
                    <a href="#" class="list-group-item list-group-item-action d-flex justify-content-between align-items-center ${isActive ? 'active' : ''}"
                       onclick="selectFile('${escapeHtml(file.path)}'); return false;">
                        <span><i class="bi bi-file-earmark-code me-2"></i>${escapeHtml(file.path)}</span>
                        ${isActive ? '<i class="bi bi-check2"></i>' : ''}
                    </a>
                `;
            }).join('');
        })
        .catch(err => {
            fileListEl.innerHTML = '<div class="text-danger text-center py-3">Error loading files</div>';
            console.error('Error loading file list:', err);
        });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function selectFile(filepath) {
    // Check for unsaved changes
    if (isDirty) {
        pendingFileSwitchPath = filepath;
        showFileSwitchWarning();
        return;
    }

    switchToFile(filepath);
}

function showFileSwitchWarning() {
    // Hide file browser modal, show unsaved changes modal
    fileBrowserModal.hide();
    const modal = new bootstrap.Modal(document.getElementById('unsaved-changes-modal'));
    modal.show();
}

function handleFileSwitchCancel() {
    pendingFileSwitchPath = null;
    bootstrap.Modal.getInstance(document.getElementById('unsaved-changes-modal')).hide();

    // Reopen file browser
    setTimeout(() => showFileBrowser(), 300);
}

function handleFileSwitchConfirm() {
    const filepath = pendingFileSwitchPath;
    pendingFileSwitchPath = null;
    bootstrap.Modal.getInstance(document.getElementById('unsaved-changes-modal')).hide();

    // Switch to the file
    switchToFile(filepath);
}

function switchToFile(filepath) {
    setStatus('running', 'Loading file...');

    fetch('/api/files/' + encodeURIComponent(filepath))
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                setStatus('error', 'Error: ' + data.error);
                return;
            }

            // Update current file path
            currentFilePath = filepath;
            document.getElementById('current-file-path').value = filepath;

            // Update URL hash so page reload opens same file
            updateHashWithCurrentState();

            // Update editor content
            if (editor._isCodeMirror && editor.setValue) {
                editor.setValue(data.content);
            } else if (editor.value !== undefined) {
                editor.value = data.content;
            }

            // Update file watching state
            lastKnownMtime = data.mtime;
            lastSavedContent = data.content;
            setDirty(false);

            // Update displayed filename in navbar
            updateNavbarFilename(filepath);

            // Re-analyse template
            analyseTemplate();

            setStatus('success', 'Opened: ' + filepath);

            // Hide file browser modal
            if (fileBrowserModal) {
                fileBrowserModal.hide();
            }
        })
        .catch(err => {
            setStatus('error', 'Error loading file: ' + err.message);
            console.error('Error switching file:', err);
        });
}

function updateNavbarFilename(filepath) {
    // Update filename display in navbar
    const filenameEl = document.querySelector('#top-navbar .text-light.opacity-75');
    if (filenameEl) {
        filenameEl.innerHTML = '<i class="bi bi-file-earmark-code"></i> ' + escapeHtml(filepath);
    } else {
        // Create filename element if it doesn't exist
        const brand = document.querySelector('#top-navbar .navbar-brand');
        if (brand) {
            const span = document.createElement('span');
            span.className = 'text-light opacity-75 small ms-2';
            span.innerHTML = '<i class="bi bi-file-earmark-code"></i> ' + escapeHtml(filepath);
            brand.after(span);
        }
    }
}

function createNewFile() {
    const input = document.getElementById('new-file-name');
    let filename = input.value.trim();

    if (!filename) {
        input.classList.add('is-invalid');
        return;
    }
    input.classList.remove('is-invalid');

    // Ensure .sd extension
    if (!filename.endsWith('.sd')) {
        filename += '.sd';
    }

    setStatus('running', 'Creating file...');

    fetchWithCsrf('/api/files/new', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({filename: filename})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            setStatus('error', 'Error: ' + data.error);
            return;
        }

        // Clear input
        input.value = '';

        // Switch to the new file
        selectFile(data.path);
    })
    .catch(err => {
        setStatus('error', 'Error creating file: ' + err.message);
        console.error('Error creating file:', err);
    });
}

// ============================================
// Chat Mode
// ============================================

let chatModeEnabled = false;
let usesHistory = false;
let savedModeBeforeChat = 'interactive'; // Remember mode before entering chat

// Chat session state
let chatSession = {
    turns: [],           // Array of {index, role, content, slots, cost, timestamp}
    totalCost: { tokens: 0, amount: 0 },
    model: null,
    templateSnapshot: null,
    startTime: null
};

// Current turn being streamed
let currentTurnSlots = {};
let currentTurnIndex = -1;
let selectedTurnIndex = -1;
let isChatStreaming = false;

// Toggle between single mode and chat mode
function toggleChatMode(enabled, skipReset = false) {
    chatModeEnabled = enabled;
    const singleView = document.getElementById('single-mode-view');
    const chatView = document.getElementById('chat-mode-view');
    const fileMode = document.getElementById('mode-file');
    const batchMode = document.getElementById('mode-batch');
    const interactiveMode = document.getElementById('mode-single');

    // Persist to localStorage
    localStorage.setItem('struckdown_chat_mode', enabled ? 'true' : 'false');

    if (enabled) {
        // Save current mode and force interactive
        const currentMode = document.querySelector('input[name="mode"]:checked');
        if (currentMode) {
            savedModeBeforeChat = currentMode.value;
        }

        // Disable all mode options in chat mode
        if (interactiveMode) interactiveMode.disabled = true;
        if (fileMode) fileMode.disabled = true;
        if (batchMode) batchMode.disabled = true;

        // Toggle views using classes (to override Bootstrap !important)
        singleView.classList.add('hidden');
        chatView.classList.add('active');

        // Reset chat session when entering chat mode (unless restoring from localStorage)
        if (!skipReset) {
            resetChatSession();
        }
    } else {
        // Re-enable all mode options
        if (interactiveMode) interactiveMode.disabled = false;
        if (fileMode) fileMode.disabled = false;
        if (batchMode) batchMode.disabled = false;

        // Restore previous mode
        const previousModeRadio = document.getElementById('mode-' + savedModeBeforeChat);
        if (previousModeRadio) {
            previousModeRadio.checked = true;
        }

        // Toggle views using classes
        singleView.classList.remove('hidden');
        chatView.classList.remove('active');
    }
}

// Reset chat session to initial state
function resetChatSession() {
    const modelInput = document.getElementById('model-input');
    chatSession = {
        turns: [],
        totalCost: { tokens: 0, amount: 0 },
        model: modelInput ? modelInput.value : '',
        templateSnapshot: getSyntax(),
        startTime: Date.now()
    };
    currentTurnSlots = {};
    currentTurnIndex = -1;
    selectedTurnIndex = -1;
    isChatStreaming = false;

    // Clear chat UI
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.innerHTML = `
        <p class="text-muted text-center mt-3" id="chat-empty-state">
            <i class="bi bi-chat-dots"></i>
            Type a message below to start chatting
        </p>
    `;

    // Clear thinking pane
    const thinkingContainer = document.getElementById('thinking-container');
    thinkingContainer.innerHTML = `
        <p class="text-muted small mb-0" id="thinking-empty-state">
            Slot completions will appear here as they stream in.
            Click a message above to see its slots.
        </p>
    `;
    document.getElementById('thinking-turn-label').textContent = '';

    // Reset cost display
    document.getElementById('chat-cost-display').style.display = 'none';

}

// Parse chat history seed text into messages
function parseSeedText(text) {
    if (!text || !text.trim()) return [];

    const lines = text.split('\n').filter(line => line.trim());
    const messages = [];

    // First line is assistant, alternating from there
    lines.forEach((line, i) => {
        const role = (i % 2 === 0) ? 'assistant' : 'user';
        messages.push({ role, content: line.trim() });
    });

    return messages;
}

// Add seed history to chat session from popover
function addSeedToChat() {
    const textarea = document.getElementById('seed-popover-textarea');
    if (!textarea) return;

    const seedMessages = parseSeedText(textarea.value);
    if (seedMessages.length === 0) {
        return;
    }

    // Remove empty state
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) emptyState.remove();

    // Add seed messages as turns (no slots for these)
    seedMessages.forEach((msg) => {
        const turnIndex = chatSession.turns.length;
        chatSession.turns.push({
            index: turnIndex,
            role: msg.role,
            content: msg.content,
            slots: {},
            cost: null,
            timestamp: Date.now(),
            isSeeded: true
        });
        renderChatMessage(msg.role, msg.content, turnIndex, true);
    });

    // Clear textarea and hide popover
    textarea.value = '';
    const popover = bootstrap.Popover.getInstance(document.getElementById('seed-chat-link'));
    if (popover) popover.hide();
}

// Restart chat - clear all turns and reset
function restartChat() {
    if (isChatStreaming) return;
    resetChatSession();
}

// Initialize seed popover
function initSeedPopover() {
    const seedLink = document.getElementById('seed-chat-link');
    if (!seedLink) return;

    const popoverContent = `
        <div style="width: 500px;">
            <textarea id="seed-popover-textarea" class="form-control mb-2" rows="10" style="font-size: 14px;"
                placeholder="One message per line:&#10;Hello, how can I help?&#10;I have a question...&#10;What about X?&#10;Let me explain..."></textarea>
            <div class="d-flex justify-content-between align-items-center">
                <small class="text-muted">First line = assistant, then alternating. Cmd+Return to add.</small>
                <button type="button" class="btn btn-sm btn-primary" onclick="addSeedToChat()">Add</button>
            </div>
        </div>
    `;

    new bootstrap.Popover(seedLink, {
        html: true,
        content: popoverContent,
        trigger: 'click',
        sanitize: false,
        customClass: 'seed-popover-wide'
    });

    // Add Cmd+Return handler when popover is shown
    seedLink.addEventListener('shown.bs.popover', function() {
        const textarea = document.getElementById('seed-popover-textarea');
        if (textarea) {
            textarea.focus();
            textarea.addEventListener('keydown', function(e) {
                if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault();
                    addSeedToChat();
                }
            });
        }
    });
}

// Build history messages array for API from chat session
function buildHistoryMessages() {
    return chatSession.turns.map(turn => ({
        role: turn.role,
        content: turn.content
    }));
}

// Send a chat message
function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message || isChatStreaming) return;

    // Clear input
    input.value = '';

    // Remove empty state if present
    const emptyState = document.getElementById('chat-empty-state');
    if (emptyState) emptyState.remove();

    // Add user message to session
    const userTurnIndex = chatSession.turns.length;
    chatSession.turns.push({
        index: userTurnIndex,
        role: 'user',
        content: message,
        slots: {},
        cost: null,
        timestamp: Date.now()
    });

    // Render user message
    renderChatMessage('user', message, userTurnIndex);

    // Prepare for assistant response
    const assistantTurnIndex = chatSession.turns.length;
    currentTurnIndex = assistantTurnIndex;
    currentTurnSlots = {};
    currentTurnSlotDetails = {};
    isChatStreaming = true;

    // Add placeholder for assistant turn
    chatSession.turns.push({
        index: assistantTurnIndex,
        role: 'assistant',
        content: '',
        slots: {},
        cost: null,
        timestamp: Date.now()
    });

    // Render streaming assistant message
    renderChatMessage('assistant', '', assistantTurnIndex, false, true);

    // Clear thinking pane and show streaming state
    clearThinkingPane();
    document.getElementById('thinking-turn-label').textContent = 'Current turn';

    // Disable send button
    document.getElementById('chat-send-btn').disabled = true;
    setStatus('running', 'Generating response...');

    // Execute template with chat history
    runChatTurn();
}

// Execute template for current chat turn
function runChatTurn() {
    const syntax = getSyntax();
    const inputs = getInputValues();
    const model = document.getElementById('model-input').value;
    const remoteMode = document.getElementById('remote-mode').value === 'true';

    const historyMessages = buildHistoryMessages();

    const body = {
        syntax: syntax,
        inputs: inputs,
        model: model || null,
        history_messages: historyMessages,
        strict_undefined: getStrictMode(),
        session_id: sessionId  // For evidence loading
    };

    if (remoteMode) {
        const apiKey = document.getElementById('api-key-input').value;
        const apiBase = document.getElementById('api-base-input').value;
        if (apiKey) body.api_key = apiKey;
        if (apiBase) body.api_base = apiBase;
    }

    fetch('/api/run-chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken()
        },
        body: JSON.stringify(body)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Failed to start chat execution');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function processStream() {
            return reader.read().then(({done, value}) => {
                if (done) {
                    finishChatTurn();
                    return;
                }

                buffer += decoder.decode(value, {stream: true});

                // Parse SSE events
                const events = buffer.split('\n\n');
                buffer = events.pop() || '';

                for (const eventBlock of events) {
                    if (!eventBlock.trim()) continue;

                    let eventType = null;
                    let eventData = null;

                    for (const line of eventBlock.split('\n')) {
                        if (line.startsWith('event: ')) {
                            eventType = line.substring(7).trim();
                        } else if (line.startsWith('data: ')) {
                            eventData = line.substring(6);
                        }
                    }

                    if (eventType && eventData) {
                        try {
                            const data = JSON.parse(eventData);
                            handleChatEvent(eventType, data);
                        } catch (e) {
                            console.error('Failed to parse chat event:', eventData, e);
                        }
                    }
                }

                return processStream();
            });
        }

        return processStream();
    }).catch(error => {
        console.error('Chat execution error:', error);
        setStatus('error', 'Error: ' + error.message);
        finishChatTurn(true);
    });
}

// Store full slot details for export (not just output)
let currentTurnSlotDetails = {};

// Handle SSE event from chat execution
function handleChatEvent(eventType, data) {
    switch (eventType) {
        case 'slot_completed':
            // Store slot result (output only for display)
            currentTurnSlots[data.slot_key] = data.result.output;

            // Store full slot details for export
            currentTurnSlotDetails[data.slot_key] = {
                output: data.result.output,
                messages: data.result.messages || [],
                prompt: data.result.prompt || null,
                elapsed_ms: data.elapsed_ms || null,
                was_cached: data.was_cached || false
            };

            // Update thinking pane
            renderThinkingSlot(data.slot_key, data.result.output, false);

            // Update status
            setStatus('running', `Completed: ${data.slot_key}`);
            break;

        case 'checkpoint':
            setStatus('running', `Checkpoint ${data.segment_index + 1}`);
            break;

        case 'complete':
            // Get final response (last slot value)
            const slotKeys = Object.keys(currentTurnSlots);
            const lastSlotKey = slotKeys[slotKeys.length - 1];
            const response = lastSlotKey ? currentTurnSlots[lastSlotKey] : '';

            // Update assistant turn
            if (currentTurnIndex >= 0 && currentTurnIndex < chatSession.turns.length) {
                chatSession.turns[currentTurnIndex].content = String(response);
                chatSession.turns[currentTurnIndex].slots = {...currentTurnSlots};
                chatSession.turns[currentTurnIndex].slotDetails = {...currentTurnSlotDetails};

                // Extract cost and model info from result
                if (data.result) {
                    chatSession.turns[currentTurnIndex].llm = {
                        model: data.result.model || chatSession.model,
                        prompt_tokens: data.result.prompt_tokens || 0,
                        completion_tokens: data.result.completion_tokens || 0,
                        total_cost: data.result.total_cost || 0
                    };
                    const cost = {
                        tokens: (data.result.prompt_tokens || 0) + (data.result.completion_tokens || 0),
                        amount: data.result.total_cost || 0
                    };
                    chatSession.turns[currentTurnIndex].cost = cost;
                    chatSession.totalCost.tokens += cost.tokens;
                    chatSession.totalCost.amount += cost.amount;
                    updateChatCostDisplay();

                    // Update session model if we got it from response
                    if (data.result.model) {
                        chatSession.model = data.result.model;
                    }
                }
            }

            // Update message bubble
            updateChatMessageContent(currentTurnIndex, String(response));

            setStatus('success', 'Complete');
            finishChatTurn();
            break;

        case 'error':
            setStatus('error', 'Error: ' + (data.error_message || 'Unknown error'));

            // Update message with error
            if (currentTurnIndex >= 0) {
                updateChatMessageContent(currentTurnIndex, '[Error: ' + (data.error_message || 'Unknown error') + ']');
            }

            finishChatTurn(true);
            break;
    }
}

// Finish current chat turn
function finishChatTurn(hadError = false) {
    isChatStreaming = false;
    document.getElementById('chat-send-btn').disabled = false;

    // Remove streaming class from message
    const messageEl = document.querySelector(`.chat-message[data-turn="${currentTurnIndex}"]`);
    if (messageEl) {
        messageEl.classList.remove('streaming');
    }

    // Mark slots as completed in thinking pane
    document.querySelectorAll('.thinking-slot.streaming').forEach(el => {
        el.classList.remove('streaming');
    });

    // Auto-scroll chat
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    currentTurnIndex = -1;
}

// Render a chat message bubble
function renderChatMessage(role, content, turnIndex, isSeeded = false, isStreaming = false) {
    const messagesContainer = document.getElementById('chat-messages');

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${role}${isSeeded ? ' seeded' : ''}${isStreaming ? ' streaming' : ''}`;
    messageEl.dataset.turn = turnIndex;
    messageEl.onclick = () => selectChatMessage(turnIndex);

    // Add restart button for assistant messages (not seeded)
    if (role === 'assistant' && !isSeeded) {
        const restartBtn = document.createElement('button');
        restartBtn.className = 'restart-btn';
        restartBtn.title = 'Restart from here';
        restartBtn.innerHTML = '<i class="bi bi-arrow-counterclockwise"></i>';
        restartBtn.onclick = (e) => {
            e.stopPropagation();
            restartFromTurn(turnIndex);
        };
        messageEl.appendChild(restartBtn);
        messageEl.style.position = 'relative';
    }

    const contentSpan = document.createElement('span');
    contentSpan.className = 'message-content';
    contentSpan.textContent = content || (isStreaming ? 'Thinking...' : '');
    messageEl.appendChild(contentSpan);

    messagesContainer.appendChild(messageEl);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update chat message content
function updateChatMessageContent(turnIndex, content) {
    const messageEl = document.querySelector(`.chat-message[data-turn="${turnIndex}"]`);
    if (messageEl) {
        const contentSpan = messageEl.querySelector('.message-content');
        if (contentSpan) {
            contentSpan.textContent = content;
        }
    }
}

// Select a chat message to view its slots
function selectChatMessage(turnIndex) {
    // Don't select if nothing to show
    const turn = chatSession.turns[turnIndex];
    if (!turn || turn.role === 'user') return;

    // Update selection state
    selectedTurnIndex = turnIndex;

    // Update visual selection
    document.querySelectorAll('.chat-message.selected').forEach(el => {
        el.classList.remove('selected');
    });
    const messageEl = document.querySelector(`.chat-message[data-turn="${turnIndex}"]`);
    if (messageEl) {
        messageEl.classList.add('selected');
    }

    // Show slots in thinking pane
    clearThinkingPane();
    document.getElementById('thinking-turn-label').textContent = `Turn ${turnIndex + 1}`;

    const slots = turn.slots || {};
    if (Object.keys(slots).length === 0) {
        document.getElementById('thinking-container').innerHTML = `
            <p class="text-muted small mb-0">No slots recorded for this turn.</p>
        `;
        return;
    }

    for (const [key, value] of Object.entries(slots)) {
        renderThinkingSlot(key, value, false);
    }
}

// Clear thinking pane
function clearThinkingPane() {
    const container = document.getElementById('thinking-container');
    container.innerHTML = '';
}

// Render a slot in the thinking pane
function renderThinkingSlot(slotKey, value, isStreaming = true) {
    // Skip history slot - it duplicates the chat interface
    if (slotKey === 'history') return;

    const container = document.getElementById('thinking-container');

    // Remove empty state
    const emptyState = document.getElementById('thinking-empty-state');
    if (emptyState) emptyState.remove();

    // Check if slot already exists
    let slotEl = container.querySelector(`.thinking-slot[data-slot="${slotKey}"]`);

    if (!slotEl) {
        slotEl = document.createElement('div');
        slotEl.className = `thinking-slot${isStreaming ? ' streaming' : ''}`;
        slotEl.dataset.slot = slotKey;
        slotEl.innerHTML = `
            <div class="thinking-slot-header">
                <code>${escapeHtml(slotKey)}</code>
            </div>
            <div class="thinking-slot-body"></div>
        `;
        container.appendChild(slotEl);
    } else {
        if (!isStreaming) {
            slotEl.classList.remove('streaming');
        }
    }

    // Update value
    const bodyEl = slotEl.querySelector('.thinking-slot-body');
    bodyEl.textContent = formatOutput(value);
}

// Restart conversation from a specific turn
function restartFromTurn(turnIndex) {
    if (isChatStreaming) return;

    // Confirm with user
    if (!confirm('Restart conversation from this message? Later messages will be removed.')) {
        return;
    }

    // Truncate turns (keep up to and including this turn)
    chatSession.turns = chatSession.turns.slice(0, turnIndex + 1);

    // Recalculate total cost
    chatSession.totalCost = { tokens: 0, amount: 0 };
    chatSession.turns.forEach(turn => {
        if (turn.cost) {
            chatSession.totalCost.tokens += turn.cost.tokens;
            chatSession.totalCost.amount += turn.cost.amount;
        }
    });
    updateChatCostDisplay();

    // Remove message bubbles after this turn
    document.querySelectorAll('.chat-message').forEach(el => {
        const idx = parseInt(el.dataset.turn, 10);
        if (idx > turnIndex) {
            el.remove();
        }
    });

    // Clear thinking pane
    clearThinkingPane();
    document.getElementById('thinking-turn-label').textContent = '';

    // Deselect
    selectedTurnIndex = -1;
    document.querySelectorAll('.chat-message.selected').forEach(el => {
        el.classList.remove('selected');
    });

    setStatus('success', 'Restarted from turn ' + (turnIndex + 1));
}

// Update chat cost display
function updateChatCostDisplay() {
    const display = document.getElementById('chat-cost-display');
    if (chatSession.totalCost.tokens > 0) {
        display.style.display = 'block';
        document.getElementById('chat-cost-tokens').textContent = chatSession.totalCost.tokens;
        document.getElementById('chat-cost-amount').textContent = '$' + chatSession.totalCost.amount.toFixed(4);
    }
}

// Export chat history as JSON
function exportChatHistory() {
    // Build user-friendly export format
    const exportData = {
        session: {
            model: chatSession.model,
            template: chatSession.templateSnapshot,
            startTime: chatSession.startTime ? new Date(chatSession.startTime).toISOString() : null,
            exportTime: new Date().toISOString(),
            totalCost: {
                tokens: chatSession.totalCost.tokens,
                amount: chatSession.totalCost.amount,
                formatted: '$' + chatSession.totalCost.amount.toFixed(4)
            }
        },
        turns: chatSession.turns.map((turn, index) => {
            const turnData = {
                turn: index + 1,
                role: turn.role,
                content: turn.content,
                timestamp: turn.timestamp ? new Date(turn.timestamp).toISOString() : null
            };

            // Add LLM details for assistant turns
            if (turn.role === 'assistant') {
                if (turn.llm) {
                    turnData.llm = {
                        model: turn.llm.model,
                        prompt_tokens: turn.llm.prompt_tokens,
                        completion_tokens: turn.llm.completion_tokens,
                        total_tokens: turn.llm.prompt_tokens + turn.llm.completion_tokens,
                        cost: turn.llm.total_cost,
                        cost_formatted: '$' + (turn.llm.total_cost || 0).toFixed(4)
                    };
                }

                // Add slot details with API messages
                if (turn.slotDetails && Object.keys(turn.slotDetails).length > 0) {
                    turnData.slots = {};
                    for (const [slotKey, details] of Object.entries(turn.slotDetails)) {
                        turnData.slots[slotKey] = {
                            output: details.output,
                            elapsed_ms: details.elapsed_ms,
                            was_cached: details.was_cached,
                            api_messages: details.messages || []
                        };
                    }
                }
            }

            return turnData;
        })
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
