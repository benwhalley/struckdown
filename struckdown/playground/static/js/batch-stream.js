// Struckdown Playground Batch Streaming JavaScript

let batchEventSource = null;
let batchData = {
    columns: {input: [], output: []},
    rows: [],
    completed: 0,
    total: 0
};

// Initialize batch streaming
function initBatchStream(taskId) {
    // Setup output container for batch mode
    const container = document.getElementById('outputs-container');
    container.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <div>
                <span class="badge bg-primary">
                    <span id="batch-completed">0</span> / <span id="batch-total">0</span> complete
                </span>
                <span id="batch-status-badge" class="badge bg-info ms-2">Running</span>
            </div>
            <button class="btn btn-success btn-sm" id="download-btn" onclick="downloadResults()" disabled>
                <i class="bi bi-download"></i> Download xlsx
            </button>
        </div>
        <div class="mb-2" id="column-toggles-container" style="display: none;">
            <span class="small text-muted">Show columns:</span>
            <div id="column-toggles" class="d-inline-flex flex-wrap gap-1 ms-2"></div>
        </div>
        <div class="table-responsive">
            <table id="batch-table" class="table table-sm table-striped table-hover">
                <thead><tr id="batch-header"><th>#</th></tr></thead>
                <tbody id="batch-body"></tbody>
            </table>
        </div>
    `;

    // Reset batch data
    batchData = {
        columns: {input: [], output: []},
        rows: [],
        completed: 0,
        total: 0
    };

    // Connect to SSE stream
    batchEventSource = new EventSource('/api/batch-stream/' + taskId);

    batchEventSource.addEventListener('row', function(event) {
        const rowData = JSON.parse(event.data);
        handleBatchRow(rowData);
    });

    batchEventSource.addEventListener('progress', function(event) {
        const progress = JSON.parse(event.data);
        updateBatchProgress(progress);
    });

    batchEventSource.addEventListener('done', function(event) {
        handleBatchComplete();
    });

    batchEventSource.addEventListener('error', function(event) {
        if (event.data) {
            const error = JSON.parse(event.data);
            handleBatchError(error);
        }
    });

    batchEventSource.onerror = function() {
        if (batchEventSource.readyState === EventSource.CLOSED) {
            // Connection closed normally
        } else {
            setStatus('error', 'Connection lost');
        }
    };
}

// Handle incoming row data
function handleBatchRow(rowData) {
    const index = rowData.index;

    // First row - setup columns
    if (batchData.rows.length === 0) {
        setupBatchColumns(rowData);
    }

    // Store row data
    batchData.rows[index] = rowData;
    batchData.completed++;

    // Update table row
    updateTableRow(rowData);

    // Update progress display
    document.getElementById('batch-completed').textContent = batchData.completed;
}

// Setup batch columns from first row
function setupBatchColumns(rowData) {
    const inputCols = Object.keys(rowData.inputs || {});
    const outputCols = Object.keys(rowData.outputs || {});

    batchData.columns.input = inputCols;
    batchData.columns.output = outputCols;

    // Get total from progress if available
    if (batchData.total > 0) {
        document.getElementById('batch-total').textContent = batchData.total;
    }

    // Build header
    const header = document.getElementById('batch-header');
    header.innerHTML = '<th>#</th>';

    inputCols.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        th.className = 'batch-col-input';
        th.dataset.column = col;
        header.appendChild(th);
    });

    outputCols.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        th.className = 'batch-col-output';
        th.dataset.column = col;
        header.appendChild(th);
    });

    // Build column toggles
    const togglesContainer = document.getElementById('column-toggles-container');
    const toggles = document.getElementById('column-toggles');
    togglesContainer.style.display = 'block';
    toggles.innerHTML = '';

    inputCols.forEach(col => {
        toggles.appendChild(createColumnToggle(col, 'input', true));
    });

    outputCols.forEach(col => {
        toggles.appendChild(createColumnToggle(col, 'output', true));
    });

    // Pre-populate empty rows
    const body = document.getElementById('batch-body');
    body.innerHTML = '';
    for (let i = 0; i < batchData.total; i++) {
        const tr = document.createElement('tr');
        tr.id = 'batch-row-' + i;
        tr.innerHTML = `<td>${i + 1}</td>` +
            inputCols.map(() => '<td class="batch-cell-input batch-cell-pending">-</td>').join('') +
            outputCols.map(() => '<td class="batch-cell-output batch-cell-pending">-</td>').join('');
        body.appendChild(tr);
    }
}

// Create column toggle checkbox
function createColumnToggle(colName, type, checked) {
    const div = document.createElement('div');
    div.className = 'form-check form-check-inline';
    div.innerHTML = `
        <input class="form-check-input" type="checkbox" id="col-toggle-${colName}"
               data-column="${colName}" ${checked ? 'checked' : ''}>
        <label class="form-check-label col-${type}" for="col-toggle-${colName}">${colName}</label>
    `;

    div.querySelector('input').addEventListener('change', function() {
        toggleColumn(colName, this.checked);
    });

    return div;
}

// Toggle column visibility
function toggleColumn(colName, visible) {
    const table = document.getElementById('batch-table');
    const headerCell = table.querySelector(`th[data-column="${colName}"]`);
    const index = Array.from(headerCell.parentNode.children).indexOf(headerCell);

    headerCell.style.display = visible ? '' : 'none';

    table.querySelectorAll('tbody tr').forEach(row => {
        const cell = row.children[index];
        if (cell) cell.style.display = visible ? '' : 'none';
    });
}

// Update table row with data
function updateTableRow(rowData) {
    const row = document.getElementById('batch-row-' + rowData.index);
    if (!row) return;

    const inputCols = batchData.columns.input;
    const outputCols = batchData.columns.output;

    let cellIndex = 1; // Skip # column

    // Update input cells
    inputCols.forEach(col => {
        const cell = row.children[cellIndex++];
        const value = rowData.inputs[col];
        cell.textContent = formatCellValue(value);
        cell.className = 'batch-cell-input';
        cell.title = String(value);
    });

    // Update output cells
    outputCols.forEach(col => {
        const cell = row.children[cellIndex++];

        if (rowData.status === 'error') {
            cell.textContent = 'Error';
            cell.className = 'batch-cell-output batch-cell-error';
            cell.title = rowData.error || 'Unknown error';
        } else {
            const value = rowData.outputs[col];
            cell.textContent = formatCellValue(value);
            cell.className = 'batch-cell-output';
            cell.title = String(value);
        }
    });
}

// Format cell value for display
function formatCellValue(value) {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

// Update batch progress
function updateBatchProgress(progress) {
    batchData.total = progress.total;
    document.getElementById('batch-total').textContent = progress.total;
    document.getElementById('batch-completed').textContent = progress.completed;

    setStatus('running', `Processing ${progress.completed}/${progress.total}...`);
}

// Handle batch completion
function handleBatchComplete() {
    if (batchEventSource) {
        batchEventSource.close();
        batchEventSource = null;
    }

    document.getElementById('batch-status-badge').className = 'badge bg-success ms-2';
    document.getElementById('batch-status-badge').textContent = 'Complete';
    document.getElementById('download-btn').disabled = false;

    setStatus('success', 'Batch complete');
    disableSaveButton(false);
}

// Handle batch error
function handleBatchError(error) {
    if (batchEventSource) {
        batchEventSource.close();
        batchEventSource = null;
    }

    document.getElementById('batch-status-badge').className = 'badge bg-danger ms-2';
    document.getElementById('batch-status-badge').textContent = 'Error';

    setStatus('error', 'Batch error: ' + (error.error || 'Unknown error'));
    disableSaveButton(false);
}

// Download results as xlsx
function downloadResults() {
    const taskId = document.getElementById('current-task-id').value;
    if (!taskId) return;

    window.location.href = '/api/download/' + taskId;
}
