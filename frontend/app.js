/**
 * FRONTEND: Sensor Breakdown Prediction App
 * ==========================================
 * Handles user interactions, API calls, and result display.
 * Connects to the FastAPI backend running on localhost:8000.
 */

// Backend API URL — reads from a config or defaults to localhost for development
// In production, this is set to the Azure App Service URL (no secrets, just a public URL)
const API_URL = window.API_CONFIG?.url || "http://localhost:8000";

// Store the last uploaded file for download functionality
let lastUploadedFile = null;

// ============================================================
// TAB SWITCHING (Manual Entry vs File Upload)
// ============================================================
function switchTab(tab) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

    // Show the selected tab
    document.getElementById(`tab-${tab}`).classList.add('active');
    event.target.classList.add('active');
}

// ============================================================
// FILE HANDLING
// ============================================================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        lastUploadedFile = file;
        document.getElementById('file-name').textContent = `✓ Selected: ${file.name}`;
    }
}

// ============================================================
// GET SELECTED MODELS
// ============================================================
function getSelectedModels() {
    const selected = [];
    if (document.getElementById('model_rf').checked) selected.push('random_forest');
    if (document.getElementById('model_ls').checked) selected.push('label_spreading');
    if (document.getElementById('model_svm').checked) selected.push('svm');
    return selected;
}

// ============================================================
// SHOW/HIDE LOADING
// ============================================================
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// ============================================================
// PREDICT FROM MANUAL INPUT
// ============================================================
async function predictManual() {
    const sensorValues = document.getElementById('sensor-input').value.trim();
    const selectedModels = getSelectedModels();

    // Validation
    if (!sensorValues) {
        alert('Please enter 20 sensor values separated by commas.');
        return;
    }
    if (selectedModels.length === 0) {
        alert('Please select at least one model.');
        return;
    }

    showLoading();

    try {
        const response = await fetch(`${API_URL}/predict/manual`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sensor_values: sensorValues,
                selected_models: selectedModels
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayManualResults(data.predictions);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// ============================================================
// PREDICT FROM FILE UPLOAD
// ============================================================
async function predictFile() {
    const selectedModels = getSelectedModels();

    if (!lastUploadedFile) {
        alert('Please upload a file first.');
        return;
    }
    if (selectedModels.length === 0) {
        alert('Please select at least one model.');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('file', lastUploadedFile);

        const modelsParam = selectedModels.join(',');
        const response = await fetch(`${API_URL}/predict/file?selected_models=${modelsParam}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const data = await response.json();
        displayFileResults(data.predictions, data.input_rows);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// ============================================================
// DOWNLOAD EXCEL FILE
// ============================================================
async function downloadExcel() {
    const selectedModels = getSelectedModels();

    if (selectedModels.length === 0) {
        alert('Please select at least one model.');
        return;
    }

    try {
        let response;

        if (lastUploadedFile) {
            // CASE 1: Download from uploaded file
            const formData = new FormData();
            formData.append('file', lastUploadedFile);
            const modelsParam = selectedModels.join(',');

            response = await fetch(`${API_URL}/predict/file/download?selected_models=${modelsParam}`, {
                method: 'POST',
                body: formData
            });
        } else {
            // CASE 2: Download from manual input
            const sensorValues = document.getElementById('sensor-input').value.trim();
            if (!sensorValues) {
                alert('Please enter sensor values or upload a file first.');
                return;
            }

            response = await fetch(`${API_URL}/predict/manual/download`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sensor_values: sensorValues,
                    selected_models: selectedModels
                })
            });
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Download failed');
        }

        // Trigger file download in the browser
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'predictions_output.xlsx';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

    } catch (error) {
        alert(`Download error: ${error.message}`);
    }
}

// ============================================================
// DISPLAY RESULTS: MANUAL INPUT (single row)
// ============================================================
function displayManualResults(predictions) {
    const container = document.getElementById('results-container');
    container.innerHTML = '';

    for (const [modelKey, results] of Object.entries(predictions)) {
        const result = results[0]; // Single row for manual input
        const html = createSingleResultCard(result);
        container.innerHTML += html;
    }

    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('download-section').classList.remove('hidden');

    // Smooth scroll to results
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// DISPLAY RESULTS: FILE INPUT (multiple rows)
// ============================================================
function displayFileResults(predictions, totalRows) {
    const container = document.getElementById('results-container');
    container.innerHTML = `<p style="color: #94a3b8; margin-bottom: 1rem;">Predictions for <strong>${totalRows} rows</strong> — showing confidence summary per model:</p>`;

    for (const [modelKey, results] of Object.entries(predictions)) {
        const modelName = results[0].model_name;

        // Calculate confidence summary
        const highCount = results.filter(r => r.confidence_level === 'High').length;
        const medCount = results.filter(r => r.confidence_level === 'Medium').length;
        const lowCount = results.filter(r => r.confidence_level === 'Low').length;
        const avgConf = (results.reduce((sum, r) => sum + r.confidence, 0) / results.length * 100).toFixed(1);

        // Category distribution
        const cat1 = results.filter(r => r.predicted_category === 1).length;
        const cat2 = results.filter(r => r.predicted_category === 2).length;
        const cat3 = results.filter(r => r.predicted_category === 3).length;

        const html = `
            <div class="result-model">
                <div class="result-model-header">
                    <span class="result-model-name">${modelName}</span>
                    <span style="color: #94a3b8;">Avg Confidence: <strong style="color: #f1f5f9;">${avgConf}%</strong></span>
                </div>

                <div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 150px;">
                        <p style="color: #64748b; font-size: 0.8rem; margin-bottom: 0.3rem;">Category Distribution</p>
                        <span class="category-badge category-1">Cat 1: ${cat1}</span>
                        <span class="category-badge category-2">Cat 2: ${cat2}</span>
                        <span class="category-badge category-3">Cat 3: ${cat3}</span>
                    </div>
                </div>

                <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                    <div class="confidence-summary-item">
                        <span class="confidence-level level-high">High (≥70%): ${highCount} (${(highCount/totalRows*100).toFixed(1)}%)</span>
                    </div>
                    <div class="confidence-summary-item">
                        <span class="confidence-level level-medium">Medium (40-70%): ${medCount} (${(medCount/totalRows*100).toFixed(1)}%)</span>
                    </div>
                    <div class="confidence-summary-item">
                        <span class="confidence-level level-low">Low (<40%): ${lowCount} (${(lowCount/totalRows*100).toFixed(1)}%)</span>
                    </div>
                </div>
            </div>
        `;
        container.innerHTML += html;
    }

    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('download-section').classList.remove('hidden');
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// CREATE A SINGLE RESULT CARD (for manual input)
// ============================================================
function createSingleResultCard(result) {
    const levelClass = result.confidence_level === 'High' ? 'level-high' :
                       result.confidence_level === 'Medium' ? 'level-medium' : 'level-low';

    return `
        <div class="result-model">
            <div class="result-model-header">
                <span class="result-model-name">${result.model_name}</span>
                <span class="confidence-level ${levelClass}">${result.confidence_level} Confidence</span>
            </div>

            <div class="result-prediction">
                <span class="category-badge category-${result.predicted_category}">
                    Category ${result.predicted_category}
                </span>
                <span style="color: #94a3b8;">Confidence: <strong style="color: #f1f5f9;">${(result.confidence * 100).toFixed(1)}%</strong></span>
            </div>

            <div class="confidence-section">
                <div class="confidence-bar-container">
                    <span class="confidence-label">Cat 1</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill fill-cat-1" style="width: ${result.prob_category_1 * 100}%">
                            ${(result.prob_category_1 * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div class="confidence-bar-container">
                    <span class="confidence-label">Cat 2</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill fill-cat-2" style="width: ${result.prob_category_2 * 100}%">
                            ${(result.prob_category_2 * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                <div class="confidence-bar-container">
                    <span class="confidence-label">Cat 3</span>
                    <div class="confidence-bar">
                        <div class="confidence-fill fill-cat-3" style="width: ${result.prob_category_3 * 100}%">
                            ${(result.prob_category_3 * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}
