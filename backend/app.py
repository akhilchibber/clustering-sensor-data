"""
BACKEND: FastAPI Application for Sensor Breakdown Prediction
=============================================================
This API loads 3 trained ML models and predicts the breakdown category
(1, 2, or 3) for given sensor data from a machine with 20 sensors.

Models available:
  - Random Forest (best model, 70% accuracy)
  - Label Spreading (semi-supervised, 77.5% with SMOTE)
  - SVM (55% accuracy)

Endpoints:
  - GET  /health          → Check if API is running
  - GET  /models          → List available models with details
  - POST /predict/manual  → Predict from comma-separated sensor values
  - POST /predict/file    → Predict from uploaded CSV/XLS/XLSX file
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

# ============================================================
# TELEMETRY SETUP (Azure Application Insights)
# ============================================================
# Sends prediction logs and metrics to Azure for monitoring.
# If no connection string is set (local dev), logging goes to console only.

logger = logging.getLogger("sensor-api")
logger.setLevel(logging.INFO)

APPINSIGHTS_CONNECTION = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")

if APPINSIGHTS_CONNECTION:
    from opencensus.ext.azure.log_exporter import AzureLogHandler
    logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION))
    logger.info("Application Insights connected successfully")
else:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(handler)
    logger.info("Running locally — logs go to console only")

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="Sensor Breakdown Prediction API",
    description="Predicts machine breakdown category (1, 2, or 3) from 20 sensor readings",
    version="1.0.0"
)

# Allow frontend to call this API (CORS = Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (for local dev)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods
    allow_headers=["*"],       # Allow all headers
)

# ============================================================
# LOAD MODELS ON STARTUP
# ============================================================
# Models are downloaded from Azure Blob Storage once when the server starts.
# After that, they stay in memory — no Blob calls during predictions.
# If AZURE_STORAGE_CONNECTION_STRING is not set (local dev), loads from local folder.

import tempfile

AZURE_STORAGE_CONN = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "models")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load_model_from_blob(blob_name: str):
    """
    Downloads a .pkl file from Azure Blob Storage and loads it with joblib.
    The file is downloaded to a temp directory (only at startup, not per request).
    """
    from azure.storage.blob import BlobServiceClient
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)
    blob_client = blob_service.get_container_client(AZURE_STORAGE_CONTAINER).get_blob_client(blob_name)

    # Download to a temp file
    temp_path = os.path.join(tempfile.gettempdir(), blob_name)
    with open(temp_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    logger.info(f"Downloaded {blob_name} from Blob Storage")
    return joblib.load(temp_path)


def load_model_local(filename: str):
    """Loads a .pkl file from the local models/ folder (for local development)."""
    return joblib.load(os.path.join(MODELS_DIR, filename))


# Choose loading method: Blob Storage (Azure) or local folder (dev)
if AZURE_STORAGE_CONN:
    logger.info("Loading models from Azure Blob Storage...")
    scaler = load_model_from_blob("scaler.pkl")
    rf_model_obj = load_model_from_blob("random_forest_model.pkl")
    ls_model_obj = load_model_from_blob("label_spreading_model.pkl")
    svm_model_obj = load_model_from_blob("svm_model.pkl")
else:
    logger.info("Loading models from local folder...")
    scaler = load_model_local("scaler.pkl")
    rf_model_obj = load_model_local("random_forest_model.pkl")
    ls_model_obj = load_model_local("label_spreading_model.pkl")
    svm_model_obj = load_model_local("svm_model.pkl")

# Store models in a dictionary for easy access
models = {
    "random_forest": {
        "model": rf_model_obj,
        "name": "Random Forest",
        "accuracy": "70.0%",
        "description": "Supervised ensemble model. Best balanced predictions.",
        "uses_smote": False
    },
    "label_spreading": {
        "model": ls_model_obj,
        "name": "Label Spreading",
        "accuracy": "77.5%",
        "description": "Semi-supervised graph-based model with SMOTE=200.",
        "uses_smote": True
    },
    "svm": {
        "model": svm_model_obj,
        "name": "SVM (Support Vector Machine)",
        "accuracy": "55.0%",
        "description": "Supervised boundary-based classifier.",
        "uses_smote": False
    }
}

print("✓ All models loaded successfully!")


# ============================================================
# DATA MODELS (Request/Response schemas)
# ============================================================
class ManualInput(BaseModel):
    """Schema for manual sensor input via comma-separated values."""
    sensor_values: str  # "0.5, -0.3, 0.8, ..." (20 values)
    selected_models: List[str]  # ["random_forest", "svm"]


class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    model_name: str
    predicted_category: int
    confidence: float
    confidence_level: str  # "High", "Medium", or "Low"
    prob_category_1: float
    prob_category_2: float
    prob_category_3: float


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def classify_confidence(confidence: float) -> str:
    """
    Categorizes a confidence score into High, Medium, or Low.
    - High (≥70%): Model is very sure
    - Medium (40-70%): Model leans toward one category
    - Low (<40%): Model is uncertain
    """
    if confidence >= 0.7:
        return "High"
    elif confidence >= 0.4:
        return "Medium"
    else:
        return "Low"


def predict_with_model(model_key: str, X_scaled: np.ndarray) -> List[dict]:
    """
    Runs prediction using a specific model on scaled sensor data.
    Returns a list of prediction results (one per row of input data).
    """
    model_info = models[model_key]
    model = model_info["model"]

    # Get predictions and probabilities
    if model_key == "label_spreading":
        # Label Spreading uses transduction (already fitted), so we use predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
    else:
        # Random Forest and SVM use standard predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

    results = []
    for i in range(len(predictions)):
        confidence = float(np.max(probabilities[i]))
        results.append({
            "model_name": model_info["name"],
            "model_key": model_key,
            "predicted_category": int(predictions[i]),
            "confidence": round(confidence, 4),
            "confidence_level": classify_confidence(confidence),
            "prob_category_1": round(float(probabilities[i][0]), 4),
            "prob_category_2": round(float(probabilities[i][1]), 4),
            "prob_category_3": round(float(probabilities[i][2]), 4),
        })

    # Log prediction to Application Insights
    logger.info("Prediction completed", extra={
        "custom_dimensions": {
            "model": model_key,
            "rows_predicted": len(predictions),
            "avg_confidence": round(float(np.mean(np.max(probabilities, axis=1))), 4),
            "category_distribution": {
                "cat_1": int(np.sum(predictions == 1)),
                "cat_2": int(np.sum(predictions == 2)),
                "cat_3": int(np.sum(predictions == 3))
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    })

    return results


def parse_sensor_values(sensor_string: str) -> np.ndarray:
    """
    Converts a comma-separated string of sensor values into a numpy array.
    Validates that exactly 20 values are provided.
    Handles trailing commas and extra whitespace gracefully.
    """
    try:
        # Split by comma, remove empty strings (handles trailing commas)
        parts = [v.strip() for v in sensor_string.split(",") if v.strip()]
        values = [float(v) for v in parts]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid input. All values must be numbers separated by commas.")

    if len(values) != 20:
        raise HTTPException(status_code=400, detail=f"Expected 20 sensor values, got {len(values)}.")

    return np.array(values).reshape(1, -1)


def read_uploaded_file(file_content: bytes, filename: str) -> tuple:
    """
    Reads an uploaded file (CSV, XLS, or XLSX) and returns a DataFrame.
    Handles two cases:
      - File with only 20 sensor columns → predict all rows
      - File with 21+ columns (includes Label) → predict only unlabeled rows
    Returns: (full_df, sensor_df, unlabeled_mask)
      - full_df: the complete original data
      - sensor_df: only the 20 sensor columns
      - unlabeled_mask: boolean array (True = needs prediction, False = already labeled)
    """
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_content))
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, XLS, or XLSX.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Validate: must have at least 20 columns
    if df.shape[1] < 20:
        raise HTTPException(status_code=400, detail=f"File must have at least 20 sensor columns. Found {df.shape[1]}.")

    # Extract only the first 20 columns as sensor data
    sensor_df = df.iloc[:, :20].copy()

    # Check if there's a 21st column (Label column)
    if df.shape[1] >= 21:
        label_col = df.iloc[:, 20]
        # Unlabeled rows = NaN or empty in the label column
        unlabeled_mask = label_col.isna() | (label_col.astype(str).str.strip() == '')
    else:
        # No label column → predict all rows
        unlabeled_mask = pd.Series([True] * len(df))

    return df, sensor_df, unlabeled_mask


def create_output_excel(df_input: pd.DataFrame, all_results: dict, full_df: pd.DataFrame = None, unlabeled_mask: pd.Series = None) -> bytes:
    """
    Creates an Excel file with input data + predictions from all selected models.
    If a label column exists, it's preserved. Predictions are only for unlabeled rows.
    Returns the Excel file as bytes for download.
    """
    # Use full original data if available, otherwise just sensor data
    if full_df is not None:
        output_df = full_df.copy()
    else:
        output_df = df_input.copy()
        output_df.columns = [f"Sensor {i}" for i in range(20)]

    # Add prediction columns for each model
    for model_key, results in all_results.items():
        model_name = models[model_key]["name"]

        if unlabeled_mask is not None and not unlabeled_mask.all():
            # Only fill predictions for unlabeled rows
            pred_col = [None] * len(output_df)
            conf_col = [None] * len(output_df)
            level_col = [None] * len(output_df)
            p1_col = [None] * len(output_df)
            p2_col = [None] * len(output_df)
            p3_col = [None] * len(output_df)

            result_idx = 0
            for i in range(len(output_df)):
                if unlabeled_mask.iloc[i]:
                    pred_col[i] = results[result_idx]["predicted_category"]
                    conf_col[i] = results[result_idx]["confidence"]
                    level_col[i] = results[result_idx]["confidence_level"]
                    p1_col[i] = results[result_idx]["prob_category_1"]
                    p2_col[i] = results[result_idx]["prob_category_2"]
                    p3_col[i] = results[result_idx]["prob_category_3"]
                    result_idx += 1

            output_df[f"{model_name} - Prediction"] = pred_col
            output_df[f"{model_name} - Confidence"] = conf_col
            output_df[f"{model_name} - Confidence Level"] = level_col
            output_df[f"{model_name} - P(Cat 1)"] = p1_col
            output_df[f"{model_name} - P(Cat 2)"] = p2_col
            output_df[f"{model_name} - P(Cat 3)"] = p3_col
        else:
            # Predict all rows
            output_df[f"{model_name} - Prediction"] = [r["predicted_category"] for r in results]
            output_df[f"{model_name} - Confidence"] = [r["confidence"] for r in results]
            output_df[f"{model_name} - Confidence Level"] = [r["confidence_level"] for r in results]
            output_df[f"{model_name} - P(Cat 1)"] = [r["prob_category_1"] for r in results]
            output_df[f"{model_name} - P(Cat 2)"] = [r["prob_category_2"] for r in results]
            output_df[f"{model_name} - P(Cat 3)"] = [r["prob_category_3"] for r in results]

    # Write to Excel in memory
    buffer = io.BytesIO()
    output_df.to_excel(buffer, index=False, sheet_name="Predictions")
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns OK if the API is running and models are loaded.
    Use this to verify the server is alive.
    """
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "message": "API is running and all models are ready for predictions."
    }


@app.get("/models")
def list_models():
    """
    Lists all available models with their details.
    Useful for the frontend to show model options to the user.
    """
    model_list = []
    for key, info in models.items():
        model_list.append({
            "key": key,
            "name": info["name"],
            "accuracy": info["accuracy"],
            "description": info["description"],
            "uses_smote": info["uses_smote"]
        })
    return {"models": model_list}


@app.post("/predict/manual")
def predict_manual(input_data: ManualInput):
    """
    Predicts breakdown category from manually entered sensor values.
    Input: 20 comma-separated sensor values + list of selected models.
    Output: Predictions with confidence scores for each selected model.
    """
    # Validate selected models
    for model_key in input_data.selected_models:
        if model_key not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}. Available: {list(models.keys())}")

    # Parse and scale the sensor values
    X_raw = parse_sensor_values(input_data.sensor_values)
    X_scaled = scaler.transform(X_raw)

    # Run predictions for each selected model
    all_predictions = {}
    for model_key in input_data.selected_models:
        all_predictions[model_key] = predict_with_model(model_key, X_scaled)

    return {"predictions": all_predictions, "input_rows": 1}


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    selected_models: str = Query(..., description="Comma-separated model keys: random_forest,svm,label_spreading")
):
    """
    Predicts breakdown categories from an uploaded file (CSV/XLS/XLSX).
    If the file has a Label column (21st column), only unlabeled rows are predicted.
    Rows with existing labels are skipped.
    """
    # Parse selected models from query string
    model_keys = [m.strip() for m in selected_models.split(",")]
    for model_key in model_keys:
        if model_key not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}. Available: {list(models.keys())}")

    # Read the uploaded file
    file_content = await file.read()
    full_df, sensor_df, unlabeled_mask = read_uploaded_file(file_content, file.filename)

    # Scale only the unlabeled rows for prediction
    X_unlabeled = sensor_df[unlabeled_mask].values
    X_scaled = scaler.transform(X_unlabeled)

    # Run predictions for each selected model (only on unlabeled rows)
    all_predictions = {}
    for model_key in model_keys:
        all_predictions[model_key] = predict_with_model(model_key, X_scaled)

    total_rows = len(sensor_df)
    predicted_rows = int(unlabeled_mask.sum())
    skipped_rows = total_rows - predicted_rows

    return {
        "predictions": all_predictions,
        "input_rows": total_rows,
        "predicted_rows": predicted_rows,
        "skipped_rows": skipped_rows,
        "filename": file.filename
    }


@app.post("/predict/file/download")
async def predict_file_download(
    file: UploadFile = File(...),
    selected_models: str = Query(..., description="Comma-separated model keys: random_forest,svm,label_spreading")
):
    """
    Same as /predict/file but returns an Excel file for download.
    The Excel contains: original data + predictions + confidence for each model.
    Rows with existing labels are preserved but not predicted.
    """
    # Parse selected models
    model_keys = [m.strip() for m in selected_models.split(",")]
    for model_key in model_keys:
        if model_key not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}. Available: {list(models.keys())}")

    # Read and process the file
    file_content = await file.read()
    full_df, sensor_df, unlabeled_mask = read_uploaded_file(file_content, file.filename)

    # Scale only unlabeled rows
    X_unlabeled = sensor_df[unlabeled_mask].values
    X_scaled = scaler.transform(X_unlabeled)

    # Run predictions (only on unlabeled rows)
    all_results = {}
    for model_key in model_keys:
        all_results[model_key] = predict_with_model(model_key, X_scaled)

    # Create Excel output (preserves full data, fills predictions only for unlabeled)
    excel_bytes = create_output_excel(sensor_df, all_results, full_df, unlabeled_mask)

    # Return as downloadable file
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions_output.xlsx"}
    )


class ManualDownloadInput(BaseModel):
    """Schema for manual input download request."""
    sensor_values: str
    selected_models: List[str]


@app.post("/predict/manual/download")
def predict_manual_download(input_data: ManualDownloadInput):
    """
    Predicts from manual sensor values and returns an Excel file for download.
    This is a dedicated endpoint for downloading results from manual input.
    """
    # Validate selected models
    for model_key in input_data.selected_models:
        if model_key not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}. Available: {list(models.keys())}")

    # Parse and scale the sensor values
    X_raw = parse_sensor_values(input_data.sensor_values)
    X_scaled = scaler.transform(X_raw)

    # Run predictions for each selected model
    all_results = {}
    for model_key in input_data.selected_models:
        all_results[model_key] = predict_with_model(model_key, X_scaled)

    # Create a DataFrame from the input for the Excel output
    df_input = pd.DataFrame(X_raw, columns=[f"Sensor {i}" for i in range(20)])

    # Create Excel output
    excel_bytes = create_output_excel(df_input, all_results)

    # Return as downloadable file
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=predictions_output.xlsx"}
    )


# ============================================================
# RUN THE SERVER (for local development)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting Sensor Prediction API on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
