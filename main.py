from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import io
import json
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.encoders import jsonable_encoder

# Load environment variables
load_dotenv()
api_key = "AIzaSyDThv38MTqhP_wMFTiMTpTmoT-XZpQBBQ0"

# Configure Gemini if API key is available
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI(title="Market Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature mappings
features = {
    "Target Market": ["Order Item Cardprod Id", "Order Item Product Price", "Order Region", "Product Price"],
    "Product Price Prediction": ["Category Name", "Department Name", "Sales", "Order Region", "Product Image"],
    "Sales Per Customer": ["Days for shipping (real)", "Department Id", "Order City", "Order Item Id", "Order Item Product Price",
                           "Order Item Quantity", "Sales", "Order Item Total", "Order Region", "Product Card Id", "Product Category Id"],
    "Shipping Days": ["Days for shipment (scheduled)", "Late_delivery_risk", "Order Item Cardprod Id"]
}

model_files = {
    "Target Market": "models/Market_model.joblib",
    "Product Price Prediction": "models/product_price_model.joblib",
    "Sales Per Customer": "models/sales_per_customer_model.joblib",
    "Shipping Days": "models/shipping_model.joblib"
}

prediction_columns = {
    "Target Market": "Target Market Prediction",
    "Product Price Prediction": "Predicted Product Price",
    "Sales Per Customer": "Predicted Sales Per Customer",
    "Shipping Days": "Predicted Shipping Days"
}

# Pydantic models
class PredictionInput(BaseModel):
    prediction_factor: str
    feature_values: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Any
    prediction_column: str

class FeatureImportanceResponse(BaseModel):
    features: List[str]
    importance_values: List[float]

class DataSummaryResponse(BaseModel):
    summary: str

class CorrelationResponse(BaseModel):
    correlation_matrix: Dict[str, Dict[str, float]]

class ChartDataRequest(BaseModel):
    chart_type: str  # 'bar' or 'pie'
    features: List[str]
    data: List[Dict[str, Any]]
    aggregate_function: Optional[str] = "mean"  # mean, sum, count, etc.

class ChartDataResponse(BaseModel):
    labels: List[str]
    datasets: List[Dict[str, Any]]

class DataPayload(BaseModel):
    data: List[Dict]

class FullPredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    predicted_column: str

# Load joblib models
def load_model(prediction_factor: str):
    model_filename = model_files.get(prediction_factor)
    if not model_filename:
        raise ValueError(f"No model defined for prediction factor: {prediction_factor}")
    model_path = os.path.abspath(model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    return joblib.load(model_path)

def generate_textual_summary(input_df: pd.DataFrame) -> str:
    try:
        basic_stats = input_df.describe(include='all').fillna("").astype(str).to_string()
        column_info = "\n".join([f"- {col}: {input_df[col].dtype}" for col in input_df.columns])

        prompt = f"""You are an expert data analyst. Analyze the dataset below and provide a detailed summary report.

        ## First 10 Rows
        {input_df.head().to_string(index=False)}

        ## Column Info
        {column_info}

        ## Basic Statistics
        {basic_stats}

        Please highlight:
        - Key trends and insights
        - Correlations or outliers
        """

        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"


@app.get("/")
def read_root():
    return {"message": "Welcome to Market Analysis API"}

@app.get("/prediction-factors")
def get_prediction_factors():
    return {"prediction_factors": list(features.keys())}

@app.get("/features/{prediction_factor}")
def get_features(prediction_factor: str):
    if prediction_factor not in features:
        raise HTTPException(status_code=400, detail=f"Invalid prediction factor: {prediction_factor}")
    return {"features": features[prediction_factor]}

@app.post("/predict", response_model=FullPredictionResponse)
async def predict_full_dataset(file: UploadFile = File(...), prediction_factor: str = Form(...)):
    if prediction_factor not in features:
        raise HTTPException(status_code=400, detail=f"Invalid prediction factor: {prediction_factor}")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        required_features = features[prediction_factor]

        # Ensure all required features are in the uploaded data
        missing_cols = [col for col in required_features if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing_cols)}")

        model = load_model(prediction_factor)

        try:
            X = df[required_features]
            predictions = model.predict(X)
            predicted_column_name = prediction_columns.get(prediction_factor, "Predicted Value")
            df[predicted_column_name] = predictions.tolist() 
            return jsonable_encoder({
                "predictions": df.to_dict(orient="records"),
                "predicted_column": predicted_column_name
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
@app.post("/predict-input", response_model=PredictionResponse)
def predict_from_input(payload: PredictionInput):
    prediction_factor = payload.prediction_factor
    feature_values = payload.feature_values

    if prediction_factor not in features:
        raise HTTPException(status_code=400, detail=f"Invalid prediction factor: {prediction_factor}")

    try:
        model = load_model(prediction_factor)
        required_features = features[prediction_factor]

        # Ensure all required features are present
        missing = [f for f in required_features if f not in feature_values]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {', '.join(missing)}")

        # Create single-row DataFrame for prediction
        input_df = pd.DataFrame([feature_values])[required_features]
        prediction = model.predict(input_df)[0]
        prediction_column = prediction_columns.get(prediction_factor, "Predicted Value")

        prediction = model.predict(input_df)[0]
        if isinstance(prediction, (np.integer, np.floating)):
            prediction = prediction.item()  # Convert to Python type

            prediction_column = prediction_columns.get(prediction_factor, "Predicted Value")
            return {"prediction": prediction, "prediction_column": prediction_column}


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/feature-importance/{prediction_factor}", response_model=FeatureImportanceResponse)
def get_feature_importance(prediction_factor: str):
    if prediction_factor not in features:
        raise HTTPException(status_code=400, detail=f"Invalid prediction factor: {prediction_factor}")
    
    model = load_model(prediction_factor)
    
    try:
        importances = model.feature_importances_
        return {
            "features": features[prediction_factor],
            "importance_values": importances.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance not available for this model: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...), prediction_factor: str = Form(...)):
    if prediction_factor not in features:
        raise HTTPException(status_code=400, detail=f"Invalid prediction factor: {prediction_factor}")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        return {
            "filename": file.filename,
            "prediction_factor": prediction_factor,
            "summary": f"Uploaded {file.filename} with {len(df)} rows and {len(df.columns)} columns.",
            "columns": df.columns.tolist(),
            "data": df.head().to_dict(orient="records"),
            "full_data": df.to_dict(orient="records")
        }
    except Exception as e:
        return {"detail": f"Error processing file: {e}"}


@app.post("/dataset-summary")
async def get_dataset_summary(payload: DataPayload):
    try:
        df = pd.DataFrame(payload.data)
        summary = generate_textual_summary(df)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
    

@app.post("/correlation", response_model=CorrelationResponse)
async def calculate_correlation(data: Dict[str, List[Any]]):
    try:
        df = pd.DataFrame(data)
        corr_matrix = df.corr(numeric_only=True).round(3).to_dict()
        return {"correlation_matrix": corr_matrix}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute correlation: {str(e)}")

@app.post("/download-dataset")
async def download_uploaded_dataset(full_data: str = Form(...)):
    """
    Downloads the previously uploaded dataset as a CSV file.
    """
    import json
    try:
        data_list = json.loads(full_data)
        if not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            raise HTTPException(status_code=400, detail="Invalid data format.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")

    if not data_list:
        raise HTTPException(status_code=400, detail="No data to download.")

    df = pd.DataFrame(data_list)
    csv_stream = io.StringIO()
    df.to_csv(csv_stream, index=False)
    csv_content = csv_stream.getvalue()

    headers = {
        "Content-Disposition": "attachment; filename=prediction_report.csv",
        "Content-Type": "text/csv",
    }

    return StreamingResponse(io.BytesIO(csv_content.encode('utf-8')), headers=headers, media_type="text/csv")

@app.post("/chart-data", response_model=ChartDataResponse)
async def generate_chart_data(request: ChartDataRequest):
    """
    Generate data for bar or pie charts based on selected features
    """
    try:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(request.data)

        # Ensure all requested features exist in the dataframe
        missing_features = [f for f in request.features if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features not found in data: {', '.join(missing_features)}"
            )

        # First feature will be used for labels/categories
        category_feature = request.features[0]

        # Remaining features are values to be aggregated
        value_features = request.features[1:] if len(request.features) > 1 else [category_feature]

        # Prepare chart data
        if request.chart_type in ['bar', 'pie']:
            # Group by the category feature
            if request.aggregate_function == "count":
                # Special case for count
                grouped_data = df.groupby(category_feature).size().reset_index(name='count')
                value_features = ['count']
            else:
                # For other aggregation functions
                agg_func = getattr(np, request.aggregate_function, np.mean)
                grouped_data = df.groupby(category_feature)[value_features].agg(agg_func).reset_index()

            # Get unique labels and convert to string
            labels = grouped_data[category_feature].astype(str).tolist()

            # Prepare datasets
            datasets = []
            for feature in value_features:
                data = grouped_data[feature].tolist()

                # Generate a consistent color for each dataset
                # Simple hash-based color generation
                color_base = hash(feature) % 360

                datasets.append({
                    "label": feature,
                    "data": data,
                    "backgroundColor": [f"hsl({(color_base + i * 37) % 360}, 70%, 60%)" for i in range(len(labels))],
                    "borderColor": [f"hsl({(color_base + i * 37) % 360}, 70%, 50%)" for i in range(len(labels))],
                    "borderWidth": 1
                })

            return {
                "labels": labels,
                "datasets": datasets
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported chart type: {request.chart_type}. Supported types are 'bar' and 'pie'."
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart data: {str(e)}")
    
    
@app.post("/generate_summary/{factor}")
async def generate_summary_endpoint(factor: str, file: UploadFile = File(...)):
    if factor not in features:
        raise HTTPException(status_code=400, detail="Invalid prediction factor")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        summary = generate_textual_summary(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
