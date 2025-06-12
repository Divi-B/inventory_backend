import joblib
import os

MODEL_DIR = "models"

def load_model(prediction_type: str):
    model_filename = prediction_type.lower().replace(" ", "_") + "_model.joblib"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for: {prediction_type}")
    
    return joblib.load(model_path)
