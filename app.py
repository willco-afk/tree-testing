import os
from fastapi import FastAPI
from google.cloud import storage
from keras.models import load_model
import tempfile
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Function to load the model from Google Cloud Storage
def load_model_from_gcs(model_path):
    client = storage.Client()
    bucket = client.get_bucket('tree-decorator-model')  # Your bucket name
    blob = bucket.blob(model_path)  # Path to your model in the bucket

    # Save the model file locally in a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)  # Download model to temporary file
        model = load_model(temp_file.name)  # Load model from the temporary file
    
    return model

# Load the model from Google Cloud Storage (provide the path to your model in the bucket)
model = load_model_from_gcs('models/your_trained_model.keras')  # Path in GCS

# Pydantic model for the incoming prediction request (adjust as needed)
class ImageData(BaseModel):
    image: str  # Base64-encoded image or URL of the image (you can adjust this)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tree Decorator API!"}

@app.post("/predict/")
async def predict(data: ImageData):
    # Example: Decode the image, preprocess it, and use the model for prediction
    # Decode and preprocess the image data as required (e.g., using Pillow, OpenCV, etc.)
    
    # For simplicity, we'll assume 'data.image' is already preprocessed or passed in an acceptable format

    # Example prediction (replace with actual image processing and prediction logic)
    # prediction = model.predict(processed_image) 
    
    # Dummy response for demonstration
    prediction = {"prediction": "decorated" if np.random.random() > 0.5 else "not decorated"}
    
    return prediction

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))