from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the trained model
model_path = 'your_trained_model.keras'
model = load_model(model_path)

# Initialize FastAPI app
app = FastAPI()

# Function to make predictions
def predict(image: Image.Image):
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)  # Make prediction
    return prediction.tolist()  # Convert numpy array to list for JSON response

# Route for predicting on an uploaded image
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read image from the uploaded file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Make prediction
    prediction = predict(image)
    
    # Return the prediction
    return {"prediction": prediction}