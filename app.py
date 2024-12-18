from google.cloud import storage
from keras.models import load_model
import tempfile

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
model = load_model_from_gcs('models/your_trained_model.keras')  # Update with your model path in GCS

# Now you can use the 'model' object for predictions or further processing