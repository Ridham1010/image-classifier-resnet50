from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # NEW LINE: Import CORSMiddleware
from PIL import Image
import io
import torch
import timm
import torch.nn as nn
from torchvision import transforms
import os

app = FastAPI()

# --- CORS Configuration ---
# Define the origins that are allowed to make requests to your FastAPI backend.
# "http://localhost:3000" is where your React frontend runs.
# "http://localhost" is good to include for general local development.
origins = [
    "http://localhost:3000",
    "http://localhost:8000", 
    "https://image-classifier-frontend-h2p2.onrender.com", # Your React app's URL
]

# Add CORSMiddleware to your FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies to be included in cross-origin HTTP requests
    allow_methods=["*"],            # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],            # Allow all HTTP headers
)
# --- END CORS Configuration ---

# --- Configuration ---
MODEL_PATH = "image_classification_model.pth"
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# --- Device Setup ---
# Check for Apple MPS (M1/M2/M3/M4) first, then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"FastAPI app using device: {device}")

# --- Model Loading ---
model = None # Initialize model as None

@app.on_event("startup")
async def load_model():
    """
    Loads the trained model when the FastAPI application starts up.
    """
    global model
    model_full_path = os.path.join(os.path.dirname(__file__), MODEL_PATH)

    if not os.path.exists(model_full_path):
        raise RuntimeError(f"Model file not found at {model_full_path}. Please ensure training was successful.")

    try:
        # Create a new ResNet18 model instance
        model = timm.create_model('resnet18', pretrained=False) # No need for pretrained weights here
        num_classes = len(CLASS_NAMES)

        # Modify the final classification layer (matching training)
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, 'classifier'):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)
        elif hasattr(model, 'head'):
            if isinstance(model.head, nn.Sequential) and isinstance(model.head[-1], nn.Linear):
                num_ftrs = model.head[-1].in_features
                model.head[-1] = nn.Linear(num_ftrs, num_classes)
            elif isinstance(model.head, nn.Linear):
                num_ftrs = model.head.in_features
                model.head = nn.Linear(num_ftrs, num_classes)
        else:
            raise AttributeError(f"Couldn't find a common classification head for model. Please inspect its structure.")


        # Load the saved state_dict (weights)
        # Use map_location to ensure it loads correctly regardless of original training device
        state_dict = torch.load(model_full_path, map_location=device)
        model.load_state_dict(state_dict)

        model.eval() # Set model to evaluation mode
        model.to(device) # Move model to the selected device
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

# --- Image Preprocessing for Prediction ---
# This must match the validation transforms used during training!
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image, performs inference, and returns the predicted class.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image data
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

        # Move input to the same device as the model
        input_batch = input_batch.to(device)

        # Perform inference
        with torch.no_grad(): # Disable gradient calculations for inference
            output = model(input_batch)

        # Get probabilities and predicted class
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]

        return {
            "filename": file.filename,
            "prediction": predicted_class_name,
            "confidence": probabilities[predicted_class_idx].item()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the CIFAR-10 Image Classifier API!"}