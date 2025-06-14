# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for image processing (Pillow etc.)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire 'app' directory into the container's working directory
COPY app/ ./app/

# Copy the trained model file
# We assume image_classification_model.pth is directly in the backend folder
# because your main.py expects it in the same directory as itself.

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the application
# --host 0.0.0.0 is crucial for allowing external connections to the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]