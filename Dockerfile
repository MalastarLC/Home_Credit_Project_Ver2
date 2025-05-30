# --- Stage 1: Base Image ---
# Use an official Python runtime as a parent image.
# 'python:3.11-slim' is a good choice: '3.11' is your Python version,
# and '-slim' means it's a smaller version of the image, good for production.
FROM python:3.11-slim

# --- Stage 2: Set Working Directory ---
# Sets the working directory for any subsequent RUN, CMD, ENTRYPOINT, COPY, ADD instructions.
# If the directory doesn't exist, it will be created.
# It's common practice to use /app or /usr/src/app.
WORKDIR /app

# --- Stage 3: Copy Requirements and Install Dependencies ---
# Copy the requirements file first. This is a Docker best practice for caching.
# If requirements.txt doesn't change, Docker can reuse this layer from a previous build, speeding things up.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt.
# --no-cache-dir: Reduces the image size by not storing the pip download cache.
# --upgrade pip: Ensures you have the latest pip.
# -r requirements.txt: Tells pip to install from the requirements file.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# --- Stage 4: Copy Application Files and Artifacts ---
# Copy your application code and necessary artifacts into the container's /app directory.

# Copy your main API script
COPY API_Script.py .

# Copy your preprocessing pipeline script
COPY preprocessing_pipeline.py .

# Copy the MLflow runs directory.
# CRITICAL: Ensure 'mlruns' is in the same directory as your Dockerfile (your project root)
# or adjust the source path accordingly. This directory contains your trained model.
COPY mlruns ./mlruns

# Copy the file containing the expected input column names for your model.
COPY pipeline_input_columns.txt .

# If you have other .py files or resources directly imported/used by your scripts,
# copy them here as well. For example:
# COPY utils.py .
# COPY static ./static
# COPY templates ./templates

# --- Stage 5: Set Environment Variables (Optional but Recommended) ---
# Set environment variables that your application might need.
# This tells MLflow where to find the tracking data (your runs) INSIDE the container.
ENV MLFLOW_TRACKING_URI="file:./mlruns"

# Flask specific environment variables (often not strictly needed if Gunicorn is configured right)
# ENV FLASK_APP=API_Script.py
# ENV FLASK_ENV=production # Good practice for production

# --- Stage 6: Expose Port ---
# Informs Docker that the container listens on the specified network port at runtime.
# This does NOT actually publish the port. It's more like documentation between the
# person who builds the image and the person who runs the container.
# Your Gunicorn command will bind to this port.
EXPOSE 8000

# --- Stage 7: Define the Command to Run the Application ---
# Specifies the command to run when a container is started from this image.
# We use Gunicorn, a production-grade WSGI server, to run your Flask app.
# "API_Script:app" means: In the file API_Script.py, find the Flask instance named 'app'.
# --workers 1: Number of worker processes. For the free tier on Render, 1 is fine.
#              You might increase this for paid tiers with more CPU.
# --bind 0.0.0.0:8000: Tells Gunicorn to listen on all available network interfaces
#                      inside the container on port 8000. Render will map an external
#                      port to this internal port 8000.
# --timeout 120: (Optional) Increase request timeout if your predictions can be long. Default is 30s.
#                Adjust based on your call_api_script.py timeout (which is 300s).
# --log-level info: (Optional) Gunicorn logging level.
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:8000", "--timeout", "120", "--log-level", "info", "API_Script:app"]