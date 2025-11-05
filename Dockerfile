# ====== Dockerfile ======
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy your training script and model into the image
COPY train.py /app/
COPY model.joblib /app/

# Default command when the container runs
CMD ["python", "-c", "print('Model artifact present:', __import__('os').listdir('/app'))"]
