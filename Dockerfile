FROM python:3.12

# Set work directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose HF default port
EXPOSE 7860

# Run Flask
CMD ["python", "app.py"]
