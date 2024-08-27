# Use a base image with Python
FROM python:3.10.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt /app/

# Update the package list and install system dependencies, then clean up
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    libpng-dev \
    libjpeg-dev \
    cmake \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker's cache
RUN pip install --no-cache-dir Cython cmake lit

# Install any required packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --use-deprecated=legacy-resolver

# Copy the rest of the application code into the container
COPY . .

# Run app.py when the container launches
CMD ["python", "test.py"]
