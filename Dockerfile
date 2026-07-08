# Uses a lightened Python image, matching the environement I coded this in
FROM python:3.11-slim 

# Creates/sets the working directory the app is in
WORKDIR /app

# Copies the requirements needed
COPY requirements.txt .

# OS level graphics required for RDKit's drawing
RUN apt-get update && apt-get install -y --no-install-recommends \
          libxrender1 libxext6 libsm6 libexpat1 \
      && rm -rf /var/lib/apt/lists/*


# Installs the requirements, so Docker caches the install layer
RUN pip install --no-cache-dir -r requirements.txt

# Copies the rest of the project into the image
COPY . .

# The app listens on port 8000
EXPOSE 8000

# The web server launch command run on start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]