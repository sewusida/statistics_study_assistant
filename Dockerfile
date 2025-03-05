# Use the base Hugging Face Space image
FROM python:3.10-slim

# Install system dependencies (LaTeX tools)
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-xetex \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app.py .

# Expose the port (optional, Gradio uses port 7860 by default)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]

# Verify LaTeX tools are installed
RUN latexmk --version