# -------------------------------
# Base Image
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# Set Working Directory
# -------------------------------
WORKDIR /app

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Install UV
# -------------------------------
RUN pip install --no-cache-dir uv

# -------------------------------
# Copy dependency files first (for caching)
# -------------------------------
COPY pyproject.toml uv.lock ./

# -------------------------------
# Install dependencies using uv
# -------------------------------
RUN uv sync --no-dev

# -------------------------------
# Copy project code
# -------------------------------
COPY src/ ./src/
COPY ui/ ./ui/
COPY evaluation/ ./evaluation/
COPY results/ ./results/

# -------------------------------
# Streamlit config
# -------------------------------
RUN mkdir -p /root/.streamlit

RUN echo "\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > /root/.streamlit/config.toml

# -------------------------------
# Expose port
# -------------------------------
EXPOSE 8501

# -------------------------------
# Healthcheck
# -------------------------------
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# -------------------------------
# Run Streamlit App (via uv)
# -------------------------------
CMD ["uv", "run", "streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]