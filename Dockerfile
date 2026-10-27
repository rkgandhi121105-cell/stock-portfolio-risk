# Use official lightweight Python 3.11 runtime
FROM python:3.11-slim

# Environment settings: prevent bytecode generation and ensure unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Set application working directory
WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for security best practices
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application source modules and configuration
COPY .streamlit/ .streamlit/
COPY portfolio_risk/ portfolio_risk/
COPY app.py .

# Switch to non-root user
USER appuser

# Expose default Streamlit network port
EXPOSE 8501

# Healthcheck endpoint provided by Streamlit core
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
