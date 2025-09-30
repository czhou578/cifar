# Use a small Python base
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential

# install node + npm (for building frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# create non-root user (optional but recommended)
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PATH="/home/user/.local/bin:$PATH"

# Copy project files
COPY --chown=user:user . .

# Build frontend (adjust folder & build command to your stack)
WORKDIR /home/user/app/frontend
RUN npm ci --no-audit --prefer-offline
RUN npm run build

# Back to python app
WORKDIR /home/user/app

# Add the current directory to Python path
ENV PYTHONPATH=/home/user/app:$PYTHONPATH

# Install python deps (must have requirements.txt in repo root)
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# create HF cache to avoid permission issues
RUN mkdir -p /home/user/app/.cache && chmod -R 777 /home/user/app/.cache
ENV HF_HOME=/home/user/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/home/user/app/.cache/datasets

# Expose default HF port
EXPOSE 7860

# CMD: run uvicorn pointing to root-level app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]