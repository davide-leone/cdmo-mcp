# Start from an official Python 3.10 image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    unzip \
    git \
    wget \
    libffi-dev \
    libgmp-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MiniZinc
RUN curl -LO https://github.com/MiniZinc/MiniZincIDE/releases/download/2.7.6/MiniZincIDE-2.7.6-bundle-linux-x86_64.tgz && \
    tar -xvzf MiniZincIDE-2.7.6-bundle-linux-x86_64.tgz && \
    mv MiniZincIDE-2.7.6-bundle-linux-x86_64 /opt/minizinc && \
    ln -s /opt/minizinc/bin/minizinc /usr/local/bin/minizinc

ENV PATH="/opt/minizinc/bin:$PATH"

# Install Z3
RUN pip install z3-solver

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY . .

# Set permissions if needed
# RUN chmod +x launcher.py

# Set up Gurobi (user must mount license at runtime)
# Expect GUROBI_HOME to be /opt/gurobi (user can override)
ENV GUROBI_HOME=/opt/gurobi
ENV PATH="${GUROBI_HOME}/bin:${PATH:-}"
# ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib:${LD_LIBRARY_PATH:-}"

# USER NOTE:
# You must bind your license file at runtime using:
#   -v /path/to/gurobi.lic:/root/gurobi.lic

# Define default command
CMD ["python"]
