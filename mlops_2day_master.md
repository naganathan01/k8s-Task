# 🚀 2-Day MLOps Mastery Plan

## 📋 **Prerequisites Setup (2 hours)**
Before starting, ensure you have:
- Docker Desktop installed
- Python 3.8+ with pip
- Git configured
- VS Code or preferred IDE
- Free accounts: GitHub, DockerHub, AWS/GCP (free tier)

---

## 🎯 **Day 1: Foundation & Core MLOps (12 hours)**

### **Morning Block 1: ML Lifecycle & Experiment Tracking (3 hours)**

#### **Theory (45 min)**
- **ML Lifecycle Phases**: Data → Preprocessing → Training → Validation → Deployment → Monitoring
- **Key Challenges**: Model drift, version control, reproducibility, scalability
- **MLOps vs DevOps**: Data dependencies, model performance decay, experimentation needs

#### **Hands-On Project 1: MLflow Setup & Experiment Tracking (2h 15min)**
```bash
# 1. Install MLflow
pip install mlflow scikit-learn pandas numpy

# 2. Create project structure
mkdir mlops-project && cd mlops-project
mkdir data models notebooks experiments
```

**Create `train_model.py`:**
```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_model(n_estimators=100, max_depth=3):
    with mlflow.start_run():
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return accuracy

if __name__ == "__main__":
    # Run experiments with different parameters
    for n_est in [50, 100, 200]:
        for depth in [3, 5, 7]:
            acc = train_model(n_est, depth)
            print(f"n_estimators={n_est}, max_depth={depth}, accuracy={acc:.3f}")
```

**Run experiments:**
```bash
python train_model.py
mlflow ui --port 5000
```

**✅ Master:** MLflow tracking, parameter logging, model registry basics

---

### **Morning Block 2: Containerization with Docker (3 hours)**

#### **Theory (30 min)**
- **Why Docker for ML**: Environment consistency, dependency isolation, deployment portability
- **Docker Concepts**: Images, containers, layers, Dockerfile best practices
- **ML-specific considerations**: Large model files, GPU support, multi-stage builds

#### **Hands-On Project 2: Containerizing ML Models (2h 30min)**

**Create `Dockerfile`:**
```dockerfile
# Multi-stage build for optimization
FROM python:3.9-slim as builder

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "serve_model.py"]
```

**Create `requirements.txt`:**
```
mlflow==2.8.0
scikit-learn==1.3.0
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.0.3
numpy==1.24.3
```

**Create `serve_model.py`:**
```python
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import numpy as np
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Iris Model API", version="1.0.0")

# Load model (in production, use model registry)
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Load latest model from MLflow
        model = mlflow.sklearn.load_model("models:/iris_model/latest")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fallback to local model
        model = mlflow.sklearn.load_model("./model")

class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: list

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Prepare input data
        input_data = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].tolist()
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Build and run:**
```bash
# Build Docker image
docker build -t iris-model:v1 .

# Run container
docker run -p 8000:8000 iris-model:v1

# Test API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**✅ Master:** Docker for ML, API serving, health checks, logging

---

### **Afternoon Block 1: CI/CD Pipeline (3 hours)**

#### **Theory (30 min)**
- **CI/CD for ML**: Data validation, model testing, automated deployment
- **Pipeline stages**: Test → Build → Deploy → Monitor
- **ML-specific tests**: Data drift, model performance, API functionality

#### **Hands-On Project 3: GitHub Actions Pipeline (2h 30min)**

**Create `.github/workflows/mlops-pipeline.yml`:**
```yaml
name: MLOps Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run data validation tests
      run: |
        pytest tests/test_data.py -v
    
    - name: Run model tests
      run: |
        pytest tests/test_model.py -v
    
    - name: Run API tests
      run: |
        pytest tests/test_api.py -v
    
    - name: Generate coverage report
      run: |
        pytest --cov=src --cov-report=xml
    
  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t iris-model:${{ github.sha }} .
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Push to Docker Hub
      run: |
        docker tag iris-model:${{ github.sha }} yourusername/iris-model:${{ github.sha }}
        docker push yourusername/iris-model:${{ github.sha }}
        docker tag iris-model:${{ github.sha }} yourusername/iris-model:latest
        docker push yourusername/iris-model:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment..."
        # Add deployment commands here
    
    - name: Run integration tests
      run: |
        echo "Running integration tests..."
        # Add integration test commands
    
    - name: Deploy to production
      run: |
        echo "Deploying to production..."
        # Add production deployment commands
```

**Create test files:**

**`tests/test_data.py`:**
```python
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def test_data_quality():
    """Test data quality and integrity"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Test data shape
    assert X.shape[0] == 150
    assert X.shape[1] == 4
    assert len(y) == 150
    
    # Test for missing values
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()
    
    # Test value ranges
    assert X.min() >= 0
    assert y.min() >= 0
    assert y.max() <= 2

def test_data_distribution():
    """Test data distribution properties"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Test class balance
    unique, counts = np.unique(y, return_counts=True)
    assert len(unique) == 3
    assert all(count == 50 for count in counts)
```

**`tests/test_model.py`:**
```python
import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_model_training():
    """Test model training process"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model is fitted
    assert hasattr(model, 'classes_')
    assert hasattr(model, 'feature_importances_')

def test_model_performance():
    """Test model performance meets minimum threshold"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert minimum accuracy threshold
    assert accuracy >= 0.8, f"Model accuracy {accuracy:.3f} below threshold"

def test_model_prediction_shape():
    """Test model prediction output shape"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Test single prediction
    single_pred = model.predict(X[:1])
    assert single_pred.shape == (1,)
    
    # Test batch prediction
    batch_pred = model.predict(X[:10])
    assert batch_pred.shape == (10,)
```

**`tests/test_api.py`:**
```python
import pytest
from fastapi.testclient import TestClient
from serve_model import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    """Test prediction endpoint"""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "prediction" in result
    assert "probability" in result
    assert isinstance(result["prediction"], int)
    assert isinstance(result["probability"], list)
    assert len(result["probability"]) == 3

def test_predict_input_validation():
    """Test input validation"""
    invalid_data = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error
```

**✅ Master:** GitHub Actions, automated testing, Docker integration, CI/CD best practices

---

### **Afternoon Block 2: Cloud & Kubernetes Basics (3 hours)**

#### **Theory (45 min)**
- **Kubernetes concepts**: Pods, Deployments, Services, ConfigMaps
- **Why Kubernetes for ML**: Scalability, resource management, rolling updates
- **Cloud services**: AWS SageMaker, GCP Vertex AI, Azure ML

#### **Hands-On Project 4: Kubernetes Deployment (2h 15min)**

**Create `k8s/deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-model-deployment
  labels:
    app: iris-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-model
  template:
    metadata:
      labels:
        app: iris-model
    spec:
      containers:
      - name: iris-model
        image: yourusername/iris-model:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/model"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: iris-model-service
spec:
  selector:
    app: iris-model
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

**Create `k8s/configmap.yaml`:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: iris-model-config
data:
  model_name: "iris_classifier"
  model_version: "v1.0.0"
  log_level: "INFO"
  max_requests_per_minute: "1000"
```

**Deploy to Kubernetes:**
```bash
# Start minikube (local Kubernetes)
minikube start

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# Test the service
kubectl port-forward service/iris-model-service 8080:80

# Test API
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**✅ Master:** Kubernetes basics, service deployment, health checks, scaling

---

## 🎯 **Day 2: Advanced MLOps & Production (12 hours)**

### **Morning Block 1: Infrastructure as Code (3 hours)**

#### **Theory (30 min)**
- **IaC Benefits**: Reproducibility, version control, automation
- **Terraform vs CloudFormation**: Multi-cloud vs AWS-specific
- **Best practices**: Modules, state management, secrets

#### **Hands-On Project 5: Terraform AWS Setup (2h 30min)**

**Create `terraform/main.tf`:**
```hcl
provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "mlops_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "mlops-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "mlops_igw" {
  vpc_id = aws_vpc.mlops_vpc.id

  tags = {
    Name = "mlops-igw"
  }
}

# Public Subnet
resource "aws_subnet" "mlops_public_subnet" {
  vpc_id                  = aws_vpc.mlops_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "mlops-public-subnet"
  }
}

# Route Table
resource "aws_route_table" "mlops_rt" {
  vpc_id = aws_vpc.mlops_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mlops_igw.id
  }

  tags = {
    Name = "mlops-rt"
  }
}

# Route Table Association
resource "aws_route_table_association" "mlops_rta" {
  subnet_id      = aws_subnet.mlops_public_subnet.id
  route_table_id = aws_route_table.mlops_rt.id
}

# Security Group
resource "aws_security_group" "mlops_sg" {
  name        = "mlops-sg"
  description = "Security group for MLOps infrastructure"
  vpc_id      = aws_vpc.mlops_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "mlops-sg"
  }
}

# EC2 Instance for MLOps
resource "aws_instance" "mlops_instance" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]
  subnet_id              = aws_subnet.mlops_public_subnet.id

  user_data = file("user_data.sh")

  tags = {
    Name = "mlops-instance"
  }
}

# S3 Bucket for MLOps artifacts
resource "aws_s3_bucket" "mlops_artifacts" {
  bucket = "${var.project_name}-mlops-artifacts-${random_string.bucket_suffix.result}"

  tags = {
    Name = "MLOps Artifacts"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# ECR Repository
resource "aws_ecr_repository" "mlops_repo" {
  name = "${var.project_name}-models"

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "MLOps Models Repository"
  }
}
```

**Create `terraform/variables.tf`:**
```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "project_name" {
  description = "Name of the MLOps project"
  type        = string
  default     = "iris-mlops"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "ami_id" {
  description = "AMI ID for EC2 instance"
  type        = string
  default     = "ami-0c02fb55956c7d316"  # Amazon Linux 2
}

variable "key_name" {
  description = "EC2 Key Pair name"
  type        = string
  default     = "mlops-key"
}
```

**Create `terraform/outputs.tf`:**
```hcl
output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.mlops_instance.public_ip
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.mlops_artifacts.bucket
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.mlops_repo.repository_url
}
```

**Create `terraform/user_data.sh`:**
```bash
#!/bin/bash
yum update -y
yum install -y docker

# Start Docker service
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install Python 3.9
amazon-linux-extras install python3.8 -y
pip3 install --upgrade pip

# Install MLflow
pip3 install mlflow boto3 psycopg2-binary

echo "MLOps infrastructure setup complete!"
```

**Deploy infrastructure:**
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

**✅ Master:** Terraform basics, AWS infrastructure, resource management

---

### **Morning Block 2: Model Monitoring & Observability (3 hours)**

#### **Theory (30 min)**
- **Model Monitoring**: Performance drift, data drift, concept drift
- **Key Metrics**: Accuracy, latency, throughput, resource usage
- **Alerting**: Threshold-based, anomaly detection, business impact

#### **Hands-On Project 6: Monitoring Setup (2h 30min)**

**Create `monitoring/prometheus.yml`:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'iris-model'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Create `monitoring/docker-compose.yml`:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

  iris-model:
    image: yourusername/iris-model:latest
    container_name: iris-model
    ports:
      - "8000:8000"
    depends_on:
      - prometheus

volumes:
  grafana-storage:
```

**Update `serve_model.py` with metrics:**
```python
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import mlflow.sklearn
import time
import numpy as np
import logging

# Metrics
REQUEST_COUNT = Counter('iris_model_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('iris_model_request_duration_seconds', 'Request latency')
MODEL_ACCURACY = Gauge('iris_model_accuracy', 'Model accuracy')
ACTIVE_PREDICTIONS = Gauge('iris_model_active_predictions', 'Active predictions')

app = FastAPI(title="Iris Model API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.observe(process_time)
    
    return response

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model("./model")
        logger.info("Model loaded successfully")
        MODEL_ACCURACY.set(0.95)  # Set initial accuracy
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.post("/predict")
async def predict(request: PredictionRequest):
    ACTIVE_PREDICTIONS.inc()
    try:
        input_data = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0].tolist()
        
        logger.info(f"Prediction made: {prediction}")
        
        return {
            "prediction": int(prediction),
            "probability": probability,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    finally:
        ACTIVE_PREDICTIONS.dec()

@app.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }
```

**Create Grafana dashboard config:**
```bash
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
```

**Create `monitoring/grafana/datasources/datasource.yml`:**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

**Start monitoring stack:**
```bash
cd monitoring
docker-compose up -d

# Access Grafana at http://localhost:3000 (admin/admin)
# Access Prometheus at http://localhost:9090
```

**✅ Master:** Prometheus, Grafana, metrics collection, alerting

---

### **Afternoon Block 1: Advanced MLOps Patterns (3 hours)**

#### **Theory (30 min)**
- **Model Versioning**: A/B testing, blue-green deployments, canary releases
- **Feature Stores**: Centralized feature management, online/offline serving
- **Pipeline Orchestration**: Airflow, Kubeflow, Prefect

#### **Hands-On Project 7: Complete MLOps Pipeline (2h 30min)**

**Create `pipeline/airflow_dag.py`:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import boto3
import joblib

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'iris_model_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for Iris model',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['mlops', 'iris', 'production'],
)

def extract_data(**context):
    """Extract data from source"""
    from sklearn.datasets import load_iris
    import pandas as pd
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save to S3 or local storage
    df.to_csv('/tmp/iris_data.csv', index=False)
    
    # Log data quality metrics
    print(f"Data extracted: {len(df)} rows, {len(df.columns)} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return '/tmp/iris_data.csv'

def validate_data(**context):
    """Validate data quality"""
    import pandas as pd
    
    # Load data
    df = pd.read_csv('/tmp/iris_data.csv')
    
    # Data quality checks
    assert len(df) > 0, "Empty dataset"
    assert df.isnull().sum().sum() == 0, "Missing values found"
    assert len(df.columns) == 5, "Incorrect number of columns"
    assert df['target'].nunique() == 3, "Incorrect number of target classes"
    
    print("Data validation passed")
    return True

def train_model(**context):
    """Train the model"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import mlflow
    import mlflow.sklearn
    
    # Load data
    df = pd.read_csv('/tmp/iris_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, '/tmp/iris_model.pkl')
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        
        # Model quality gate
        if accuracy < 0.85:
            raise ValueError(f"Model accuracy {accuracy:.4f} below threshold 0.85")
        
        return accuracy

def validate_model(**context):
    """Validate model performance"""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    
    # Load model and test data
    model = joblib.load('/tmp/iris_model.pkl')
    df = pd.read_csv('/tmp/iris_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Test model
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # Validation checks
    assert accuracy > 0.8, f"Model accuracy {accuracy:.4f} below threshold"
    assert hasattr(model, 'predict_proba'), "Model doesn't support probability predictions"
    
    print(f"Model validation passed with accuracy: {accuracy:.4f}")
    return True

def deploy_model(**context):
    """Deploy model to production"""
    import shutil
    import os
    
    # Copy model to deployment directory
    os.makedirs('/tmp/deployment', exist_ok=True)
    shutil.copy('/tmp/iris_model.pkl', '/tmp/deployment/model.pkl')
    
    # Create model metadata
    metadata = {
        'model_name': 'iris_classifier',
        'version': context['ds'],
        'accuracy': context['ti'].xcom_pull(task_ids='train_model'),
        'deployment_date': datetime.now().isoformat()
    }
    
    import json
    with open('/tmp/deployment/metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print("Model deployed successfully")
    return True

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Build Docker image
build_image_task = BashOperator(
    task_id='build_docker_image',
    bash_command='cd /tmp/deployment && docker build -t iris-model:{{ ds }} .',
    dag=dag,
)

# Run integration tests
integration_test_task = BashOperator(
    task_id='integration_tests',
    bash_command='python -m pytest tests/integration/ -v',
    dag=dag,
)

# Define dependencies
extract_task >> validate_data_task >> train_task >> validate_model_task >> deploy_task >> build_image_task >> integration_test_task
```

**Create `pipeline/kubeflow_pipeline.py`:**
```python
import kfp
from kfp import dsl
from kfp.components import create_component_from_func

@create_component_from_func
def data_extraction_op() -> str:
    """Extract data component"""
    from sklearn.datasets import load_iris
    import pandas as pd
    import pickle
    
    # Load and save data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Save to persistent volume
    df.to_csv('/tmp/iris_data.csv', index=False)
    
    return '/tmp/iris_data.csv'

@create_component_from_func
def model_training_op(data_path: str) -> float:
    """Train model component"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Save model
    joblib.dump(model, '/tmp/iris_model.pkl')
    
    return accuracy

@create_component_from_func
def model_validation_op(accuracy: float) -> bool:
    """Validate model component"""
    threshold = 0.85
    
    if accuracy < threshold:
        raise ValueError(f"Model accuracy {accuracy:.4f} below threshold {threshold}")
    
    return True

@create_component_from_func
def model_deployment_op(validation_result: bool) -> str:
    """Deploy model component"""
    import shutil
    import os
    from datetime import datetime
    
    if not validation_result:
        raise ValueError("Model validation failed")
    
    # Deploy model
    os.makedirs('/tmp/deployment', exist_ok=True)
    shutil.copy('/tmp/iris_model.pkl', '/tmp/deployment/model.pkl')
    
    deployment_info = {
        'status': 'deployed',
        'timestamp': datetime.now().isoformat(),
        'model_path': '/tmp/deployment/model.pkl'
    }
    
    return str(deployment_info)

@dsl.pipeline(
    name='iris-model-pipeline',
    description='Complete MLOps pipeline for Iris classification'
)
def iris_model_pipeline():
    """Define the pipeline"""
    
    # Data extraction
    data_extraction_task = data_extraction_op()
    
    # Model training
    model_training_task = model_training_op(data_extraction_task.output)
    
    # Model validation
    model_validation_task = model_validation_op(model_training_task.output)
    
    # Model deployment
    model_deployment_task = model_deployment_op(model_validation_task.output)
    
    # Configure resource requirements
    model_training_task.set_memory_limit('2Gi')
    model_training_task.set_cpu_limit('1000m')

if __name__ == '__main__':
    # Compile pipeline
    kfp.compiler.Compiler().compile(iris_model_pipeline, 'iris_model_pipeline.yaml')
```

**✅ Master:** Pipeline orchestration, automated workflows, quality gates

---

### **Afternoon Block 2: Production Best Practices (3 hours)**

#### **Theory (30 min)**
- **Security**: Secrets management, image scanning, access control
- **Scalability**: Auto-scaling, load balancing, resource optimization
- **Disaster Recovery**: Backup strategies, rollback procedures

#### **Hands-On Project 8: Production-Ready Setup (2h 30min)**

**Create `production/docker-compose.prod.yml`:**
```yaml
version: '3.8'

services:
  iris-model:
    image: yourusername/iris-model:latest
    container_name: iris-model-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/model
      - LOG_LEVEL=INFO
      - WORKERS=4
      - MAX_REQUESTS=1000
    secrets:
      - mlflow_token
      - aws_credentials
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - iris-model

  redis:
    image: redis:alpine
    container_name: redis-cache
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-prod
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-prod
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password
    secrets:
      - grafana_password
    volumes:
      - grafana_data:/var/lib/grafana

secrets:
  mlflow_token:
    file: ./secrets/mlflow_token.txt
  aws_credentials:
    file: ./secrets/aws_credentials.txt
  grafana_password:
    file: ./secrets/grafana_password.txt

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

**Create `production/nginx.conf`:**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream iris_model {
        server iris-model:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # Logging
        access_log /var/log/nginx/access.log;
        error_log /var/log/nginx/error.log;

        # Health check endpoint
        location /health {
            proxy_pass http://iris_model/health;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # API endpoints with rate limiting
        location /predict {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://iris_model/predict;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }

        # Metrics endpoint (restrict access)
        location /metrics {
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
            
            proxy_pass http://iris_model/metrics;
        }
    }
}
```

**Create `production/helm/iris-model/Chart.yaml`:**
```yaml
apiVersion: v2
name: iris-model
description: Iris ML Model Helm Chart
type: application
version: 1.0.0
appVersion: "1.0.0"
```

**Create `production/helm/iris-model/values.yaml`:**
```yaml
# Default values for iris-model
replicaCount: 3

image:
  repository: yourusername/iris-model
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: iris-model.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: iris-model-tls
      hosts:
        - iris-model.yourdomain.com

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

env:
  - name: MODEL_PATH
    value: "/app/model"
  - name: LOG_LEVEL
    value: "INFO"

secrets:
  - name: mlflow-token
    key: MLFLOW_TOKEN
  - name: aws-credentials
    key: AWS_CREDENTIALS
```

**Create `production/helm/iris-model/templates/deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "iris-model.fullname" . }}
  labels:
    {{- include "iris-model.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "iris-model.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "iris-model.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          env:
            {{- toYaml .Values.env | nindent 12 }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

**Create `production/security/rbac.yaml`:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: iris-model-sa
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: iris-model-role
  namespace: default
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: iris-model-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: iris-model-sa
  namespace: default
roleRef:
  kind: Role
  name: iris-model-role
  apiGroup: rbac.authorization.k8s.io
```

**Create `production/scripts/deploy.sh`:**
```bash
#!/bin/bash

set -e

# Configuration
NAMESPACE=${NAMESPACE:-default}
IMAGE_TAG=${IMAGE_TAG:-latest}
ENVIRONMENT=${ENVIRONMENT:-production}

echo "🚀 Starting deployment to $ENVIRONMENT environment..."

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t yourusername/iris-model:$IMAGE_TAG .

echo "🔄 Pushing Docker image..."
docker push yourusername/iris-model:$IMAGE_TAG

# Deploy with Helm
echo "🎯 Deploying to Kubernetes..."
helm upgrade --install iris-model \
  ./helm/iris-model \
  --namespace $NAMESPACE \
  --set image.tag=$IMAGE_TAG \
  --set environment=$ENVIRONMENT \
  --wait \
  --timeout 300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=iris-model
kubectl get services -n $NAMESPACE -l app.kubernetes.io/name=iris-model

# Run health checks
echo "🏥 Running health checks..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=iris-model -n $NAMESPACE --timeout=300s

# Test endpoints
echo "🧪 Testing endpoints..."
SERVICE_URL=$(kubectl get service iris-model -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl -f http://$SERVICE_URL/health || echo "Health check failed"

echo "🎉 Deployment completed successfully!"
```

**Create `production/scripts/rollback.sh`:**
```bash
#!/bin/bash

set -e

NAMESPACE=${NAMESPACE:-default}
REVISION=${REVISION:-0}

echo "🔄 Rolling back deployment..."

if [ "$REVISION" -eq 0 ]; then
    # Rollback to previous revision
    helm rollback iris-model -n $NAMESPACE
else
    # Rollback to specific revision
    helm rollback iris-model $REVISION -n $NAMESPACE
fi

echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/iris-model -n $NAMESPACE --timeout=300s

echo "✅ Rollback completed successfully!"
```

**✅ Master:** Production deployment, security, monitoring, scaling

---

## 🎯 **Final Assessment & Practice (2 hours)**

### **Mock Interview Questions (1 hour)**

**Technical Questions:**
1. **Explain MLOps pipeline**: Walk through data → training → deployment → monitoring
2. **Container vs VM**: Resource efficiency, isolation, portability
3. **Kubernetes scaling**: HPA, VPA, cluster autoscaling
4. **Model monitoring**: Drift detection, performance tracking, alerts
5. **CI/CD for ML**: Testing strategies, deployment patterns, rollback procedures

**Practical Scenarios:**
1. **Model performance degradation**: How to detect and respond?
2. **Scaling issues**: API latency increases with load
3. **Security breach**: Container compromised, what's your response?
4. **Data drift detected**: Model accuracy drops from 95% to 75%
5. **Deployment failure**: New model version fails health checks

### **Hands-On Challenge (1 hour)**

**Challenge**: Deploy a complete MLOps pipeline in 60 minutes

**Requirements:**
- ✅ Containerized model serving
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Kubernetes deployment with scaling
- ✅ Monitoring with Prometheus/Grafana
- ✅ Automated testing and validation

**Success Criteria:**
- API responds to predictions in <200ms
- Pipeline triggers on code changes
- Monitoring shows key metrics
- System scales under load
- All tests pass

---

## 🏆 **Mastery Checklist**

After completing this 2-day intensive, you should master:

### **Core MLOps (Day 1)**
- ✅ MLflow experiment tracking and model registry
- ✅ Docker containerization best practices
- ✅ GitHub Actions CI/CD pipelines
- ✅ Kubernetes deployment and scaling
- ✅ API development with FastAPI
- ✅ Automated testing strategies

### **Advanced MLOps (Day 2)**
- ✅ Infrastructure as Code with Terraform
- ✅ Production monitoring and alerting
- ✅ Pipeline orchestration (Airflow/Kubeflow)
- ✅ Security and RBAC implementation
- ✅ Helm charts and GitOps
- ✅ Disaster recovery and rollback procedures

### **Soft Skills**
- ✅ Communication with data science teams
- ✅ Problem-solving under pressure
- ✅ System design thinking
- ✅ Production mindset and reliability

---

## 🚀 **Next Steps**

1. **Practice Daily**: Build one component each day
2. **Join Communities**: MLOps Discord, Reddit, Stack Overflow
3. **Real Projects**: Contribute to open-source MLOps projects
4. **Certifications**: AWS ML Specialty, GCP ML Engineer
5. **Stay Updated**: Follow MLOps blogs, conferences, papers

**Remember**: MLOps is about reliability, scalability, and automation. Focus on solving real problems, not just using tools!