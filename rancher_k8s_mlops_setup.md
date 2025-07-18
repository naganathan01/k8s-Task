# MLOps Monitoring Stack - Rancher Kubernetes Setup

## 🎯 **Overview**

This guide shows you how to deploy a complete MLOps monitoring stack on Rancher Kubernetes, including:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **ML Model Simulator** for generating realistic metrics
- **Persistent storage** for data retention
- **Ingress** for external access
- **Service monitoring** and alerting

---

## 📋 **Prerequisites**

### **Rancher Setup Requirements**
- Rancher Server running (v2.6+)
- Kubernetes cluster provisioned through Rancher
- `kubectl` configured to access your cluster
- Helm 3 installed
- Ingress controller enabled (nginx-ingress recommended)

### **Verify Your Setup**
```bash
# Check Rancher cluster access
kubectl get nodes

# Check if Helm is installed
helm version

# Check ingress controller
kubectl get ingressclass
```

---

## 🚀 **Step 1: Create Namespace and Setup**

### **Create Namespace**
```bash
kubectl create namespace mlops-monitoring
kubectl label namespace mlops-monitoring name=mlops-monitoring
```

### **Create namespace.yaml**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-monitoring
  labels:
    name: mlops-monitoring
    app.kubernetes.io/name: mlops-monitoring
    app.kubernetes.io/instance: mlops-monitoring
```

---

## 🔧 **Step 2: Setup Persistent Storage**

### **Create storage-class.yaml**
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: mlops-storage
  namespace: mlops-monitoring
provisioner: kubernetes.io/aws-ebs  # Change based on your cloud provider
parameters:
  type: gp3
  fsType: ext4
reclaimPolicy: Retain
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

### **Create persistent-volumes.yaml**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage
  namespace: mlops-monitoring
  labels:
    app: prometheus
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: mlops-storage
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-storage
  namespace: mlops-monitoring
  labels:
    app: grafana
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: mlops-storage
```

---

## 📊 **Step 3: ConfigMaps and Secrets**

### **Create prometheus-config.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: mlops-monitoring
  labels:
    app: prometheus
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'rancher-mlops'
        replica: 'prometheus'

    rule_files:
      - "/etc/prometheus/rules/*.yml"

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093

    scrape_configs:
      - job_name: 'prometheus'
        static_configs:
          - targets: ['localhost:9090']
        scrape_interval: 30s

      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
            action: keep
            regex: default;kubernetes;https

      - job_name: 'kubernetes-nodes'
        kubernetes_sd_configs:
          - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
          - target_label: __address__
            replacement: kubernetes.default.svc:443
          - source_labels: [__meta_kubernetes_node_name]
            regex: (.+)
            target_label: __metrics_path__
            replacement: /api/v1/nodes/${1}/proxy/metrics

      - job_name: 'kubernetes-cadvisor'
        kubernetes_sd_configs:
          - role: node
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
          - target_label: __address__
            replacement: kubernetes.default.svc:443
          - source_labels: [__meta_kubernetes_node_name]
            regex: (.+)
            target_label: __metrics_path__
            replacement: /api/v1/nodes/${1}/proxy/metrics/cadvisor

      - job_name: 'ml-model-simulator'
        kubernetes_sd_configs:
          - role: endpoints
            namespaces:
              names:
                - mlops-monitoring
        relabel_configs:
          - source_labels: [__meta_kubernetes_service_name]
            action: keep
            regex: ml-model-simulator
          - source_labels: [__meta_kubernetes_endpoint_port_name]
            action: keep
            regex: http
        scrape_interval: 5s
        metrics_path: /metrics

      - job_name: 'grafana'
        kubernetes_sd_configs:
          - role: endpoints
            namespaces:
              names:
                - mlops-monitoring
        relabel_configs:
          - source_labels: [__meta_kubernetes_service_name]
            action: keep
            regex: grafana
          - source_labels: [__meta_kubernetes_endpoint_port_name]
            action: keep
            regex: http

  alert-rules.yml: |
    groups:
      - name: mlops-alerts
        rules:
          - alert: MLModelHighLatency
            expr: histogram_quantile(0.95, rate(ml_model_request_duration_seconds_bucket[5m])) > 0.5
            for: 2m
            labels:
              severity: warning
            annotations:
              summary: "ML Model has high latency"
              description: "ML Model 95th percentile latency is {{ $value }} seconds"

          - alert: MLModelLowAccuracy
            expr: ml_model_accuracy < 0.85
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "ML Model accuracy is too low"
              description: "ML Model accuracy is {{ $value }}, below threshold of 0.85"

          - alert: MLModelHighErrorRate
            expr: rate(ml_model_requests_total{status=~"5.."}[5m]) > 0.1
            for: 1m
            labels:
              severity: warning
            annotations:
              summary: "ML Model has high error rate"
              description: "ML Model error rate is {{ $value }} requests per second"

          - alert: PrometheusDown
            expr: up{job="prometheus"} == 0
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "Prometheus is down"
              description: "Prometheus has been down for more than 5 minutes"
```

### **Create grafana-config.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: mlops-monitoring
  labels:
    app: grafana
data:
  datasources.yml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
        basicAuth: false
        editable: true
        jsonData:
          httpMethod: POST
          prometheusType: Prometheus
          prometheusVersion: 2.40.0
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards-config
  namespace: mlops-monitoring
  labels:
    app: grafana
data:
  dashboards.yml: |
    apiVersion: 1
    providers:
      - name: 'mlops-dashboards'
        orgId: 1
        folder: 'MLOps'
        type: file
        disableDeletion: false
        updateIntervalSeconds: 10
        allowUiUpdates: true
        options:
          path: /var/lib/grafana/dashboards
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-mlops
  namespace: mlops-monitoring
  labels:
    app: grafana
data:
  mlops-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "MLOps Model Monitoring",
        "tags": ["mlops", "monitoring", "ml"],
        "timezone": "browser",
        "schemaVersion": 30,
        "version": 1,
        "refresh": "5s",
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "timeseries",
            "targets": [
              {
                "expr": "rate(ml_model_requests_total[5m])",
                "legendFormat": "{{method}} {{endpoint}} {{status}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "reqps",
                "color": {
                  "mode": "palette-classic"
                }
              }
            },
            "gridPos": {
              "h": 8,
              "w": 12,
              "x": 0,
              "y": 0
            }
          },
          {
            "id": 2,
            "title": "Model Accuracy",
            "type": "stat",
            "targets": [
              {
                "expr": "ml_model_accuracy",
                "legendFormat": "Accuracy"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "percent",
                "min": 0,
                "max": 1,
                "thresholds": {
                  "steps": [
                    {
                      "color": "red",
                      "value": 0
                    },
                    {
                      "color": "yellow",
                      "value": 0.8
                    },
                    {
                      "color": "green",
                      "value": 0.9
                    }
                  ]
                }
              }
            },
            "gridPos": {
              "h": 8,
              "w": 12,
              "x": 12,
              "y": 0
            }
          },
          {
            "id": 3,
            "title": "Response Time Percentiles",
            "type": "timeseries",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(ml_model_request_duration_seconds_bucket[5m]))",
                "legendFormat": "95th percentile"
              },
              {
                "expr": "histogram_quantile(0.50, rate(ml_model_request_duration_seconds_bucket[5m]))",
                "legendFormat": "50th percentile"
              },
              {
                "expr": "histogram_quantile(0.99, rate(ml_model_request_duration_seconds_bucket[5m]))",
                "legendFormat": "99th percentile"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "s",
                "color": {
                  "mode": "palette-classic"
                }
              }
            },
            "gridPos": {
              "h": 8,
              "w": 24,
              "x": 0,
              "y": 8
            }
          },
          {
            "id": 4,
            "title": "Active Connections",
            "type": "timeseries",
            "targets": [
              {
                "expr": "ml_model_active_connections",
                "legendFormat": "Active Connections"
              }
            ],
            "gridPos": {
              "h": 8,
              "w": 12,
              "x": 0,
              "y": 16
            }
          },
          {
            "id": 5,
            "title": "Prediction Confidence Distribution",
            "type": "timeseries",
            "targets": [
              {
                "expr": "rate(ml_model_prediction_confidence_sum[5m]) / rate(ml_model_prediction_confidence_count[5m])",
                "legendFormat": "Average Confidence"
              }
            ],
            "gridPos": {
              "h": 8,
              "w": 12,
              "x": 12,
              "y": 16
            }
          }
        ],
        "time": {
          "from": "now-1h",
          "to": "now"
        }
      }
    }
```

### **Create secrets.yaml**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: grafana-secret
  namespace: mlops-monitoring
type: Opaque
data:
  admin-password: YWRtaW4=  # base64 encoded 'admin'
  admin-user: YWRtaW4=      # base64 encoded 'admin'
```

---

## 🤖 **Step 4: ML Model Simulator Deployment**

### **Create ml-model-simulator.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-simulator
  namespace: mlops-monitoring
  labels:
    app: ml-model-simulator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model-simulator
  template:
    metadata:
      labels:
        app: ml-model-simulator
    spec:
      containers:
      - name: ml-model-simulator
        image: python:3.9-slim
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: MODEL_NAME
          value: "iris-classifier"
        - name: MODEL_VERSION
          value: "v1.0.0"
        command:
        - /bin/bash
        - -c
        - |
          pip install fastapi uvicorn prometheus_client numpy scikit-learn
          cat > /app/model_simulator.py << 'EOF'
          from fastapi import FastAPI, Response
          from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
          import time
          import random
          import numpy as np
          from sklearn.ensemble import RandomForestClassifier
          from sklearn.datasets import make_classification
          import uvicorn
          import logging
          import os

          logging.basicConfig(level=logging.INFO)
          logger = logging.getLogger(__name__)

          REQUEST_COUNT = Counter('ml_model_requests_total', 'Total ML model requests', ['method', 'endpoint', 'status'])
          REQUEST_LATENCY = Histogram('ml_model_request_duration_seconds', 'ML model request latency')
          MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
          PREDICTION_CONFIDENCE = Histogram('ml_model_prediction_confidence', 'Model prediction confidence')
          ACTIVE_CONNECTIONS = Gauge('ml_model_active_connections', 'Active connections')

          app = FastAPI(title="ML Model Simulator", version="1.0.0")

          logger.info("Training ML model simulator...")
          X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
          model = RandomForestClassifier(n_estimators=100, random_state=42)
          model.fit(X, y)
          MODEL_ACCURACY.set(0.95)
          logger.info("ML model simulator ready!")

          @app.middleware("http")
          async def add_metrics_middleware(request, call_next):
              start_time = time.time()
              ACTIVE_CONNECTIONS.inc()
              try:
                  response = await call_next(request)
                  REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
                  REQUEST_LATENCY.observe(time.time() - start_time)
                  return response
              except Exception as e:
                  REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()
                  raise
              finally:
                  ACTIVE_CONNECTIONS.dec()

          @app.get("/health")
          async def health_check():
              return {"status": "healthy", "timestamp": time.time(), "model_loaded": True}

          @app.post("/predict")
          async def predict():
              try:
                  prediction_time = random.uniform(0.01, 0.2)
                  time.sleep(prediction_time)
                  
                  input_data = np.random.rand(1, 4)
                  prediction = model.predict(input_data)[0]
                  probabilities = model.predict_proba(input_data)[0]
                  confidence = float(max(probabilities))
                  
                  PREDICTION_CONFIDENCE.observe(confidence)
                  
                  if random.random() < 0.1:
                      MODEL_ACCURACY.set(random.uniform(0.8, 0.95))
                  else:
                      MODEL_ACCURACY.set(random.uniform(0.92, 0.98))
                  
                  return {
                      "prediction": int(prediction),
                      "confidence": confidence,
                      "probabilities": probabilities.tolist(),
                      "timestamp": time.time(),
                      "model_name": os.getenv("MODEL_NAME", "unknown"),
                      "model_version": os.getenv("MODEL_VERSION", "unknown")
                  }
              except Exception as e:
                  logger.error(f"Prediction error: {e}")
                  raise

          @app.get("/metrics")
          async def get_metrics():
              return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

          @app.get("/")
          async def root():
              return {"message": "ML Model Simulator API", "model_name": os.getenv("MODEL_NAME"), "model_version": os.getenv("MODEL_VERSION")}

          if __name__ == "__main__":
              uvicorn.run(app, host="0.0.0.0", port=8000)
          EOF
          cd /app && python model_simulator.py
        workingDir: /app
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
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-simulator
  namespace: mlops-monitoring
  labels:
    app: ml-model-simulator
spec:
  selector:
    app: ml-model-simulator
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-simulator-hpa
  namespace: mlops-monitoring
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-simulator
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 📊 **Step 5: Prometheus Deployment**

### **Create prometheus-rbac.yaml**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: mlops-monitoring
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: mlops-monitoring
```

### **Create prometheus-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: mlops-monitoring
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--storage.tsdb.retention.time=30d'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--web.enable-lifecycle'
        - '--web.route-prefix=/'
        - '--web.enable-admin-api'
        ports:
        - containerPort: 9090
          name: http
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
        - name: rules
          mountPath: /etc/prometheus/rules
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: prometheus-config
          items:
          - key: prometheus.yml
            path: prometheus.yml
      - name: rules
        configMap:
          name: prometheus-config
          items:
          - key: alert-rules.yml
            path: alert-rules.yml
      - name: storage
        persistentVolumeClaim:
          claimName: prometheus-storage
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534
        fsGroup: 65534
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: mlops-monitoring
  labels:
    app: prometheus
spec:
  selector:
    app: prometheus
  ports:
  - name: http
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
```

---

## 📈 **Step 6: Grafana Deployment**

### **Create grafana-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: mlops-monitoring
  labels:
    app: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: GF_SECURITY_ADMIN_USER
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-user
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-password
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        - name: GF_INSTALL_PLUGINS
          value: "grafana-piechart-panel,grafana-worldmap-panel"
        volumeMounts:
        - name: storage
          mountPath: /var/lib/grafana
        - name: datasources
          mountPath: /etc/grafana/provisioning/datasources
        - name: dashboards-config
          mountPath: /etc/grafana/provisioning/dashboards
        - name: dashboards
          mountPath: /var/lib/grafana/dashboards
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: grafana-storage
      - name: datasources
        configMap:
          name: grafana-datasources
      - name: dashboards-config
        configMap:
          name: grafana-dashboards-config
      - name: dashboards
        configMap:
          name: grafana-dashboard-mlops
      securityContext:
        runAsNonRoot: true
        runAsUser: 472
        fsGroup: 472
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: mlops-monitoring
  labels:
    app: grafana
spec:
  selector:
    app: grafana
  ports:
  - name: http
    port: 3000
    targetPort: 3000
    protocol: TCP
  type: ClusterIP
```

---

## 🌐 **Step 7: Ingress Configuration**

### **Create ingress.yaml**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-monitoring-ingress
  namespace: mlops-monitoring
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"  # If using cert-manager
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - grafana.yourdomain.com
    - prometheus.yourdomain.com
    - ml-model.yourdomain.com
    secretName: mlops-monitoring-tls
  rules:
  - host: grafana.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: grafana
            port:
              number: 3000
  - host: prometheus.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prometheus
            port:
              number: 9090
  - host: ml-model.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-model-simulator
            port:
              number: 8000
```

---

## 🔄 **Step 8: Deployment Scripts**

### **Create deploy.sh**
```bash
#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying MLOps Monitoring Stack to Rancher Kubernetes${NC}"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl is not configured or cluster is not accessible${NC}"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace mlops-monitoring &> /dev/null; then
    echo -e "${YELLOW}📦 Creating namespace...${NC}"
    kubectl apply -f namespace.yaml
fi

# Apply storage
echo -e "${YELLOW}💾 Setting up persistent storage...${NC}"
kubectl apply -f storage-class.yaml
kubectl apply -f persistent-volumes.yaml

# Apply RBAC
echo -e "${YELLOW}🔐 Setting up RBAC...${NC}"
kubectl apply -f prometheus-rbac.yaml

# Apply ConfigMaps and Secrets
echo -e "${YELLOW}⚙️ Applying configurations...${NC}"
kubectl apply -f prometheus-config.yaml
kubectl apply -f grafana-config.yaml
kubectl apply -f secrets.yaml

# Deploy applications
echo -e "${YELLOW}🤖 Deploying ML Model Simulator...${NC}"
kubectl apply -f ml-model-simulator.yaml

echo -e "${YELLOW}📊 Deploying Prometheus...${NC}"
kubectl apply -f prometheus-deployment.yaml

echo -e "${YELLOW}📈 Deploying Grafana...${NC}"
kubectl apply -f grafana-deployment.yaml

# Setup Ingress
echo -e "${YELLOW}🌐 Setting up Ingress...${NC}"
kubectl apply -f ingress.yaml

# Wait for deployments to be ready
echo -e "${YELLOW}⏳ Waiting for deployments to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/ml-model-simulator -n mlops-monitoring
kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n mlops-monitoring
kubectl wait --for=condition=available --timeout=300s deployment/grafana -n mlops-monitoring

# Get service information
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}📋 Service Information:${NC}"
kubectl get services -n mlops-monitoring
echo ""
kubectl get ingress -n mlops-monitoring
echo ""
kubectl get pods -n mlops-monitoring

echo -e "${GREEN}🎉 MLOps Monitoring Stack is now running!${NC}"
echo -e "${YELLOW}Access URLs:${NC}"
echo -e "  Grafana: https://grafana.yourdomain.com (admin/admin)"
echo -e "  Prometheus: https://prometheus.yourdomain.com"
echo -e "  ML Model API: https://ml-model.yourdomain.com"
echo ""
echo -e "${YELLOW}Local Port-Forward (if no ingress):${NC}"
echo -e "  kubectl port-forward -n mlops-monitoring svc/grafana 3000:3000"
echo -e "  kubectl port-forward -n mlops-monitoring svc/prometheus 9090:9090"
echo -e "  kubectl port-forward -n mlops-monitoring svc/ml-model-simulator 8000:8000"
```

### **Create monitoring-check.sh**
```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 MLOps Monitoring Stack Health Check${NC}"

# Check namespace
echo -e "${YELLOW}📦 Checking namespace...${NC}"
if kubectl get namespace mlops-monitoring &> /dev/null; then
    echo -e "${GREEN}✅ Namespace mlops-monitoring exists${NC}"
else
    echo -e "${RED}❌ Namespace mlops-monitoring not found${NC}"
    exit 1
fi

# Check deployments
echo -e "${YELLOW}🚀 Checking deployments...${NC}"
DEPLOYMENTS=("ml-model-simulator" "prometheus" "grafana")

for deployment in "${DEPLOYMENTS[@]}"; do
    if kubectl get deployment $deployment -n mlops-monitoring &> /dev/null; then
        READY=$(kubectl get deployment $deployment -n mlops-monitoring -o jsonpath='{.status.readyReplicas}')
        DESIRED=$(kubectl get deployment $deployment -n mlops-monitoring -o jsonpath='{.spec.replicas}')
        if [ "$READY" = "$DESIRED" ]; then
            echo -e "${GREEN}✅ $deployment: $READY/$DESIRED ready${NC}"
        else
            echo -e "${RED}❌ $deployment: $READY/$DESIRED ready${NC}"
        fi
    else
        echo -e "${RED}❌ $deployment: not found${NC}"
    fi
done

# Check services
echo -e "${YELLOW}🔗 Checking services...${NC}"
SERVICES=("ml-model-simulator" "prometheus" "grafana")

for service in "${SERVICES[@]}"; do
    if kubectl get service $service -n mlops-monitoring &> /dev/null; then
        echo -e "${GREEN}✅ Service $service exists${NC}"
    else
        echo -e "${RED}❌ Service $service not found${NC}"
    fi
done

# Check persistent volumes
echo -e "${YELLOW}💾 Checking persistent volumes...${NC}"
PVCs=("prometheus-storage" "grafana-storage")

for pvc in "${PVCs[@]}"; do
    STATUS=$(kubectl get pvc $pvc -n mlops-monitoring -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$STATUS" = "Bound" ]; then
        echo -e "${GREEN}✅ PVC $pvc: $STATUS${NC}"
    else
        echo -e "${RED}❌ PVC $pvc: $STATUS${NC}"
    fi
done

# Check ingress
echo -e "${YELLOW}🌐 Checking ingress...${NC}"
if kubectl get ingress mlops-monitoring-ingress -n mlops-monitoring &> /dev/null; then
    echo -e "${GREEN}✅ Ingress mlops-monitoring-ingress exists${NC}"
    kubectl get ingress mlops-monitoring-ingress -n mlops-monitoring
else
    echo -e "${RED}❌ Ingress mlops-monitoring-ingress not found${NC}"
fi

# Test endpoints
echo -e "${YELLOW}🧪 Testing endpoints...${NC}"

# Port-forward and test (in background)
kubectl port-forward -n mlops-monitoring svc/ml-model-simulator 8000:8000 &
PF_PID=$!
sleep 3

# Test health endpoint
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ ML Model health endpoint responding${NC}"
else
    echo -e "${RED}❌ ML Model health endpoint not responding${NC}"
fi

# Test metrics endpoint
if curl -s http://localhost:8000/metrics > /dev/null; then
    echo -e "${GREEN}✅ ML Model metrics endpoint responding${NC}"
else
    echo -e "${RED}❌ ML Model metrics endpoint not responding${NC}"
fi

# Clean up port-forward
kill $PF_PID 2>/dev/null

echo -e "${GREEN}🎉 Health check completed!${NC}"
```

### **Create load-test.sh**
```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 MLOps Load Testing Script${NC}"

# Check if ML model service is available
if ! kubectl get service ml-model-simulator -n mlops-monitoring &> /dev/null; then
    echo -e "${RED}❌ ML Model service not found${NC}"
    exit 1
fi

# Start port-forward
echo -e "${YELLOW}🔗 Setting up port-forward...${NC}"
kubectl port-forward -n mlops-monitoring svc/ml-model-simulator 8000:8000 &
PF_PID=$!
sleep 3

# Function to cleanup
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    kill $PF_PID 2>/dev/null
}

# Set trap for cleanup
trap cleanup EXIT

# Test single request
echo -e "${YELLOW}🧪 Testing single request...${NC}"
response=$(curl -s -X POST http://localhost:8000/predict)
if echo "$response" | grep -q "prediction"; then
    echo -e "${GREEN}✅ Single request successful${NC}"
    echo "Response: $response"
else
    echo -e "${RED}❌ Single request failed${NC}"
    exit 1
fi

# Load test parameters
CONCURRENT_USERS=10
REQUESTS_PER_USER=10
TOTAL_REQUESTS=$((CONCURRENT_USERS * REQUESTS_PER_USER))

echo -e "${YELLOW}🔥 Starting load test...${NC}"
echo "  Concurrent users: $CONCURRENT_USERS"
echo "  Requests per user: $REQUESTS_PER_USER"
echo "  Total requests: $TOTAL_REQUESTS"

# Start load test
start_time=$(date +%s)

for i in $(seq 1 $CONCURRENT_USERS); do
    {
        for j in $(seq 1 $REQUESTS_PER_USER); do
            curl -s -X POST http://localhost:8000/predict > /dev/null
        done
    } &
done

# Wait for all background jobs to complete
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "${GREEN}✅ Load test completed!${NC}"
echo "  Duration: ${duration}s"
echo "  Total requests: $TOTAL_REQUESTS"
echo "  Requests per second: $((TOTAL_REQUESTS / duration))"

# Show metrics
echo -e "${YELLOW}📊 Fetching metrics...${NC}"
curl -s http://localhost:8000/metrics | grep -E "(ml_model_requests_total|ml_model_request_duration_seconds)"

echo -e "${GREEN}🎉 Load test completed successfully!${NC}"
echo -e "${YELLOW}📈 Check Grafana dashboards for detailed metrics${NC}"
```

---

## 🎯 **Step 9: Rancher-Specific Configuration**

### **Create rancher-project.yaml**
```yaml
apiVersion: management.cattle.io/v3
kind: Project
metadata:
  name: mlops-monitoring
  namespace: cattle-global-data
spec:
  clusterId: "c-m-xxxxx"  # Replace with your cluster ID
  displayName: "MLOps Monitoring"
  description: "MLOps monitoring and observability stack"
  resourceQuota:
    limit:
      limitsCpu: "4000m"
      limitsMemory: "8Gi"
      requestsCpu: "2000m"
      requestsMemory: "4Gi"
  namespaceDefaultResourceQuota:
    limit:
      limitsCpu: "2000m"
      limitsMemory: "4Gi"
      requestsCpu: "1000m"
      requestsMemory: "2Gi"
```

### **Create rancher-monitoring-config.yaml**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rancher-monitoring-config
  namespace: mlops-monitoring
  labels:
    app: rancher-monitoring
data:
  prometheus-additional.yml: |
    - job_name: 'rancher-cluster-metrics'
      kubernetes_sd_configs:
        - role: endpoints
          namespaces:
            names:
              - cattle-system
              - cattle-monitoring-system
      relabel_configs:
        - source_labels: [__meta_kubernetes_service_name]
          action: keep
          regex: 'rancher-monitoring-.*'
        - source_labels: [__meta_kubernetes_endpoint_port_name]
          action: keep
          regex: 'http-metrics'

    - job_name: 'rancher-cluster-agent'
      kubernetes_sd_configs:
        - role: endpoints
          namespaces:
            names:
              - cattle-system
      relabel_configs:
        - source_labels: [__meta_kubernetes_service_name]
          action: keep
          regex: 'cattle-cluster-agent'
```

---

## 🎛️ **Step 10: Helm Chart Alternative**

### **Create helm-values.yaml**
```yaml
# values.yaml for Helm deployment
global:
  namespace: mlops-monitoring
  storageClass: mlops-storage
  domain: yourdomain.com

prometheus:
  enabled: true
  retention: 30d
  storage:
    size: 20Gi
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
  serviceMonitor:
    enabled: true
    interval: 30s

grafana:
  enabled: true
  adminPassword: admin
  persistence:
    enabled: true
    size: 10Gi
  plugins:
    - grafana-piechart-panel
    - grafana-worldmap-panel
  dashboardsConfigMaps:
    - configMapName: grafana-dashboard-mlops
      fileName: mlops-dashboard.json
  datasources:
    prometheus:
      url: http://prometheus:9090
      isDefault: true

mlModelSimulator:
  enabled: true
  replicaCount: 2
  image:
    repository: python
    tag: "3.9-slim"
  service:
    type: ClusterIP
    port: 8000
  ingress:
    enabled: true
    hostname: ml-model.yourdomain.com
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"

ingress:
  enabled: true
  ingressClassName: nginx
  tls:
    enabled: true
    secretName: mlops-monitoring-tls
  hosts:
    - host: grafana.yourdomain.com
      service: grafana
      port: 3000
    - host: prometheus.yourdomain.com
      service: prometheus
      port: 9090
    - host: ml-model.yourdomain.com
      service: ml-model-simulator
      port: 8000
```

### **Create Chart.yaml**
```yaml
apiVersion: v2
name: mlops-monitoring
description: Complete MLOps monitoring stack for Kubernetes
type: application
version: 1.0.0
appVersion: "1.0.0"
home: https://github.com/your-org/mlops-monitoring
sources:
  - https://github.com/your-org/mlops-monitoring
maintainers:
  - name: MLOps Team
    email: mlops@yourcompany.com
dependencies:
  - name: prometheus
    version: "15.x.x"
    repository: "https://prometheus-community.github.io/helm-charts"
    condition: prometheus.enabled
  - name: grafana
    version: "6.x.x"
    repository: "https://grafana.github.io/helm-charts"
    condition: grafana.enabled
```

---

## 🔧 **Step 11: Deployment Commands**

### **Complete Deployment Process**

```bash
# 1. Clone or create your setup directory
mkdir mlops-rancher-setup
cd mlops-rancher-setup

# 2. Create all the YAML files from above
# (Copy all the YAML content into respective files)

# 3. Make scripts executable
chmod +x deploy.sh
chmod +x monitoring-check.sh
chmod +x load-test.sh

# 4. Deploy the stack
./deploy.sh

# 5. Check deployment status
./monitoring-check.sh

# 6. Run load test
./load-test.sh
```

### **Alternative: Using Helm**

```bash
# Create Helm chart
helm create mlops-monitoring
cd mlops-monitoring

# Replace values.yaml with helm-values.yaml content
# Add templates from the YAML files above

# Install with Helm
helm install mlops-monitoring . \
  --namespace mlops-monitoring \
  --create-namespace \
  --values values.yaml

# Upgrade
helm upgrade mlops-monitoring . \
  --namespace mlops-monitoring \
  --values values.yaml
```

### **Port-Forward for Local Access**

```bash
# If you don't have ingress configured
kubectl port-forward -n mlops-monitoring svc/grafana 3000:3000 &
kubectl port-forward -n mlops-monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n mlops-monitoring svc/ml-model-simulator 8000:8000 &

# Access services
echo "Grafana: http://localhost:3000 (admin/admin)"
echo "Prometheus: http://localhost:9090"
echo "ML Model API: http://localhost:8000"
```

---

## 📊 **Step 12: Verification and Testing**

### **1. Verify Deployments**
```bash
# Check all pods are running
kubectl get pods -n mlops-monitoring

# Check services
kubectl get svc -n mlops-monitoring

# Check ingress
kubectl get ingress -n mlops-monitoring

# Check persistent volumes
kubectl get pvc -n mlops-monitoring
```

### **2. Test API Endpoints**
```bash
# Test ML model (using port-forward)
curl -X POST http://localhost:8000/predict

# Test health check
curl http://localhost:8000/health

# Test metrics
curl http://localhost:8000/metrics
```

### **3. Access Dashboards**
- **Grafana**: Navigate to your Grafana URL
- **Import dashboards**: The MLOps dashboard should be pre-configured
- **Verify metrics**: Check if ML model metrics are showing up

### **4. Monitor Logs**
```bash
# Check application logs
kubectl logs -n mlops-monitoring -l app=ml-model-simulator
kubectl logs -n mlops-monitoring -l app=prometheus
kubectl logs -n mlops-monitoring -l app=grafana
```

---

## 🛠️ **Step 13: Troubleshooting**

### **Common Issues and Solutions**

**1. Pods not starting:**
```bash
# Check pod status
kubectl describe pod <pod-name> -n mlops-monitoring

# Check logs
kubectl logs <pod-name> -n mlops-monitoring

# Check resource limits
kubectl top pods -n mlops-monitoring
```

**2. Persistent volumes not binding:**
```bash
# Check PVC status
kubectl get pvc -n mlops-monitoring
kubectl describe pvc <pvc-name> -n mlops-monitoring

# Check storage class
kubectl get storageclass
```

**3. Ingress not working:**
```bash
# Check ingress controller
kubectl get ingressclass
kubectl get pods -n ingress-nginx

# Check ingress configuration
kubectl describe ingress mlops-monitoring-ingress -n mlops-monitoring
```

**4. Services not accessible:**
```bash
# Check service endpoints
kubectl get endpoints -n mlops-monitoring

# Test internal connectivity
kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- sh
# Inside the pod: curl http://ml-model-simulator.mlops-monitoring.svc.cluster.local:8000/health
```

---

## 🏆 **Step 14: Production Enhancements**

### **Security Enhancements**
```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlops-monitoring-network-policy
  namespace: mlops-monitoring
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: mlops-monitoring
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: mlops-monitoring
```

### **Resource Quotas**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: mlops-monitoring-quota
  namespace: mlops-monitoring
spec:
  hard:
    requests.cpu: "2"
    requests.memory: 4Gi
    limits.cpu: "4"
    limits.memory: 8Gi
    persistentvolumeclaims: "5"
```

### **Pod Disruption Budget**
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: ml-model-simulator-pdb
  namespace: mlops-monitoring
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: ml-model-simulator
```

---

## 🎉 **Success Metrics**

After successful deployment, you should see:

✅ **All pods running** in mlops-monitoring namespace
✅ **Grafana dashboard** showing ML model metrics
✅ **Prometheus** scraping metrics from all services
✅ **ML Model API** responding to predictions
✅ **Ingress** providing external access (if configured)
✅ **Auto-scaling** working based on CPU/memory usage
✅ **Persistent storage** maintaining data across restarts
✅ **Monitoring alerts** configured for critical thresholds

This setup provides a production-ready MLOps monitoring stack on Rancher Kubernetes that demonstrates enterprise-level practices and can be used as a foundation for real ML deployments.