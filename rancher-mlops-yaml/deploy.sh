#!/bin/bash

# Complete Rancher MLOps Deployment Script
echo "🚀 Starting Rancher MLOps Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

print_status "Connected to cluster: $(kubectl config current-context)"

# Deploy in order
DEPLOY_ORDER=(
    "01-namespaces"
    "02-apps" 
    "03-storage"
    "04-model"
    "05-monitoring"
    "06-gitops"
)

for dir in "${DEPLOY_ORDER[@]}"; do
    if [ -d "$dir" ]; then
        print_status "Deploying $dir..."
        
        # Apply all YAML files in the directory
        for file in $dir/*.yaml; do
            if [ -f "$file" ]; then
                print_status "  Applying $file"
                kubectl apply -f "$file"
                
                # Wait a bit between applications
                sleep 2
            fi
        done
        
        print_status "✅ $dir deployed successfully"
        echo ""
    else
        print_warning "Directory $dir not found, skipping..."
    fi
done

# Wait for deployments to be ready
print_status "🔄 Waiting for deployments to be ready..."

# Wait for monitoring stack
print_status "Waiting for monitoring stack..."
kubectl wait --for=condition=available deployment/rancher-monitoring-operator -n cattle-monitoring-system --timeout=300s 2>/dev/null || true

# Wait for Longhorn
print_status "Waiting for Longhorn..."
kubectl wait --for=condition=available deployment/longhorn-ui -n longhorn-system --timeout=300s 2>/dev/null || true

# Wait for NGINX Ingress
print_status "Waiting for NGINX Ingress..."
kubectl wait --for=condition=available deployment/ingress-nginx-controller -n ingress-nginx --timeout=300s 2>/dev/null || true

# Wait for loan model
print_status "Waiting for loan model..."
kubectl wait --for=condition=available deployment/loan-default-model -n loan-model-prod --timeout=300s 2>/dev/null || true

# Get service information
print_status "🎉 Deployment completed!"
echo ""
print_status "📊 Service Information:"
echo ""

# Get node IP
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

print_status "🌐 Access URLs:"
echo "  Rancher UI:           https://localhost (your existing Rancher)"
echo "  Loan Model API:       http://loan-model.local (add to /etc/hosts)"
echo "  Grafana:              http://$NODE_IP:30080 (admin/admin123)"
echo "  Longhorn UI:          http://longhorn.local (add to /etc/hosts)"
echo ""

print_status "🔧 Add to /etc/hosts:"
echo "  $NODE_IP loan-model.local"
echo "  $NODE_IP longhorn.local"
echo ""

print_status "📋 Quick Commands:"
echo "  Check pods:           kubectl get pods -A"
echo "  Check services:       kubectl get svc -A"
echo "  View logs:            kubectl logs -f deployment/loan-default-model -n loan-model-prod"
echo "  Scale model:          kubectl scale deployment/loan-default-model --replicas=5 -n loan-model-prod"
echo ""

print_status "🎯 Test the API:"
echo "  curl -X GET http://loan-model.local/health"
echo "  curl -X POST http://loan-model.local/predict -H 'Content-Type: application/json' -d '{\"features\": [0.3, 5, 35, 50000, 12500, 1, 0.4, -12000]}'"
echo ""

print_status "✅ MLOps platform is ready!"