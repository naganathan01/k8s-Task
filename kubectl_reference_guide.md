# Complete kubectl Command Reference Guide

## 1. 🏷️ **NAMESPACE OPERATIONS**

### Basic Namespace Commands
```bash
# List all namespaces
kubectl get namespaces
kubectl get ns

# Create a namespace
kubectl create namespace <namespace-name>
kubectl create ns <namespace-name>

# Delete a namespace (deletes all resources in it)
kubectl delete namespace <namespace-name>

# Describe a namespace
kubectl describe namespace <namespace-name>

# Set default namespace for current context
kubectl config set-context --current --namespace=<namespace-name>

# Get current namespace
kubectl config view --minify | grep namespace:
```

**Examples:**
```bash
kubectl create namespace mlops-monitoring
kubectl get ns
kubectl delete namespace test-ns
```

---

## 2. 🏃 **POD OPERATIONS**

### Basic Pod Commands
```bash
# List all pods in current namespace
kubectl get pods
kubectl get po

# List pods in specific namespace
kubectl get pods -n <namespace>

# List pods in all namespaces
kubectl get pods --all-namespaces
kubectl get pods -A

# Show detailed pod information
kubectl get pods -o wide

# Describe a specific pod
kubectl describe pod <pod-name>
kubectl describe pod <pod-name> -n <namespace>

# Get pod logs
kubectl logs <pod-name>
kubectl logs <pod-name> -n <namespace>

# Follow logs (real-time)
kubectl logs -f <pod-name>

# Get logs from specific container in pod
kubectl logs <pod-name> -c <container-name>

# Execute command in pod
kubectl exec -it <pod-name> -- <command>
kubectl exec -it <pod-name> -- /bin/bash

# Copy files to/from pod
kubectl cp <local-file> <pod-name>:<path>
kubectl cp <pod-name>:<path> <local-file>

# Delete a pod
kubectl delete pod <pod-name>

# Create pod from YAML
kubectl apply -f pod.yaml

# Port forward from pod
kubectl port-forward <pod-name> <local-port>:<pod-port>
```

**Examples:**
```bash
kubectl get pods -n mlops-monitoring
kubectl logs grafana-794ddf4bd9-f5cwb -n mlops-monitoring
kubectl exec -it ml-model-simulator-7d9b65966f-n6zl6 -n mlops-monitoring -- /bin/bash
kubectl port-forward grafana-794ddf4bd9-f5cwb 3000:3000 -n mlops-monitoring
```

---

## 3. 📦 **DEPLOYMENT OPERATIONS**

### Basic Deployment Commands
```bash
# List all deployments
kubectl get deployments
kubectl get deploy

# List deployments in specific namespace
kubectl get deployments -n <namespace>

# Create deployment
kubectl create deployment <name> --image=<image>

# Apply deployment from YAML
kubectl apply -f deployment.yaml

# Describe deployment
kubectl describe deployment <deployment-name>

# Scale deployment
kubectl scale deployment <deployment-name> --replicas=<number>

# Update deployment image
kubectl set image deployment/<deployment-name> <container-name>=<new-image>

# Check rollout status
kubectl rollout status deployment/<deployment-name>

# Rollback deployment
kubectl rollout undo deployment/<deployment-name>

# See rollout history
kubectl rollout history deployment/<deployment-name>

# Delete deployment
kubectl delete deployment <deployment-name>

# Restart deployment (rolling restart)
kubectl rollout restart deployment/<deployment-name>
```

**Examples:**
```bash
kubectl get deployments -n mlops-monitoring
kubectl scale deployment ml-model-simulator --replicas=3 -n mlops-monitoring
kubectl rollout status deployment/prometheus -n mlops-monitoring
```

---

## 4. 🔗 **SERVICE OPERATIONS**

### Basic Service Commands
```bash
# List all services
kubectl get services
kubectl get svc

# List services in specific namespace
kubectl get services -n <namespace>

# Describe a service
kubectl describe service <service-name>

# Create service
kubectl create service clusterip <name> --tcp=<port>:<target-port>

# Apply service from YAML
kubectl apply -f service.yaml

# Delete service
kubectl delete service <service-name>

# Port forward from service
kubectl port-forward service/<service-name> <local-port>:<service-port>
kubectl port-forward svc/<service-name> <local-port>:<service-port>

# Get service endpoints
kubectl get endpoints <service-name>
```

**Examples:**
```bash
kubectl get svc -n mlops-monitoring
kubectl port-forward svc/grafana 3000:3000 -n mlops-monitoring
kubectl describe svc prometheus -n mlops-monitoring
```

---

## 5. 📊 **CONFIGMAP & SECRETS**

### ConfigMap Commands
```bash
# List all configmaps
kubectl get configmaps
kubectl get cm

# Create configmap from file
kubectl create configmap <name> --from-file=<file-path>

# Create configmap from literal values
kubectl create configmap <name> --from-literal=<key>=<value>

# Describe configmap
kubectl describe configmap <configmap-name>

# Edit configmap
kubectl edit configmap <configmap-name>

# Delete configmap
kubectl delete configmap <configmap-name>

# Apply configmap from YAML
kubectl apply -f configmap.yaml
```

### Secret Commands
```bash
# List all secrets
kubectl get secrets

# Create secret from file
kubectl create secret generic <name> --from-file=<file-path>

# Create secret from literal values
kubectl create secret generic <name> --from-literal=<key>=<value>

# Create docker registry secret
kubectl create secret docker-registry <name> --docker-server=<server> --docker-username=<user> --docker-password=<password>

# Describe secret (values are hidden)
kubectl describe secret <secret-name>

# Get secret values (base64 encoded)
kubectl get secret <secret-name> -o yaml

# Decode secret value
kubectl get secret <secret-name> -o jsonpath='{.data.<key>}' | base64 -d

# Delete secret
kubectl delete secret <secret-name>
```

**Examples:**
```bash
kubectl get cm -n mlops-monitoring
kubectl get secrets -n mlops-monitoring
kubectl describe cm prometheus-config -n mlops-monitoring
```

---

## 6. 💾 **PERSISTENT VOLUME OPERATIONS**

### PV and PVC Commands
```bash
# List persistent volumes
kubectl get persistentvolumes
kubectl get pv

# List persistent volume claims
kubectl get persistentvolumeclaims
kubectl get pvc

# List PVCs in specific namespace
kubectl get pvc -n <namespace>

# Describe PV
kubectl describe pv <pv-name>

# Describe PVC
kubectl describe pvc <pvc-name>

# Delete PVC
kubectl delete pvc <pvc-name>

# Create PVC from YAML
kubectl apply -f pvc.yaml

# Check storage classes
kubectl get storageclass
kubectl get sc
```

**Examples:**
```bash
kubectl get pvc -n mlops-monitoring
kubectl describe pvc prometheus-storage -n mlops-monitoring
kubectl get sc
```

---

## 7. 🔐 **RBAC (Role-Based Access Control)**

### RBAC Commands
```bash
# List cluster roles
kubectl get clusterroles

# List roles in namespace
kubectl get roles -n <namespace>

# List cluster role bindings
kubectl get clusterrolebindings

# List role bindings
kubectl get rolebindings -n <namespace>

# List service accounts
kubectl get serviceaccounts
kubectl get sa

# Create service account
kubectl create serviceaccount <sa-name>

# Describe role
kubectl describe role <role-name>

# Describe cluster role
kubectl describe clusterrole <clusterrole-name>

# Check permissions
kubectl auth can-i <verb> <resource> --as=<user>
kubectl auth can-i get pods --as=system:serviceaccount:default:my-sa
```

**Examples:**
```bash
kubectl get sa -n mlops-monitoring
kubectl auth can-i get pods --as=system:serviceaccount:mlops-monitoring:prometheus
```

---

## 8. 🌐 **INGRESS OPERATIONS**

### Ingress Commands
```bash
# List all ingresses
kubectl get ingress
kubectl get ing

# List ingresses in specific namespace
kubectl get ingress -n <namespace>

# Describe ingress
kubectl describe ingress <ingress-name>

# Create ingress from YAML
kubectl apply -f ingress.yaml

# Delete ingress
kubectl delete ingress <ingress-name>

# Get ingress with more details
kubectl get ingress -o wide
```

**Examples:**
```bash
kubectl get ing -n mlops-monitoring
kubectl describe ing mlops-monitoring-ingress -n mlops-monitoring
```

---

## 9. 📈 **MONITORING & DEBUGGING**

### Resource Usage Commands
```bash
# Get resource usage of nodes
kubectl top nodes

# Get resource usage of pods
kubectl top pods

# Get resource usage of pods in specific namespace
kubectl top pods -n <namespace>

# Get resource usage of specific pod
kubectl top pod <pod-name>

# List events (troubleshooting)
kubectl get events

# List events in specific namespace
kubectl get events -n <namespace>

# Sort events by timestamp
kubectl get events --sort-by='.lastTimestamp'

# Watch events in real-time
kubectl get events --watch

# Get events for specific object
kubectl get events --field-selector involvedObject.name=<object-name>
```

### Debugging Commands
```bash
# Check cluster info
kubectl cluster-info

# Get cluster nodes
kubectl get nodes

# Describe node
kubectl describe node <node-name>

# Check API versions
kubectl api-versions

# Check API resources
kubectl api-resources

# Validate YAML without applying
kubectl apply -f <file.yaml> --dry-run=client

# Check what would be applied
kubectl diff -f <file.yaml>

# Explain resource fields
kubectl explain pod
kubectl explain deployment.spec
```

**Examples:**
```bash
kubectl top pods -n mlops-monitoring
kubectl get events -n mlops-monitoring --sort-by='.lastTimestamp'
kubectl describe node
```

---

## 10. ⚙️ **CONTEXT & CONFIGURATION**

### Context Commands
```bash
# List all contexts
kubectl config get-contexts

# Get current context
kubectl config current-context

# Switch context
kubectl config use-context <context-name>

# Set default namespace for context
kubectl config set-context --current --namespace=<namespace>

# View kubeconfig
kubectl config view

# Set cluster
kubectl config set-cluster <cluster-name> --server=<server-url>

# Set credentials
kubectl config set-credentials <user-name> --token=<token>

# Create context
kubectl config set-context <context-name> --cluster=<cluster> --user=<user>
```

---

## 11. 🔄 **ADVANCED OPERATIONS**

### Batch Operations
```bash
# Apply multiple files
kubectl apply -f <directory>/

# Apply all YAML files in directory
kubectl apply -f .

# Delete all resources in namespace
kubectl delete all --all -n <namespace>

# Label resources
kubectl label pods <pod-name> <key>=<value>

# Annotate resources
kubectl annotate pods <pod-name> <key>=<value>

# Patch resource
kubectl patch pod <pod-name> -p '{"spec":{"containers":[{"name":"<container>","image":"<new-image>"}]}}'

# Get resource in different formats
kubectl get pods -o json
kubectl get pods -o yaml
kubectl get pods -o wide
kubectl get pods -o custom-columns=NAME:.metadata.name,STATUS:.status.phase
```

### Waiting and Watching
```bash
# Wait for condition
kubectl wait --for=condition=ready pod/<pod-name>
kubectl wait --for=condition=available deployment/<deployment-name>

# Watch resources
kubectl get pods --watch
kubectl get pods -w

# Watch specific resource
kubectl get pod <pod-name> --watch
```

---

## 12. 🎯 **COMMON TROUBLESHOOTING COMMANDS**

### Quick Diagnostics
```bash
# Check pod status and recent events
kubectl get pods -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp' | tail -20

# Check resource quotas
kubectl describe quota -n <namespace>

# Check node capacity
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check if image can be pulled
kubectl run test-pod --image=<image> --dry-run=client -o yaml

# Force delete stuck pod
kubectl delete pod <pod-name> --force --grace-period=0

# Check persistent volume status
kubectl get pv,pvc -n <namespace>

# Check service endpoints
kubectl get endpoints -n <namespace>
```

---

## 13. 🏃‍♂️ **QUICK REFERENCE COMMANDS**

### Most Used Commands
```bash
# The Big 5 (most commonly used)
kubectl get pods                          # List pods
kubectl get services                      # List services  
kubectl get deployments                   # List deployments
kubectl logs <pod-name>                   # Get logs
kubectl describe pod <pod-name>           # Describe pod

# Resource Creation
kubectl create namespace <name>           # Create namespace
kubectl apply -f <file.yaml>             # Apply YAML file
kubectl create deployment <name> --image=<image>  # Create deployment

# Resource Deletion
kubectl delete pod <pod-name>             # Delete pod
kubectl delete deployment <deployment-name>  # Delete deployment
kubectl delete -f <file.yaml>            # Delete from YAML

# Access and Debugging
kubectl port-forward <pod-name> <local-port>:<pod-port>  # Port forward
kubectl exec -it <pod-name> -- /bin/bash    # Execute shell in pod
kubectl logs -f <pod-name>                   # Follow logs
```

---

## 📝 **Command Structure Tips**

### General Pattern
```bash
kubectl [command] [TYPE] [NAME] [flags]

# Where:
# command: get, create, apply, delete, describe, etc.
# TYPE: pod, service, deployment, etc.
# NAME: specific resource name
# flags: -n namespace, -o output, etc.
```

### Useful Flags
```bash
-n <namespace>          # Specify namespace
-o <format>             # Output format (yaml, json, wide, etc.)
-f <file>               # From file
--dry-run=client        # Preview without applying
--help                  # Get help for command
-w, --watch             # Watch for changes
-l <selector>           # Label selector
--all-namespaces, -A    # All namespaces
```

### Examples with Flags
```bash
kubectl get pods -n mlops-monitoring -o wide
kubectl apply -f deployment.yaml --dry-run=client
kubectl get pods -l app=grafana -n mlops-monitoring
kubectl logs -f deployment/prometheus -n mlops-monitoring
```

This reference guide covers all the essential kubectl commands you'll need for managing Kubernetes resources. Save it for quick reference!