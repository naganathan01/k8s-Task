# 🚀 High, Medium, and Low Priority Commands

## 🐳 Docker Commands

### 🔴 High Priority
| Command | Use |
|---------|-----|
| `docker ps` | List running containers |
| `docker exec -it <container> bash` | Access shell of running container |
| `docker logs <container>` | View container logs |
| `docker build -t <image>:tag .` | Build Docker image from Dockerfile |
| `docker run -d -p 8080:80 <image>` | Run container and map ports |
| `docker push <ecr-repo>:tag` | Push image to AWS ECR |
| `docker pull <ecr-repo>:tag` | Pull image from ECR |

### 🟠 Medium Priority
| Command | Use |
|---------|-----|
| `docker-compose up -d` | Start containers from docker-compose.yml |
| `docker-compose down` | Stop and remove all containers |
| `docker images` | List all local images |
| `docker tag <image> <ecr-url>` | Tag image for pushing to ECR |

### 🟢 Low Priority
| Command | Use |
|---------|-----|
| `docker system prune` | Clean up unused Docker data |

---

## ☸️ Kubernetes (K8s) Commands

### 🔴 High Priority
| Command | Use |
|---------|-----|
| `kubectl get pods` | List all pods |
| `kubectl get svc` | List services |
| `kubectl logs <pod>` | View logs of a pod |
| `kubectl exec -it <pod> -- bash` | SSH into a pod |
| `kubectl describe pod <pod>` | Pod details and events |
| `kubectl apply -f <file>.yaml` | Apply manifest file |
| `kubectl delete -f <file>.yaml` | Delete resources from manifest |
| `kubectl get deployment` | View all deployments |

### 🟠 Medium Priority
| Command | Use |
|---------|-----|
| `kubectl get nodes` | List nodes in cluster |
| `kubectl top pod` | Pod-level resource usage |
| `kubectl top node` | Node-level resource usage |
| `kubectl delete pod <pod>` | Delete pod |
| `kubectl rollout restart deployment <name>` | Restart a deployment |
| `kubectl scale deployment <name> --replicas=3` | Manually scale replicas |
| `kubectl get events` | View all cluster events |
| `kubectl port-forward <pod> 8080:80` | Forward pod port to local |

### 🟢 Low Priority
| Command | Use |
|---------|-----|
| `kubectl edit deployment <name>` | Live-edit a deployment |
| `kubectl get all` | Get all resources |
| `kubectl delete svc <name>` | Delete a service |
| `kubectl get ingress` | View ingress resources |
| `kubectl expose pod <pod> --type=NodePort` | Expose pod as service |
| `kubectl label pod <pod> env=dev` | Add a label to pod |

---

## ☁️ AWS EKS / CLI Commands

### 🔴 High Priority
| Command | Use |
|---------|-----|
| `aws configure` | Set AWS CLI credentials |
| `aws eks update-kubeconfig --name <cluster>` | Connect to EKS cluster |
| `kubectl config use-context <context>` | Switch context between clusters |
| `aws ecr get-login-password | docker login` | Login to ECR registry |

### 🟠 Medium Priority
| Command | Use |
|---------|-----|
| `aws eks list-clusters` | List EKS clusters |
| `aws eks describe-cluster --name <cluster>` | Get EKS cluster details |
| `aws ecr create-repository --repository-name <name>` | Create new ECR repo |

### 🟢 Low Priority
| Command | Use |
|---------|-----|
| `aws sts get-caller-identity` | Show current AWS identity |
| `aws iam list-roles` | List all IAM roles |
