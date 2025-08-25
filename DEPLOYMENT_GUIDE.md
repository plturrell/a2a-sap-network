# A2A Platform Deployment Guide

## Quick Start Deployment

### 1. Prerequisites Verification
```bash
# Check Python version (requires 3.9+)
python3 --version

# Check Node.js version (requires 16+)
node --version

# Check system resources
free -h  # Linux
vm_stat  # macOS

# Verify all dependencies
pip3 install -r requirements.txt
npm install
```

### 2. Environment Configuration
```bash
# Copy environment templates
cp .env.example .env
cp a2aAgents/.env.example a2aAgents/.env
cp a2aNetwork/.env.example a2aNetwork/.env

# Set required environment variables
export A2A_ENV=production
export A2A_PORT=8000
export A2A_DB_PATH=/var/lib/a2a/database
export A2A_LOG_LEVEL=INFO
```

### 3. Database Setup
```bash
# Initialize databases
python3 scripts/init_database.py

# Run migrations
python3 -m alembic upgrade head

# Verify database
sqlite3 a2a_data.db "SELECT COUNT(*) FROM agents;"
```

### 4. Start Services
```bash
# Start all services
./start.sh

# Or start individually:
# Backend agents
python3 -m a2aAgents.backend.main

# Frontend (separate terminal)
cd a2aAgents/frontend
npm start

# Network service (separate terminal)
cd a2aNetwork
npm start
```

## Production Deployment Options

### Option 1: Docker Deployment
```bash
# Build Docker image
docker build -t a2a-platform:latest .

# Run container
docker run -d \
  --name a2a-platform \
  -p 8000:8000 \
  -v /path/to/data:/app/data \
  -e A2A_ENV=production \
  a2a-platform:latest

# Check container status
docker ps
docker logs a2a-platform
```

### Option 2: Kubernetes Deployment
```yaml
# a2a-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a2a-platform
  template:
    metadata:
      labels:
        app: a2a-platform
    spec:
      containers:
      - name: a2a-platform
        image: a2a-platform:latest
        ports:
        - containerPort: 8000
        env:
        - name: A2A_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

```bash
# Deploy to Kubernetes
kubectl apply -f a2a-deployment.yaml
kubectl apply -f a2a-service.yaml
kubectl get pods -l app=a2a-platform
```

### Option 3: Cloud Platform Deployment

#### AWS EC2
```bash
# Launch EC2 instance (t3.large recommended)
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Install dependencies
sudo yum update -y
sudo yum install python3 python3-pip nodejs -y

# Clone and setup
git clone https://github.com/your-org/a2a-platform.git
cd a2a-platform
pip3 install -r requirements.txt
npm install

# Configure systemd service
sudo cp deployment/a2a-platform.service /etc/systemd/system/
sudo systemctl enable a2a-platform
sudo systemctl start a2a-platform
```

#### Google Cloud Platform
```bash
# Create GCE instance
gcloud compute instances create a2a-platform \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# Deploy using Cloud Run
gcloud run deploy a2a-platform \
  --image gcr.io/your-project/a2a-platform \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2
```

#### Azure
```bash
# Create Azure VM
az vm create \
  --resource-group a2a-rg \
  --name a2a-platform \
  --image UbuntuLTS \
  --size Standard_D2s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Deploy to Azure App Service
az webapp create \
  --resource-group a2a-rg \
  --plan a2a-plan \
  --name a2a-platform \
  --runtime "PYTHON|3.9"
```

## Load Balancing Configuration

### NGINX Configuration
```nginx
upstream a2a_backend {
    least_conn;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    server_name a2a.yourdomain.com;

    location / {
        proxy_pass http://a2a_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # SSL configuration
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
}
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'a2a-platform'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/metrics'
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "A2A Platform Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(a2a_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, a2a_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Active Agents",
        "targets": [
          {
            "expr": "a2a_active_agents"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening

### 1. Firewall Rules
```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw allow 8000/tcp # A2A Platform
sudo ufw enable
```

### 2. SSL/TLS Setup
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d a2a.yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

### 3. Security Headers
```python
# Add to main application
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app, force_https=True)

# Security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response
```

## Backup Strategy

### Automated Backups
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/a2a"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup databases
sqlite3 a2a_data.db ".backup $BACKUP_DIR/a2a_data_$TIMESTAMP.db"

# Backup configuration
tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz .env* *.json *.yaml

# Backup logs
tar -czf $BACKUP_DIR/logs_$TIMESTAMP.tar.gz logs/

# Keep only last 30 days of backups
find $BACKUP_DIR -type f -mtime +30 -delete

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-bucket/a2a-backups/
```

### Restore Procedure
```bash
# Restore database
cp /backup/a2a/a2a_data_20240825_120000.db a2a_data.db

# Restore configuration
tar -xzf /backup/a2a/config_20240825_120000.tar.gz

# Restart services
systemctl restart a2a-platform
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
journalctl -u a2a-platform -n 100

# Verify permissions
ls -la /var/lib/a2a/
chown -R a2a:a2a /var/lib/a2a/

# Check port availability
netstat -tulpn | grep 8000
```

#### 2. Database Connection Issues
```bash
# Test database connection
python3 -c "import sqlite3; conn = sqlite3.connect('a2a_data.db'); print('OK')"

# Check database integrity
sqlite3 a2a_data.db "PRAGMA integrity_check;"
```

#### 3. Performance Issues
```bash
# Monitor system resources
htop

# Check application metrics
curl http://localhost:8000/api/metrics

# Analyze slow queries
sqlite3 a2a_data.db "EXPLAIN QUERY PLAN SELECT * FROM agents;"
```

## Health Checks

### Application Health Check
```bash
#!/bin/bash
# health_check.sh

# Check if service is running
if ! systemctl is-active --quiet a2a-platform; then
    echo "ERROR: A2A Platform service is not running"
    exit 1
fi

# Check HTTP endpoint
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/health)
if [ $HTTP_STATUS -ne 200 ]; then
    echo "ERROR: Health endpoint returned $HTTP_STATUS"
    exit 1
fi

# Check database
if ! python3 -c "import sqlite3; sqlite3.connect('a2a_data.db').execute('SELECT 1')"; then
    echo "ERROR: Database connection failed"
    exit 1
fi

echo "OK: All health checks passed"
```

## Scaling Guidelines

### Vertical Scaling
- Increase CPU cores for compute-intensive operations
- Add RAM for caching and concurrent connections
- Use SSD storage for database performance

### Horizontal Scaling
1. Deploy multiple instances behind load balancer
2. Use shared database (PostgreSQL/MySQL)
3. Implement session management (Redis)
4. Configure message queue (RabbitMQ/Kafka)

### Auto-Scaling Rules
```yaml
# AWS Auto Scaling
ScalingPolicy:
  TargetValue: 70.0
  PredefinedMetricType: ASGAverageCPUUtilization
  ScaleUpCooldown: 300
  ScaleDownCooldown: 600
  MinCapacity: 2
  MaxCapacity: 10
```

## Maintenance Windows

### Weekly Maintenance
- Security updates: Sunday 2-4 AM
- Database optimization: Sunday 4-5 AM
- Log rotation: Daily at midnight

### Monthly Maintenance
- Full system backup: First Sunday
- Performance analysis: Second Tuesday
- Security audit: Third Wednesday

## Support Contacts

- **Technical Support**: support@a2aplatform.com
- **Security Issues**: security@a2aplatform.com
- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.a2aplatform.com

---
*Last Updated: 2025-08-25*