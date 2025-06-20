#!/bin/bash
# Deploy new 156-feature model to EC2

echo "==========================================="
echo "Deploying new 156-feature model to EC2"
echo "==========================================="

# EC2 details
EC2_HOST="ubuntu@13.212.91.54"
KEY_PATH="~/.ssh/mlbot-key-1749802416.pem"

# First, backup existing model on EC2
echo "1. Creating backup of existing model on EC2..."
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd ~/mlbot
if [ -d "models/v1.0" ]; then
    echo "Backing up existing model..."
    mkdir -p models/backups
    cp -r models/v1.0 models/backups/v1.0_$(date +%Y%m%d_%H%M%S)
    echo "Backup created"
fi
EOF

# Create v2.0 directory on EC2
echo -e "\n2. Creating v2.0 directory on EC2..."
ssh -i $KEY_PATH $EC2_HOST "mkdir -p ~/mlbot/models/v2.0"

# Copy new model files
echo -e "\n3. Copying new model files..."
scp -i $KEY_PATH models/v2.0/* $EC2_HOST:~/mlbot/models/v2.0/

# Verify files
echo -e "\n4. Verifying deployment..."
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd ~/mlbot
echo "Files in models/v2.0:"
ls -la models/v2.0/
echo -e "\nModel metadata:"
cat models/v2.0/metadata.json | head -20
EOF

# Update model configuration if needed
echo -e "\n5. Updating model configuration..."
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd ~/mlbot
# Check if we need to update config to use v2.0
if grep -q "models/v1.0" src/common/config.py; then
    echo "Updating config to use v2.0 model..."
    sed -i 's|models/v1.0|models/v2.0|g' src/common/config.py
    echo "Config updated"
else
    echo "Config already points to correct model path"
fi
EOF

# Restart the trading system
echo -e "\n6. Restarting trading system..."
ssh -i $KEY_PATH $EC2_HOST << 'EOF'
cd ~/mlbot
echo "Stopping current system..."
docker-compose down

echo "Starting with new model..."
docker-compose up -d

echo "Waiting for services to start..."
sleep 10

echo "Checking service status..."
docker-compose ps

echo "Checking logs for errors..."
docker-compose logs --tail=50 | grep -E "(ERROR|Failed|Exception)"
EOF

echo -e "\n==========================================="
echo "Deployment complete!"
echo "==========================================="
echo "To monitor the system:"
echo "ssh -i $KEY_PATH $EC2_HOST"
echo "cd mlbot && docker-compose logs -f"