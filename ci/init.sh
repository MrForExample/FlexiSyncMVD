#!/bin/sh

set -e

# Configuration
BUCKET_NAME="sortium"
MINIO_HOST="minio:9000"
ACCESS_KEY="minioadmin"
SECRET_KEY="minioadmin"
EXAMPLES_DIR="/examples"

# Configure mc alias
echo "Configuring mc alias..."
mc alias set local http://$MINIO_HOST $ACCESS_KEY $SECRET_KEY

# Create the bucket if it doesn't exist
if ! mc ls local | grep -q $BUCKET_NAME/; then
  echo "Creating bucket '$BUCKET_NAME'..."
  mc mb local/$BUCKET_NAME
else
  echo "Bucket '$BUCKET_NAME' already exists."
fi

# Set bucket policy to public
echo "Setting bucket policy to public..."
mc anonymous set public local/$BUCKET_NAME

# Upload files
echo "Uploading files from '$EXAMPLES_DIR' to bucket '$BUCKET_NAME'..."
mc cp --recursive $EXAMPLES_DIR/ local/$BUCKET_NAME

echo "Initialization complete."