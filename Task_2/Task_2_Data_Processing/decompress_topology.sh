#!/bin/bash

# Script to decompress production.top.gz to production.top in all protein directories

# Set the base directory containing protein folders
BASE_DIR="${1:-data/MD/restart}"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist"
    exit 1
fi

# Counter for processed files
count=0

# Find all production.top.gz files and decompress them
echo "Searching for production.top.gz files in $BASE_DIR..."

for gz_file in "$BASE_DIR"/*/production.top.gz; do
    # Check if file exists (handles case where no matches found)
    if [ -f "$gz_file" ]; then
        dir=$(dirname "$gz_file")
        protein=$(basename "$dir")
        
        echo "Processing $protein..."
        
        # Decompress the file
        gunzip -f "$gz_file"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully decompressed: $gz_file"
            ((count++))
        else
            echo "  ✗ Failed to decompress: $gz_file"
        fi
    fi
done

if [ $count -eq 0 ]; then
    echo "No production.top.gz files found in $BASE_DIR"
else
    echo "Done! Decompressed $count file(s)"
fi