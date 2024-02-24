#!/bin/bash

# Instructions:
# 1. Save this file in your project root directory
# 2. Navigate to the project root directory in your terminal
# 3a. Run the following command: `bash setup_project.sh`
# 3b. If 3a doesn't work, run the following command: `chmod +x setup_project.sh` then `bash setup_project.sh`

# Define the directory structure you want to create
folders=(
    "data"
    "notebooks"
    "output"
    "reports"
    "src"
    "data/raw"
    "data/clean"
    "data/pulled"
    "data/manual"
)

# Create the folders
for folder in "${folders[@]}"; do
    mkdir -p "$folder"
done

#touch README.md .gitignore .env

echo "Folder structure created successfully!"