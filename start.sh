#!/bin/bash

echo "Downloading models from Google Drive..."
python download_models.py

echo "Starting Flask API..."
gunicorn wsgi:app
