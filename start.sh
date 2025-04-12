#!/bin/bash
python3.11 -m pip install -r requirements.txt
python download_models.py
python3.11 api.py


