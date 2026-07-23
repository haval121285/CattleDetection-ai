#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3.11 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
