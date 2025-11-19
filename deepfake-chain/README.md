# MiniProject

This project is a simple blockchain-based backend built with FastAPI and Python.

## Folder Structure

- `backend/`  
  Contains all backend source code:
  - `app.py` — FastAPI application entry point
  - `blockchain.py` — Blockchain logic
  - `utils.py` — Utility functions
  - `requirement.txt` — Python dependencies

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r backend/requirement.txt
   ```

2. **Run the FastAPI server:**
   ```
   python -m uvicorn backend.app:app --reload
   ```

3. **Access the API docs:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## Notes

- Python 3.12+ recommended.
- All cache and environment files are ignored via `.gitignore`.

##