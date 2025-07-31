import uvicorn
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Uvicorn to run the app for high-performance async operations
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")