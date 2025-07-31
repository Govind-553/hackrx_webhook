import uvicorn
from app import create_app
from asgiref.wsgi import WsgiToAsgi

# Create the standard Flask app instance
flask_app = create_app()

# This allows the ASGI server (Uvicorn) to communicate correctly with the WSGI app (Flask)
asgi_app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    # Run the wrapped ASGI app with Uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000, log_level="info")