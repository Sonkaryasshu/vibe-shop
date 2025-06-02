from flask import Flask, send_from_directory
from dotenv import load_dotenv
import os
import logging

def create_app():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') 
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment variables from: {dotenv_path}")
    else:
        logger.warning(f".env file not found at {dotenv_path}, relying on system environment variables.")

    frontend_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
    
    app = Flask(__name__, static_folder=frontend_folder)


    from .main import main_bp
    app.register_blueprint(main_bp)

    @app.route('/')
    def serve_index():
        return send_from_directory(frontend_folder, 'index.html')

    @app.route('/<path:filename>')
    def serve_static_files(filename):
        if filename not in ['app.js', 'style.css']:
            pass
        return send_from_directory(frontend_folder, filename)
            
    return app
