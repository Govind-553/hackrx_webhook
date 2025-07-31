from flask import Blueprint, request, jsonify
from .processor import process_document_and_answer
import time
import logging
import traceback
from functools import wraps
import asyncio

main = Blueprint('main', __name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(f):
    """Decorator to measure API response times for async functions."""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = await f(*args, **kwargs)
        end_time = time.time()
        
        processing_time = round(end_time - start_time, 4)
        logger.info(f"API {f.__name__} took {processing_time} seconds")

        if isinstance(result, tuple) and len(result) >= 2:
            response_data, status_code = result
            data = response_data.get_json()
            if isinstance(data, dict):
                data['processing_time_seconds'] = processing_time
                return jsonify(data), status_code
        return result
    return decorated_function

def error_handler(f):
    """Decorator for consistent async error handling."""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in {f.__name__}: {error_msg}\n{traceback.format_exc()}")
            
            error_type = "internal_server_error"
            if "fetch" in error_msg.lower() or "network" in error_msg.lower():
                error_type = "network_error"
            elif "processing" in error_msg.lower() or "readable" in error_msg.lower():
                error_type = "document_processing_error"
            
            return jsonify({
                "error": "An error occurred while processing your request.",
                "details": error_msg,
                "error_type": error_type
            }), 500
    return decorated_function

@main.route('/hackrx/run', methods=['POST'])
@timing_decorator
@error_handler
async def hackrx_run():
    """Main API endpoint for processing documents asynchronously."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    if not data or 'documents' not in data or 'questions' not in data:
        return jsonify({"error": "Invalid request body. Missing 'documents' or 'questions'."}), 400
    
    document_url = data.get('documents')
    questions = data.get('questions')
    
    logger.info(f"Received request for document: {document_url}")
    answers = await process_document_and_answer(document_url, questions)
    
    response_data = {
        "answers": answers,
        "metadata": {
            "total_questions": len(questions),
            "document_url": document_url,
            "status": "success"
        }
    }
    
    return jsonify(response_data), 200