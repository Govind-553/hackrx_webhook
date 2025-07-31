from flask import Blueprint, request, jsonify, g
from .processor import process_document_and_answer, health_check
from .utils import validate_input, preprocess_question
import time
import logging
from functools import wraps
import traceback

main = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(f):
    """Decorator to measure API response times"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Add timing info to response if it's a tuple
        if isinstance(result, tuple) and len(result) >= 2:
            response_data, status_code = result[0], result[1]
            if isinstance(response_data, dict) or hasattr(response_data, 'get_json'):
                try:
                    if hasattr(response_data, 'get_json'):
                        data = response_data.get_json()
                    else:
                        data = response_data
                    
                    if isinstance(data, dict):
                        data['processing_time_seconds'] = round(end_time - start_time, 3)
                        logger.info(f"API {f.__name__} took {data['processing_time_seconds']} seconds")
                        return jsonify(data), status_code
                except:
                    pass
        
        logger.info(f"API {f.__name__} took {round(end_time - start_time, 3)} seconds")
        return result
    
    return decorated_function

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in {f.__name__}: {error_msg}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return appropriate error response
            if "fetch" in error_msg.lower() or "network" in error_msg.lower():
                return jsonify({
                    "error": "Failed to fetch document. Please check the URL and try again.",
                    "details": error_msg,
                    "error_type": "network_error"
                }), 400
            elif "processing" in error_msg.lower() or "text" in error_msg.lower():
                return jsonify({
                    "error": "Failed to process document. The document may be corrupted or unreadable.",
                    "details": error_msg,
                    "error_type": "processing_error"
                }), 422
            else:
                return jsonify({
                    "error": "An internal error occurred while processing your request.",
                    "details": error_msg,
                    "error_type": "internal_error"
                }), 500
    
    return decorated_function

@main.route('/hackrx/run', methods=['POST'])
@timing_decorator
@error_handler
def hackrx_run():
    """Enhanced main API endpoint with validation and optimization"""
    
    # Validate Content-Type
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json",
            "error_type": "validation_error"
        }), 400
    
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({
            "error": "Invalid JSON format",
            "details": str(e),
            "error_type": "validation_error"
        }), 400
    
    if not data:
        return jsonify({
            "error": "Empty request body",
            "error_type": "validation_error"
        }), 400
    
    # Validate input structure
    is_valid, validation_message = validate_input(data)
    if not is_valid:
        return jsonify({
            "error": validation_message,
            "error_type": "validation_error"
        }), 400
    
    document_url = data.get('documents')
    questions = data.get('questions')
    
    # Preprocess questions for better matching
    processed_questions = []
    for i, question in enumerate(questions):
        try:
            processed_q = preprocess_question(question)
            if processed_q:
                processed_questions.append(processed_q)
            else:
                processed_questions.append(question.strip())
        except Exception as e:
            logger.warning(f"Error preprocessing question {i+1}: {str(e)}")
            processed_questions.append(question.strip())
    
    logger.info(f"Processing document with {len(processed_questions)} questions")
    logger.info(f"Document URL: {document_url}")
    
    # Process document and get answers
    answers = process_document_and_answer(document_url, processed_questions)
    
    # Validate answers
    if not answers or len(answers) != len(processed_questions):
        logger.warning(f"Answer count mismatch: got {len(answers) if answers else 0}, expected {len(processed_questions)}")
        # Pad with default answers if needed
        while len(answers) < len(processed_questions):
            answers.append("Unable to find a relevant answer in the document.")
    
    # Create detailed response
    response_data = {
        "answers": answers,
        "metadata": {
            "total_questions": len(processed_questions),
            "document_url": document_url,
            "status": "success"
        }
    }
    
    logger.info(f"Successfully processed {len(processed_questions)} questions")
    return jsonify(response_data), 200

@main.route('/hackrx/batch', methods=['POST'])
@timing_decorator
@error_handler
def hackrx_batch():
    """Batch processing endpoint for multiple documents"""
    
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json",
            "error_type": "validation_error"
        }), 400
    
    data = request.get_json()
    if not data:
        return jsonify({
            "error": "Empty request body",
            "error_type": "validation_error"
        }), 400
    
    batch_requests = data.get('batch', [])
    if not isinstance(batch_requests, list) or len(batch_requests) == 0:
        return jsonify({
            "error": "Batch field must be a non-empty list",
            "error_type": "validation_error"
        }), 400
    
    if len(batch_requests) > 10:  # Reasonable limit
        return jsonify({
            "error": "Too many batch requests (maximum 10 allowed)",
            "error_type": "validation_error"
        }), 400
    
    results = []
    for i, batch_item in enumerate(batch_requests):
        try:
            # Validate each batch item
            is_valid, validation_message = validate_input(batch_item)
            if not is_valid:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": validation_message,
                    "error_type": "validation_error"
                })
                continue
            
            # Process the batch item
            document_url = batch_item.get('documents')
            questions = batch_item.get('questions')
            
            answers = process_document_and_answer(document_url, questions)
            results.append({
                "index": i,
                "status": "success",
                "answers": answers
            })
        except Exception as e:
            logger.error(f"Error processing batch item {i}: {str(e)}")
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
                "error_type": "processing_error"
            })