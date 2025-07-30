from flask import Blueprint, request, jsonify
from .processor import process_document_and_answer

main = Blueprint('main', __name__)

@main.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    try:
        data = request.get_json()

        document_url = data.get('documents')
        questions = data.get('questions')

        if not document_url or not questions:
            return jsonify({"error": "Missing 'documents' URL or 'questions'"}), 400

        answers = process_document_and_answer(document_url, questions)
        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route('/webhook', methods=['POST'])
def webhook_listener():
    try:
        payload = request.get_json()

        # Optional: log or debug webhook input
        print("Received webhook payload:", payload)

        # Example: Extract basic info if needed
        event_type = payload.get('event_type', 'unknown')
        message = f"Webhook received successfully. Event Type: {event_type}"

        return jsonify({"status": "success", "message": message}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
