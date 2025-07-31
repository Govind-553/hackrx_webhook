import requests
import json
import time

API_URL = "http://127.0.0.1:5000/hackrx/run"

# Using a detailed insurance policy document is a great test for your upgraded model.
PDF_URL = "https://www.sbigeneral.in/portal/static/images/sgi/pdf/Arogya_Plus_Policy.pdf"

# A list of questions based on the submission guidelines to test the model's understanding
QUESTIONS = [
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "Is there a benefit for a preventive health check-up?",
    "What is the extent of coverage for dental treatment?",
    "Are there any sub-limits on room rent for a Single Private A/C Room?"
]

# --- API Test Function ---
def test_model_endpoint():
    """
    Sends a request to the API with a test document and questions,
    then prints the results.
    """
    payload = {
        "documents": PDF_URL,
        "questions": QUESTIONS
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    print("Sending request to your upgraded AI model...")
    print(f"   Endpoint: {API_URL}")
    print(f"   Document: {PDF_URL}\n")
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=90) # Increased timeout for first run
        end_time = time.time()
        
        # Check for HTTP errors
        response.raise_for_status()
        
        response_data = response.json()
        
        print(f"✅ Success! Response received in {end_time - start_time:.2f} seconds.")
        print("-" * 50)
        
        # Nicely print the questions and the AI-generated answers
        if 'answers' in response_data:
            for i, (question, answer) in enumerate(zip(QUESTIONS, response_data['answers'])):
                print(f"Question {i+1}: {question}")
                print(f"AI Answer: {answer}\n")
        
        # Print metadata if available
        if 'processing_time_seconds' in response_data:
            print(f"Server-side processing time: {response_data['processing_time_seconds']:.4f} seconds")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"   Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: Could not connect to the API at {API_URL}.")
        print(f"   Please ensure your server is running. Details: {e}")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response. The API might have returned an HTML error page.")
        print(f"   Raw Response: {response.text}")

if __name__ == "__main__":
    test_model_endpoint()