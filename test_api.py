import requests
import json
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/hackrx/run"
PDF_URL = "https://trec.nist.gov/pubs/trec13/papers/clresearch.qa.novelty.pdf"
QUESTIONS = [
    "What is the main challenge in determining an appropriate set of metadata?",
    "What is the central approach for performing tasks like question answering and summarization?",
    "What are some of the specialized lexical resources included in the Knowledge Management System (KMS)?",
    "How was the KMS initially developed?",
    "What is the primary focus of the TREC QA track?",
    "What kind of analysis is required to 'understand' text?"
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

    print("    Sending request to your upgraded AI model...")
    print(f"   Endpoint: {API_URL}")
    print(f"   Document: {PDF_URL}\n")
    
    start_time = time.time()
    try:
        # --- THE FIX ---
        # Increased timeout to 300 seconds (5 minutes) to give the server enough time to process.
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=300)
        # --- END OF FIX ---
        
        end_time = time.time()
        response.raise_for_status()
        response_data = response.json()
        
        print(f"✅ Success! Response received in {end_time - start_time:.2f} seconds.")
        print("-" * 50)
        
        if 'answers' in response_data:
            for i, (question, answer) in enumerate(zip(QUESTIONS, response_data['answers'])):
                print(f"  Question {i+1}: {question}")
                print(f"  AI Answer: {answer}\n")
        
        if 'processing_time_seconds' in response_data:
            print(f"  Server-side processing time: {response_data['processing_time_seconds']:.4f} seconds")

    except requests.exceptions.HTTPError as e:
        print(f"   HTTP Error: {e.response.status_code}")
        print(f"   Response from server: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"   Connection Error or Timeout: Could not get a response in time.")
        print(f"   Please ensure your server is running and check its console for errors. Details: {e}")
    except json.JSONDecodeError:
        print("    Error: Failed to decode JSON from the response.")
        print(f"   Raw Response: {response.text}")

if __name__ == "__main__":
    test_model_endpoint()