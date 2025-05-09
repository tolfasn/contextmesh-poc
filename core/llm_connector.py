# core/llm_connector.py
# This module will connect to the LLM (Ollama).
import requests
import json

class OllamaConnector:
    def __init__(self, base_url: str = 'http://localhost:11434'):
        """
        Initializes the OllamaConnector.
        :param base_url: Base URL for the Ollama API.
        """
        self.base_url = base_url
        # Test connection on init
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5) 
            response.raise_for_status() 
            print(f"Successfully connected to Ollama at {self.base_url} and API is responsive.")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url} on init or API not responsive. Error: {e}")
            print("Please ensure Ollama is running, accessible, and has models pulled (e.g., 'ollama pull mistral').")


    def query_llm(self, model_name: str, prompt_text: str, stream: bool = False) -> str | None:
        """
        Queries the specified Ollama model with the given prompt.
        Returns the LLM's response text, or None if an error occurs.
        """
        api_url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": stream,
            "options": { 
                "temperature": 0.7, 
                # Further Ollama options (model-dependent, consult Ollama docs):
                # "num_ctx": 2048, # Example context window size
                # "top_k": 40,     # Example sampling params
                # "top_p": 0.9
            }
        }
        
        print(f"\nQuerying Ollama model '{model_name}' (stream={stream})...")
        # print(f"Prompt (first 100 chars): {prompt_text[:100]}...") # Uncomment for debugging prompts

        try:
            with requests.Session() as session:
                response = session.post(api_url, json=payload, stream=stream, timeout=180) 
            
            response.raise_for_status() 

            if stream:
                full_response_text = ""
                print("Streaming response:")
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            chunk = json_line.get("response", "")
                            full_response_text += chunk
                            print(chunk, end='', flush=True) 
                            if json_line.get("done") is True: # Explicitly check for True
                                break 
                        except json.JSONDecodeError:
                            # This can happen if a line is not valid JSON (e.g. empty keep-alive)
                            # For robust streaming, one might need more sophisticated line handling
                            print(f"\nWarning: Could not decode JSON line from stream: {line}")
                            continue 
                print() 
                return full_response_text.strip()
            else:
                response_data = response.json()
                print("Received non-streaming response.")
                return response_data.get("response", "").strip()

        except requests.exceptions.Timeout:
            print(f"Error: Ollama API request timed out after 180 seconds.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama API at {api_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Ollama error detail: {error_detail.get('error')}")
                    if "model not found" in error_detail.get('error', '').lower():
                        print(f"Hint: Make sure model '{model_name}' is pulled via 'ollama pull {model_name}'")
                except json.JSONDecodeError: # If error response itself is not JSON
                    print(f"Ollama error response (not JSON): {e.response.text}")
            return None
        except Exception as e: 
            print(f"An unexpected error occurred while querying LLM: {e}")
            return None

if __name__ == '__main__':
    try:
        print("--- OllamaConnector __main__ Test ---")
        connector = OllamaConnector() 
        
        available_models = []
        try:
            models_response = requests.get(f"{connector.base_url}/api/tags", timeout=5)
            if models_response.status_code == 200:
                # Get unique base model names (e.g., 'mistral' from 'mistral:latest')
                available_models = list(set(
                    m.get('name', 'unknown:unknown').split(':')[0] 
                    for m in models_response.json().get('models', [])
                ))
                print(f"Available Ollama models (base names): {available_models}")
            else:
                print(f"Could not fetch available models from Ollama (status {models_response.status_code}).")
        except Exception as e:
            print(f"Error fetching available models from Ollama: {e}")

        if not available_models:
             print("Warning: No models seem to be available in Ollama or could not fetch list.")
        
        target_model = "mistral" 
        
        if target_model not in available_models:
            print(f"Warning: Target model '{target_model}' not found in available models: {available_models}.")
            if available_models: # If there are other models, try the first one
                first_available = available_models[0]
                print(f"Attempting to use first available model instead: '{first_available}'")
                target_model = first_available
            else:
                print(f"No models available to fallback to. Will attempt query with '{target_model}' anyway.")
        
        prompt1 = "Why is the sky blue? Explain concisely in one or two sentences."
        print(f"\n--- Querying '{target_model}' (non-streaming) for: '{prompt1}' ---")
        response_text1 = connector.query_llm(target_model, prompt1, stream=False)
        if response_text1:
            print(f"\nLLM Response (non-streaming):\n{response_text1}")
        else:
            print(f"\nFailed to get non-streaming response for prompt 1 from '{target_model}'.")

        prompt2 = "Write a very short, 2-line poem about a curious cat."
        print(f"\n--- Querying '{target_model}' (streaming) for: '{prompt2}' ---")
        response_text2 = connector.query_llm(target_model, prompt2, stream=True) 
        if response_text2:
             print(f"\nFull streamed response collected:\n{response_text2}")
        else:
            print(f"\nFailed to get streaming response for prompt 2 from '{target_model}'.")

    except Exception as e: 
        print(f"An error occurred in OllamaConnector __main__ example: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- OllamaConnector __main__ Test Complete ---")
