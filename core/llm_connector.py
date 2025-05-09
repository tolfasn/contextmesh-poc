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
            # Check if the /api/tags endpoint is reachable (lists installed models)
            response = requests.get(f"{self.base_url}/api/tags", timeout=5) # 5-second timeout
            response.raise_for_status() # Will raise an exception for HTTP error codes
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
        # Basic payload structure for Ollama's /api/generate
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": stream,
            "options": { 
                "temperature": 0.7, # A common default, adjust as needed
                # "num_ctx": 2048, # Example context window size, model dependent
                # "top_k": 40,     # Example sampling params
                # "top_p": 0.9
            }
        }

        print(f"\nQuerying Ollama model '{model_name}' (stream={stream})...")
        # For debugging, you might want to see the prompt, but be careful with very long prompts.
        # print(f"Prompt (first 100 chars): {prompt_text[:100]}...")

        try:
            # Using a session for potential keep-alive benefits if making many calls
            with requests.Session() as session:
                response = session.post(api_url, json=payload, stream=stream, timeout=180) # Increased timeout to 3 minutes for potentially long generations

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
                            print(chunk, end='', flush=True) # Print chunk as it arrives
                            if json_line.get("done") and json_line.get("done") is True:
                                break 
                        except json.JSONDecodeError:
                            print(f"\nWarning: Could not decode JSON line from stream: {line}")
                            continue 
                print() # Newline after stream finishes
                return full_response_text.strip()
            else:
                # Non-streaming response
                response_data = response.json()
                print("Received non-streaming response.")
                return response_data.get("response", "").strip()

        except requests.exceptions.Timeout:
            print(f"Error: Ollama API request timed out.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama API at {api_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Ollama error detail: {error_detail.get('error')}")
                    if "model not found" in error_detail.get('error', '').lower():
                        print(f"Hint: Make sure model '{model_name}' is pulled via 'ollama pull {model_name}'")
                except json.JSONDecodeError:
                    print(f"Ollama error response (not JSON): {e.response.text}")
            return None
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred while querying LLM: {e}")
            return None

if __name__ == '__main__':
    try:
        print("--- OllamaConnector __main__ Test ---")
        connector = OllamaConnector() # __init__ will attempt to connect and print status

        # Check available models to provide better feedback if the target model isn't present
        available_models = []
        try:
            models_response = requests.get(f"{connector.base_url}/api/tags", timeout=5)
            if models_response.status_code == 200:
                available_models = [m.get('name', 'unknown:unknown').split(':')[0] for m in models_response.json().get('models', [])]
                print(f"Available Ollama models (base names): {list(set(available_models))}") # Use set for unique base names
            else:
                print(f"Could not fetch available models from Ollama (status {models_response.status_code}).")
        except Exception as e:
            print(f"Error fetching available models from Ollama: {e}")

        if not available_models:
             print("No models seem to be available in Ollama. Please ensure Ollama is running and you have pulled models.")
             # exit() # Don't exit, let user see further attempts/errors

        target_model = "mistral" # Default model for PoC

        if target_model not in available_models and available_models:
            print(f"Warning: Target model '{target_model}' not found in available models: {available_models}.")
            # Fallback to the first available model if target_model is not present
            first_available = available_models[0]
            print(f"Attempting to use first available model instead: '{first_available}'")
            target_model = first_available
        elif not available_models:
            print(f"No models available, but will still attempt to query '{target_model}' in case list was incomplete.")


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

    except Exception as e: # Catch errors from __init__ if Ollama wasn't running initially
        print(f"An error occurred in OllamaConnector __main__ example: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- OllamaConnector __main__ Test Complete ---")