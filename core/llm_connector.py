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
            requests.get(f"{self.base_url}/api/tags", timeout=3).raise_for_status()
            print(f"Successfully connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Ollama at {self.base_url} on init. Error: {e}")
            print("Ensure Ollama is running and accessible.")


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
            "options": { # Add some common options
                "temperature": 0.7, # Adjust for creativity vs. factuality
                # "num_predict": 1024 # Max tokens to generate, adjust as needed
            }
        }

        print(f"\nQuerying Ollama model '{model_name}'...")
        # print(f"Prompt (first 100 chars): {prompt_text[:100]}...")

        try:
            response = requests.post(api_url, json=payload, timeout=120) # 120-second timeout
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            if stream:
                full_response_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            chunk = json_line.get("response", "")
                            full_response_text += chunk
                            # Optionally print stream chunks:
                            # print(chunk, end='', flush=True) 
                            if json_line.get("done"):
                                break
                        except json.JSONDecodeError:
                            print(f"\nWarning: Could not decode JSON line from stream: {line}")
                            continue # Skip malformed lines
                # print() # Newline after stream
                return full_response_text.strip()
            else:
                # Non-streaming response
                response_data = response.json()
                return response_data.get("response", "").strip()

        except requests.exceptions.Timeout:
            print(f"Error: Ollama API request timed out after 120 seconds.")
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
        except json.JSONDecodeError as e: # Should be less likely with stream handling
            print(f"Error decoding JSON response from Ollama: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while querying LLM: {e}")
            return None

if __name__ == '__main__':
    # Example usage:
    try:
        print("OllamaConnector __main__ example:")
        connector = OllamaConnector()

        # First, check available models to ensure 'mistral' (or your target) exists
        try:
            models_response = requests.get(f"{connector.base_url}/api/tags")
            models_response.raise_for_status()
            available_models = [m['name'] for m in models_response.json().get('models', [])]
            print(f"Available Ollama models: {available_models}")
            if not available_models:
                 print("No models found in Ollama. Please pull a model, e.g., 'ollama pull mistral'")
                 exit()
        except Exception as e:
            print(f"Could not fetch available models from Ollama: {e}")
            print("Ensure Ollama is running.")
            exit()

        target_model = "mistral" # Default model for PoC
        # target_model = "nous-hermes2" # Another example if you have it

        if target_model not in available_models:
            print(f"Model '{target_model}' not found in available models: {available_models}")
            print(f"Please pull it first, e.g., 'ollama pull {target_model}' or choose an available one.")
            # Fallback to the first available model if target_model is not present
            if available_models:
                target_model = available_models[0].split(':')[0] # Use base name
                print(f"Using fallback model: {target_model}")
            else:
                exit()

        prompt1 = "Why is the sky blue? Explain concisely."
        print(f"\n--- Querying '{target_model}' (non-streaming) ---")
        response_text1 = connector.query_llm(target_model, prompt1, stream=False)
        if response_text1:
            print(f"\nLLM Response (non-streaming):\n{response_text1}")
        else:
            print("\nFailed to get non-streaming response.")

        prompt2 = "Write a short poem about coding."
        print(f"\n--- Querying '{target_model}' (streaming) ---")
        print(f"LLM Response (streaming):")
        response_text2 = connector.query_llm(target_model, prompt2, stream=True) # Streamed output prints within method if enabled
        if response_text2: # This will be the full concatenated response
             print(f"\nFull streamed response collected:\n{response_text2}")
        else:
            print("\nFailed to get streaming response.")

    except Exception as e:
        print(f"Error in OllamaConnector example: {e}")
        import traceback
        traceback.print_exc()