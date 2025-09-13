import ollama

def ask_llama(prompt: str):
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error talking to LLaMA3: {e}"
