import ollama

desired_model = "llama3:latest"
question = "What is the capital of France?"

response = ollama.chat(desired_model,messages=[
    {
        'role': 'user',
        'content': question
    },
])

OllamaResponse = response['message']['content']

print(f"Response from {desired_model}: {OllamaResponse}")