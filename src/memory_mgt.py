#pip install ollama


import ollama

# Configure client pointing to your home lab server
client = ollama.Client(host='http://localhost:11434')

# Initialize the conversation history (system prompt guides the agent)
messages = [
    {
        'role': 'system', 
        'content': 'You are a helpful AI Agent running on a local home lab server.'
    },
    {
        'role': 'user', 
        'content': 'Hello! What model are you and what hardware are we running on?'
    }
]

# Send request to your hardware-optimized Qwen model with streaming enabled
response = client.chat(
    model='qwen2.5:3b',
    messages=messages,
    stream=True
)

# Stream chunks to the terminal as your Xeon/1050 Ti processes them
print("Agent Response: ", end="", flush=True)
for chunk in response:
    print(chunk['message']['content'], end="", flush=True)
print()
