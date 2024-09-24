import openai

# Configure the client
openai.api_key = "dummy"  # The API key doesn't matter for local deployments
openai.api_base = "http://localhost:8000/v1"  # Your local API endpoint

MODEL = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8"

def chat_with_api(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        full_response = ""
        for chunk in response:
            content = chunk['choices'][0]['delta'].get('content', '')
            if content:
                print(content, end='', flush=True)
                full_response += content

        print("\n")  # New line after the complete response
        return full_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        chat_with_api(user_input)

if __name__ == "__main__":
    main()
