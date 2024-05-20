import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

class Chatbot:
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []
        self.history_json = []

    def get_response(self, user_input):
        # Append user input to history
        self.history.append(f"User: {user_input}")
        
        # Prepare the input with history
        input_text = "\n".join(self.history) + "\nAI:"
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Generate a response
        response_ids = self.model.generate(inputs, max_length=500, pad_token_id=self.tokenizer.eos_token_id)
        response_text = self.tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        
        # Append the AI's response to history
        self.history.append(f"AI: {response_text}")
        
        # Add to JSON history
        self.history_json.append({"User": user_input, "AI": response_text})
        
        return response_text

    def clear_history(self):
        self.history = []
        self.history_json = []

    def get_history_json(self):
        return json.dumps(self.history_json, indent=2)

# Example usage
chatbot = Chatbot(model_name='microsoft/DialoGPT-medium')
print(chatbot.get_response("Hello!"))
print(chatbot.get_response("How are you?"))
if __name__ == "__main__":
    bot = Chatbot(model_name='microsoft/DialoGPT-medium')

    print("Start chatting with the bot (type 'exit' to stop, 'clear' to reset history):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            bot.clear_history()
            print("History cleared.")
        else:
            response = bot.get_response(user_input)
            print(f"AI: {response}")

    # Print the conversation history in JSON format upon exit
    print("\nChat History in JSON format:")
    print(bot.get_history_json())
