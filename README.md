

# NeuralIntents



## Setting Up A Basic Assistant

```python
from neuralintents.assistants import BasicAssistant

assistant = BasicAssistant('intents.json')

assistant.fit_model(epochs=50)
assistant.save_model()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.process_input(message))
```

## Binding Functions To Requests

```python
from neuralintents.assistants import BasicAssistant

stocks = ['AAPL', 'META', 'TSLA', 'NVDA']

def print_stocks():
    print(f'Stocks: {stocks}')

assistant = BasicAssistant('intents.json', method_mappings={
    "stocks": print_stocks,
    "goodbye": lambda: exit(0)
})

assistant.fit_model(epochs=50)
assistant.save_model()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.process_input(message))
```

## Sample `intents.json` File

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up", "Hey", "greetings"],
      "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"],
      "context_set": ""
    },
    {
      "tag": "goodbye",
      "patterns": ["cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "bye", "cao", "see ya"],
      "responses": ["Sad to see you go :(", "Talk to you later", "Goodbye!"],
      "context_set": ""
    },
    {
      "tag": "programming",
      "patterns": ["What is programming?", "What is coding?", "Tell me about programming", "Tell me about coding", "What is software development?"],
      "responses": ["Programming, coding or software development, means writing computer code to automate tasks."],
      "context_set": ""
    },
    {
      "tag": "resource",
      "patterns": ["Where can I learn to code?", "Best way to learn to code", "How can I learn programming", "Good programming resources", "Can you recommend good coding resources?"],
      "responses": ["Check out the NeuralNine YouTube channel and The Python Bible series (7 in 1)."],
      "context_set": ""
    },
    {
      "tag": "stocks",
      "patterns": ["What are my stocks?", "Which stocks do I own?", "Show my stock portfolio"],
      "responses": ["Here are your stocks!"],
      "context_set": ""
    }
  ]
}
```

## Loading a Pre-Trained Model

Once you've trained a model, you can load it to avoid training every time.

```python
from neuralintents import GenericAssistant

assistant = GenericAssistant('intents.json', model_name="test_model")
assistant.load_model()  # Load the previously trained model

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.request(message))
```
