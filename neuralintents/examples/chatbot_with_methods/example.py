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