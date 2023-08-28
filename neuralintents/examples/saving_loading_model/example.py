import os

from neuralintents.assistants import BasicAssistant

assistant = BasicAssistant('intents.json', model_name='my_model')

if os.path.exists('my_model.keras'):
    assistant.load_model()
else:
    assistant.fit_model(epochs=50)
    assistant.save_model()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.process_input(message))