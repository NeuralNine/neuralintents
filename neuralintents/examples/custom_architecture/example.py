import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

from neuralintents.assistants import BasicAssistant


own_layers = [
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5)
]


assistant = BasicAssistant('intents.json', hidden_layers=own_layers)

assistant.fit_model(epochs=50)
assistant.save_model()

assistant.model.summary()

done = False

while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        print(assistant.process_input(message))