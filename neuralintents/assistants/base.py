import os
import random
import typing

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neuralintents.models.base import BasicModel
from neuralintents.utils.preprocessing import bag_of_words, bags_of_words, parse_intents, tokenize_and_lemmatize


class BasicAssistant:

    def __init__(self, intents_path: str | os.PathLike) -> None:
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self._load_intents()

    def train_model(self, batch_size: int = 8, lr: float = 0.001, epochs: int = 50) -> None:
        X, y = bags_of_words(self.documents, self.vocabulary, self.intents)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        y_indices = np.argmax(y, axis=1)
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        self.model = BasicModel(X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch}: Loss: {running_loss / len(loader):.4f}')

    def save_model(self, model_path: str | os.PathLike) -> None:
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path: str | os.PathLike) -> None:
        pass

    def save_assistant(self, assistant_path: str | os.PathLike) -> None:
        pass

    def load_assistant(self, assistant_path: str | os.PathLike) -> None:
        pass

    def process(self, input_message: str) -> str:
        words = tokenize_and_lemmatize(input_message)
        bag = bag_of_words(words, self.vocabulary)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        predicted_probability = torch.max(predictions).item()

        return random.choice(self.intents_responses[predicted_intent])

    def _load_intents(self) -> None:
        self.documents, self.vocabulary, self.intents, self.intents_responses = parse_intents(self.intents_path)
        

class AdvancedAssistant(BasicAssistant):
    def __init__(self, intents_path: str | os.PathLike, method_mappings: dict[str, typing.Callable]) -> None:
        super(AdvancedAssistant, self).__init__(intents_path)

        self.method_mappings = method_mappings

    def process(self, input_message: str) -> str:
        words = tokenize_and_lemmatize(input_message)
        bag = bag_of_words(words, self.vocabulary)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        predicted_probability = torch.max(predictions).item()

        if predicted_intent in self.method_mappings:
            self.method_mappings[predicted_intent]()

        return random.choice(self.intents_responses[predicted_intent])

