import os
import random
import typing
import inspect

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neuralintents.models.base import BasicModel
from neuralintents.utils.preprocessing import bag_of_words, bags_of_words, parse_intents, tokenize_and_lemmatize
from neuralintents.entities.extractors import TrainableExtractor


class BasicAssistant:

    def __init__(self, intents_path: str | os.PathLike) -> None:
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self._load_intents()

    def train_model(self, batch_size: int = 8, lr: float = 0.001, epochs: int = 50, verbose: int = 1) -> None:
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

            if verbose > 0:
                print(f'Epoch {epoch}: Loss: {running_loss / len(loader):.4f}')

    def save_model(self, model_path: str | os.PathLike) -> None:
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, model_path: str | os.PathLike) -> None:
        if self.model is None:
            X, y = bags_of_words(self.documents, self.vocabulary, self.intents)
            self.model = BasicModel(X.shape[1], len(self.intents))

        self.model.load_state_dict(torch.load(model_path))

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

    # TODO: Replace with base class for extractors
    def __init__(self, intents_path: str | os.PathLike, method_mappings: dict[str, typing.Callable], entity_extractor: typing.Optional[TrainableExtractor] = None) -> None:
        super(AdvancedAssistant, self).__init__(intents_path)

        self.method_mappings = method_mappings
        self.entity_extractor = entity_extractor

    def process(self, input_message: str) -> str:
        predicted_intent = self._predict_intent(input_message)

        if predicted_intent in self.method_mappings:
            self.method_mappings[predicted_intent]()

        return random.choice(self.intents_responses[predicted_intent])

    def process_with_entities(self, input_message: str) -> str:
        if not self.entity_extractor:
            raise RuntimeError('No entity extractor in assistant.')

        predicted_intent = self._predict_intent(input_message)

        # extract entities (one per label for now, improve later)
        # TODO: There should be a possibility to add multiple extractors, a different one for each intent
        extracted_entities = self.entity_extractor.extract_entities(input_message)
        entities_dict = {}


        for entity_text, entity_label in extracted_entities:
            entities_dict[entity_label] = entity_text  # TODO: change later, mutliple entities overwrite one another

        if predicted_intent in self.method_mappings:
            method_to_call = self.method_mappings[predicted_intent]
            if set(inspect.getargspec(method_to_call)[0]) == set(entities_dict.keys()):
                # TODO: Handle optional parameters
                method_to_call(**entities_dict)
            else:
                method_to_call()
        
        # TODO: process entities as part of response (not just method call)
        # Map labels to placeholders in responses
        # Map labels to kwargs
        return random.choice(self.intents_responses[predicted_intent])

    def _predict_intent(self, input_message: str) -> str:
        words = tokenize_and_lemmatize(input_message)
        bag = bag_of_words(words, self.vocabulary)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]
        predicted_probability = torch.max(predictions).item()

        return predicted_intent

