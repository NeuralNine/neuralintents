import os
import json
import random
import itertools
from typing import Literal

import spacy
from spacy.util import minibatch
from spacy.training.example import Example


class BaseExtractor:
    pass  # TODO: Later -> Inheritance

class BasicExtractor:
    pass  # Not sure what this will be

class LocationExtractor:
    pass  # and more like this... (pre-trained)


class TrainableExtractor:

    def __init__(self, train_data_path: str | os.PathLike, model_type: ['sm', 'md', 'lg', 'trf'] = 'sm') -> None:
        # TODO: python -m spacy download en_core_web_lg  (handle this)
        if model_type not in ['sm', 'md', 'lg', 'trf']:
            raise ValueError('Model type needs to be part of ["sm", "md", "lg", "trf"]')

        self.model = spacy.load(f'en_core_web_{model_type}')
        self._train_data = None

        self._load_train_data(train_data_path)

    def train_model(self, epochs: int = 50, batch_size: int = 8) -> None:
        with self.model.disable_pipes([pipe for pipe in self.model.pipe_names if pipe != 'ner']):
            optimizer = self.model.begin_training()

            for i in range(epochs):
                random.shuffle(self._train_data)
                losses = {}

                batches = minibatch(self._train_data, size=batch_size)
                for batch in batches:
                    examples = []
                    for text, batch_entities in batch:
                        doc = self.model.make_doc(text)
                        example = Example.from_dict(doc, {'entities': batch_entities})
                        examples.append(example)

                    self.model.update(examples, drop=0.5, sgd=optimizer, losses=losses)

                print(f'Epoch {i+1}, Losses: {losses}')

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        document = self.model(text)
        return [(entity.text, entity.label_) for entity in document.ents]

    def _load_train_data(self, train_data_path: str | os.PathLike) -> None:
        with open(train_data_path, 'r') as file:
            train_data = json.load(file)

        templates = train_data['templates']
        placeholders = train_data['placeholders']

        placeholder_values = {placeholder: train_data[category] for placeholder, category in placeholders.items()}

        combinations = itertools.product(*placeholder_values.values())

        sentences = []
        entities = []

        for template in templates:
            for combination in combinations:
                sentence = template
                entity_list = []
                for placeholder, value in zip(placeholder_values.keys(), combination):
                    label = placeholders[placeholder]
                    placeholder_position = sentence.index(placeholder)
                    sentence = sentence[:placeholder_position] + value + sentence[placeholder_position+1:]
                    entity_list.append((placeholder_position, placeholder_position+len(value), label))
                sentences.append(sentence)
                entities.append(entity_list)

        if 'ner' not in self.model.pipe_names:
            ner = self.model.add_pipe('ner', last=True)
        else:
            ner = self.model.get_pipe('ner')

        for sentence_entities in entities:
            for entity in sentence_entities:
                ner.add_label(entity[2])

        self._train_data = list(zip(sentences, entities))

