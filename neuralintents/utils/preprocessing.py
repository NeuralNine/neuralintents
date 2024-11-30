import os
import json

import nltk  # TODO: Replace later on with nlp module and also download punkt etc.
import numpy as np
import numpy.typing as npt


def tokenize_and_lemmatize(text: str) -> list[str]:
    lemmatizer = nltk.stem.WordNetLemmatizer()

    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]

    return words


def parse_intents(intents_path: str | os.PathLike, ignore_symbols: list[str] = ['.', ',', ':', '!']) -> tuple[list[tuple[list[str], str]], list[str], list[str]]:
    lemmatizer = nltk.stem.WordNetLemmatizer()

    if os.path.exists(intents_path):
        with open(intents_path, 'r') as file:
            intents_data = json.load(file)

        intents = []
        intents_responses = {}
        vocabulary = []
        documents = []

        for intent in intents_data['intents']:
            if intent['tag'] not in intents:
                intents.append(intent['tag'])
                intents_responses[intent['tag']] = intent['responses']
            
            for pattern in intent['patterns']:
                pattern_words = tokenize_and_lemmatize(pattern)
                vocabulary.extend(pattern_words)
                documents.append((pattern_words, intent['tag']))

        vocabulary = sorted(set(vocabulary))

        return documents, vocabulary, intents, intents_responses
    else:
        raise FileNotFoundError(f'File "{intents_path}" not found.')


def bag_of_words(words: list[str], vocabulary: list[str]) -> list[int]:
    return [1 if word in words else 0 for word in vocabulary]


def bags_of_words(documents: list[tuple[list[str], str]], vocabulary: list[str], intents: list[str]) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    # TODO: Probably split into bag of words and prepare_data (bag of words only for X data, and re-usable for inference)


    # document -> (['what', 'is', 'programming', 'about'], 'programming_question')
    # vocabulary -> ['what', 'how', 'why', 'if', 'is', 'coding', 'programming', 'about', 'hello', 'bye', ...]
    # bag of words -> [1, 0, 0, 0, 1, 0, 1, 1, 0, 0,....] (X data)

    # intents -> ['greeting', 'programming_question', 'video_question', 'stock_info', ...]
    # output -> [0, 1, 0, 0, ...] (Y data) (only one position is equal to 1)
    bags = []
    outputs = []

    for document in documents:
        words = document[0]

        bag = bag_of_words(words, vocabulary)

        output = [0] * len(intents)
        output[intents.index(document[1])] = 1

        bags.append(bag)
        outputs.append(output)

    X = np.array(bags)
    y = np.array(outputs)

    return X, y

