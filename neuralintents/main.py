from abc import ABCMeta, abstractmethod

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self):
        """ Implemented in child class """

    @abstractmethod
    def request_tag(self, message):
        """ Implemented in child class """

    @abstractmethod
    def get_tag_by_id(self, id):
        """ Implemented in child class """

    @abstractmethod
    def request_method(self, message):
        """ Implemented in child class """

    @abstractmethod
    def request(self, message):
        """ Implemented in child class """


class GenericAssistant(IAssistant):

    def __init__(self, intents, method_list, model_name="assistant_model"):
        self.intents = intents
        self.method_list = method_list
        self.model_name = model_name

        if intents.endswith(".json"):
            self.load_json_intents(intents)

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def train_model(self):
        self.lemmatizer = WordNetLemmatizer()

        words = []
        classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
        words = sorted(list(set(words)))

        classes = sorted(list(set(classes)))

        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

        training = []
        output_empty = [0] * len(classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save(f"{self.model_name}.h5", hist)


    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        p = self._bag_of_words(sentence, words)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag']  == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result

    def request_tag(self, message):
        pass

    def get_tag_by_id(self, id):
        pass

    def request_method(self, message):
        pass

    def request(self, message):
        pass
