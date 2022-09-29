import os
import nltk
import voice
import random
import pickle
import numpy as np
from json import load
from gtts import gTTS, langs
from abc import ABCMeta, abstractmethod
from nltk.stem import WordNetLemmatizer
from .voice_assistant import VoiceAssistant
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Dropout
from typing import Union, List, Dict, Text, Optional, Any
from tensorflow.keras.models import Sequential, load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self) -> Optional[Any]:
        """ Implemented in child class """

    @abstractmethod
    def request_tag(self, message: Text) -> Optional[Any]:
        """ Implemented in child class """

    @abstractmethod
    def get_tag_by_id(self, id: Text) -> Optional[Any]:
        """ Implemented in child class """

    @abstractmethod
    def request_method(self, message: Text) -> Optional[Any]:
        """ Implemented in child class """

    @abstractmethod
    def request(self, message: Text) -> Optional[Any]:
        """ Implemented in child class """


class GenericAssistant(IAssistant):
    def __init__(self,
                 intents: Text,
                 intent_methods: Dict = {},
                 model_name: Text = "assistant_model",
                 voice_assistant: bool = False,
                 voice_saving: bool = False,
                 language: Text = "english",
                 encoding: Text = "utf-8") -> None:
        """

        :param intents: str, the intents file name
        :param intent_methods: Dict, the methods of intents file
        :param model_name: str, the name for your model
        :param voice_assistant: bool, True if you want to play the answer
        :param voice_saving: bool, True if you want to save the answer voice to file
        :param language: str, the language for your NLTP
        :param encoding: str, the model file encoding
        """
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        self.language = language
        self.encoding = encoding
        self.voice_assistant = voice_assistant
        self.voice_saving = voice_saving

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents: Text) -> None:
        with open(intents, encoding=self.encoding) as intents_file:
            self.intents = load(intents_file)

    def train_model(self) -> None:
        """

        Trains the model for NLTP
        :return: None
        """
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern, language=self.language)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def save_model(self, model_name: Text = None) -> None:
        """

        Saves trained model
        :param model_name: Optional (str), the new model name for saving
        :return: None
        """
        self.model.save(f"{model_name if model_name is not None else self.model_name}.h5", self.hist)
        with open(f'{model_name if model_name is not None else self.model_name}_words.pkl', 'wb') as file:
            pickle.dump(self.words, file)
        with open(f'{model_name if model_name is not None else self.model_name}_classes.pkl', 'wb') as file:
            pickle.dump(self.classes, file)

    def load_model(self, model_name: Text = None) -> None:
        """

        Loads the existing model
        :param model_name: Optional (str), the name of your model
        :return: None
        """
        with open(f'{model_name if model_name is not None else self.model_name}_words.pkl', 'rb') as file:
            self.words = pickle.load(file)
        with open(f'{model_name if model_name is not None else self.model_name}_classes.pkl', 'rb') as file:
            self.classes = pickle.load(file)
        self.model = load_model(f'{self.model_name}.h5')

    def _clean_up_sentence(self, sentence: Text) -> List[Text]:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence: Text, words: Text):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence: Text) -> List[Dict[Text, Union[Text, List[Text]]]]:
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': Text(r[1])})
        return return_list

    def _get_response(self, ints: list, intents_json: Dict) -> Text:
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

    def request_tag(self, message: Text) -> None:
        pass

    def get_tag_by_id(self, id: Text) -> None:
        pass

    def request_method(self, message: Text) -> None:
        pass

    def request(self, message: Text) -> Text:
        """

        Request for the result of your message
        :param message: str, the message to the NLTP
        :return: str, answer of NLTP
        """
        ints = self._predict_class(message)

        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            response = self._get_response(ints, self.intents)
            if self.voice_saving:
                tts = gTTS(text=response,
                           lang=list(langs._langs.keys())[list(langs._langs.values()).index(self.language.capitalize())])
                tts.save(f"{self.model_name}_answer.mp3")
            if self.voice_assistant:
                VoiceAssistant(response).say()
            else:
                return response
