from typing import Text
from pyttsx3 import init

engine = init()
voices = engine.getProperty('voices')
engine.setProperty('voices',voices[0].id)

class VoiceAssistant:
    def __init__(self, audiostring: Text):
        self.audiostring = audiostring

    def say(self):
        print(self.audiostring)
        engine.say(self.audiostring)
        engine.runAndWait()
