import pyttsx3
engine=pyttsx3.init()
voices=engine.getProperty('voices')
engine.setProperty('voices',voices[0].id)
def say(audiostring):
    print(audiostring)
    engine.say(audiostring)
    engine.runAndWait()
