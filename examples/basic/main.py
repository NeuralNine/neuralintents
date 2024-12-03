from neuralintents.assistants.base import AdvancedAssistant
from neuralintents.entities.extractors import TrainableExtractor


if __name__ == "__main__":
    def mymethod(quantities=None, products=None):
        print('Got', quantities, products)

    extractor = TrainableExtractor('intents.json', 'cost')
    extractor.train_model(epochs=20)

    # extractor.save_model('myextractor')
    # extractor.load_model('myextractor')

    assistant = AdvancedAssistant('intents.json',
            {'cost': mymethod}, {'cost': extractor})
    assistant.train_model()

    # assistant.save_model('mymodel')
    # assistant.load_model('mymodel')

    print(assistant.process('How much is the price of 5 laptops?'))
    print(assistant.process('Hello'))
    print(assistant.process('What is programming?'))

