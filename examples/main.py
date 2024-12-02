from neuralintents.assistants.base import AdvancedAssistant
from neuralintents.entities.extractors import TrainableExtractor


if __name__ == "__main__":
    # TODO: Change labels by reading extra fields from JSON
    def mymethod(quantities=None, products=None):
        print('Got', quantities, products)

    extractor = TrainableExtractor('entity_data.json')
    extractor.train_model(epochs=20)

    # extractor.save_model('myextractor')
    # extractor.load_model('myextractor')

    assistant = AdvancedAssistant('intents.json',
            {'cost': mymethod}, extractor)
    assistant.train_model()

    # assistant.save_model('mymodel')
    # assistant.load_model('mymodel')

    # print(assistant.process('Hello'))
    print(assistant.process_with_entities('How much is the price of 5 laptops?'))

