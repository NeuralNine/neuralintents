import yfinance as yf

from neuralintents.assistants.base import AdvancedAssistant
from neuralintents.entities.extractors import TrainableExtractor


def get_stock_price(ticker):
    price = yf.Ticker(ticker).history()['Close'].iloc[-1]
    print(f'The stock price of {ticker} is ${price:.2f}')


if __name__ == "__main__":

    ticker_extractor = TrainableExtractor('intents.json', 'stock_price')
    ticker_extractor.train_model(epochs=50)

    # extractor.save_model('myextractor')
    # extractor.load_model('myextractor')

    assistant = AdvancedAssistant('intents.json',
            {'stock_price': get_stock_price}, {'stock_price': ticker_extractor})
    assistant.train_model(epochs=200)

    # assistant.save_model('mymodel')
    # assistant.load_model('mymodel')

    print(assistant.process('Hello'))
    print(assistant.process('Explain stocks to me!'))
    assistant.process('What is the current price of AAPL?')
    assistant.process('Stock price of MSFT')
    print(assistant.process('Bye'))

