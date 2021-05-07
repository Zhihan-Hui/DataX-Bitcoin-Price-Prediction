# Bitcoin Price Predictor
Bitcoin price predictor is a dashboard that helps Bitcoin beginners make more effective investment decisions by giving them the prediction tools and credible investment suggestions backed up by machine learning models. It was designed by the Bitcoin Price Prediction team from UC Berkeley's Data-X course.
![alt text](https://github.com/Zhihan-Hui/DataX-Bitcoin-Price-Prediction/blob/main/README.file/UI.jpg)

## Organization and Documentation
### Folder Hierarchy

- Historical Price: Contains code for loading OHLC data, EDA, and indicators.
  - Historical Data Exploration.ipynb: Contains the code we used to download and do initial cleaning on a Kaggle OHLC dataset and a Gemini OHLC dataset.
  - Indicator Implementation-BB.ipynb: Contains the code used for implementing bollinger bands indicator using the ‘ta’ library and based on daily opening and closing historical prices of bitcoin.
  - RSI Indicator.py: Contained the code used for implementing the Relative Strength Indicator using the ‘ta’ library and based on daily opening and closing historical prices of bitcoin.
  - btcstock.py: contains a class that provides easier access to minute-by-minute OHLC Bitcoin data from Bitstamp.
  - price_util.py: Code used for extracting hourly and daily price data from OHLC datasets.

- Model: Contains code for running our prediction model based on both RoBERTa Twitter sentiment data and GRU Price imbedding OHLC bitcoin data. 
  - cryptoprice.py: Contains code that allows you to access minute-by-minute open, high, low, close (OHLC) Bitcoin price data from https://www.bitstamp.net.
  - data.py: Contains code that prefetches, samples, and iterates over our full Bitcoin dataset.
  - infer.py: Contains code that makes inferences about future price by calculating daily and total loss as well as using profiler. 
  - main.py: Contains the code to run our full prediction model.
  - model.py: Contains the code that builds our Perceiver that makes predictions based on price imbeddings and sentiment imbeddings. Also contains code for 
  - scrape.py: Contains code that scrapes tweets from twitter and saves the following fields ['id', 'conversation_id', 'date', 'user_id', 'username', 'name', 'tweet', 'replies_count', 'retweets_count', 'likes_count', 'link'] in a dataframe.
  - train.py: Contains code that trains our profiler and predictor using our Bitcoin dataset and twitter sentiments. 
  - tweety.py: contains a class with several methods that allow for fast tweet scraping using twint.
  - util.py: Contains code for the profiler.
 
- Visualization: Contains code for visualizing clustering model.
  - Data X T-SNE.ipynb: Contains code for data visualization of the Twitter sentiment clustering.

## Technical Sophistication and Efficiency
### System Architecture
![alt text](https://github.com/Zhihan-Hui/DataX-Bitcoin-Price-Prediction/blob/main/README.file/architecture.jpg)

### Key Features：
- For bitcoin historical price, we collected the hourly bitcoin price from 2015 to 2021/03/31 and applied log normalization to rescale the raw price. Then we extracted the twitter data with “twint” (Python package). For the tweets we collected since 2015, we randomly sample 32 tweets according to the count of “like” and “retweet” each day with the hashtag "bitcoin". 
- After collecting the data of tweets, we feed those tweets into a RoBERTa model pre-trained with sentiment classification, which transforms the tweets into an embedding of size 768 (mathematical representation of tweets' sentiment as a vector).
- After collecting the data of bitcoin historical price, we feed those price data into GRU model to extract the features of bitcoin historical price. And the reason why we choose to use GRU instead of LSTM is that GRU trains faster.
- The model Perceiver is used to make Bitcoin price predictions. We combine the price imbeddings as well as sentiment embeddings into our feature vector and then use a latent convoy to make an initial guess of the price. Afterwards, the latent convoy would be refined through a cross attention with our feature vectors by focusing on specific features and adjusting forecasting prices. Finally, through the latent transformer, we make the final price prediction.

## About the Team
![alt text](https://github.com/Zhihan-Hui/DataX-Bitcoin-Price-Prediction/blob/main/README.file/group%20member.jpg)



