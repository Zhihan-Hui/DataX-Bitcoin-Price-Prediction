# Bitcoin Price Predictor
Bitcoin price predictor is a dashboard that helps Bitcoin beginners make more effective investment decisions by giving them the prediction tools and credible investment suggestions backed up by machine learning models. It was designed by the Bitcoin Price Prediction team from UC Berkeley's Data-X course.

## Organization and Documentation
### Folder Hierarchy

- Historical Price: Contains code for loading OHLC data, EDA, and indicators.
  - Historical Data Exploration.ipynb: Contains the code we used to download and do initial cleaning on a Kaggle OHLC dataset and a Gemini OHLC dataset.
  - Indicator Implementation-BB.ipynb: Contains the code used for implementing bollinger bands indicator using the ‘ta’ library and based on daily opening and closing historical prices of bitcoin.
  - RSI Indicator.py: Contained the code used for implementing the Relative Strength Indicator using the ‘ta’ library and based on daily opening and closing historical prices of bitcoin.
  - btcstock.py: contains a class that provides easier access to minute-by-minute OHLC Bitcoin data from Bitstamp.
  - price_util.py: Code used for extracting hourly and daily price data from OHLC datasets.

- Model: Contains code for running our prediction model based on both RoBERTa Twitter sentiment data and GRU Price imbedding OHLC bitcoin data. 


## To get started

1. Download the data generated for this project by downloading the folder 'data' from [here](https://drive.google.com/drive/u/0/folders/1ZrI5eUj9HK4FGaz8ITXussTAIITQy9M3). 

2. Download the ipinyou dataset from [UCL website](http://bunwell.cs.ucl.ac.uk/ipinyou.contest.dataset.zip)

3. Use the files in the quickstart folder to have your first results:
- one-day.ipynb processes 1 day of data to tabular format. (NB: you have to update the first line to link to the location where you put ipinyou.contest.dataset/training2nd from the ipinyou dataset)
- quickstart-fairlearn.ipynb: get quicly started with fairlearn.  

## final
In this folder there are 2 files to reproduce the results we have gathered throughout the year

- bids-to-impression.ipynb: results for the bids to impressions phase. 
- impression-to-clicks.ipynb: results for the impressions to clicks phase. 

In this folder there are also the helper functions (in the folder /helpers) that enabled us to create the datasets for the final results as well as preprocess the data. 


