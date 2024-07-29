# BoardGameRecommender
A machine learning tool for predicting how much you will enjoy particular board games based on board games you've previously played. Operates on a downloadable .csv file from your BoardGameGeek.com collection whilst scraping BoardGameGeek.com for data not available in the downloadable csv file.
The web scraper is designed to follow BoardGameGeek's Terms of Service to ensure that web access is performed at the same rate as a human would do so.
Based on my personal board game data, by including mechanics and the core data, I achieved a R^2 (coefficient of determination) of 0.92 showing that the program can be used to accurately predict how you would rate a particular game given a reasonably sized data sample.
