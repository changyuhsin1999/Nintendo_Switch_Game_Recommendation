# Nintendo_Switch_Game_Recommendation
Recommendation system for Nintendo Switch games

Nintendo Switch has gained worldwide popularity since its first debut in 2017. It transforms the traditional game console and game chip media and lets users have a more convenient gaming experience. 

However, on the official website, there is no recommendation system that suggests similar games when you purchase a game. In this project, I would like to build a recommendation system to users if they’ve found the current game being interesting and would like to search for similar games

## Data
I found a Kaggle dataset that collects Nintendo Switch games, rating and user info. No one has done a recommendation system based on this dataset. This dataset contains game title, userID, rating, genre, developers
[Kaggle Nintendo Switch Data](https://www.kaggle.com/datasets/mrmorj/nintendo-games-dataset)

## Modeling process
### NON Deep Learning

KNN
Use cosine similarity to find the similar game based on users’ previous rating
Evaluation methods: Euclidean distance


Word2Vec to train the tokenized info
Use cosine similarity to find the similar game based on combination of title, developer, genre, rating


TF-IDF tokenized the genre column and create similar matrix
Recommend game based on genre type


### Deep Learning

Collabrative filtering with user embedding and item embedding



## Stremlit Demo
![Screenshot](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/blob/main/image/Screen%20Shot%202023-08-09%20at%208.23.54%20PM.png)
