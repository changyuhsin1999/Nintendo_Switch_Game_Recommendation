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
![image](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/assets/93894065/b56e8fb2-b184-4b08-8a0d-300415933e96)


Word2Vec to train the tokenized info
Use cosine similarity to find the similar game based on combination of title, developer, genre, rating
![image](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/assets/93894065/a09ddf25-ef4c-4561-be1a-da4cb0a52fce)


TF-IDF tokenized the genre column and create similar matrix
Recommend game based on genre type
![image](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/assets/93894065/807a90d3-daeb-4d40-b600-ca10b933e2ce)

### Deep Learning

Collabrative filtering with user embedding and item embedding
![image](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/assets/93894065/659aed6c-63c6-4c1b-9ad1-801dc10d5509)





## Stremlit Demo
![Screenshot](https://github.com/changyuhsin1999/Nintendo_Switch_Game_Recommendation/blob/main/image/Screen%20Shot%202023-08-09%20at%208.23.54%20PM.png)
