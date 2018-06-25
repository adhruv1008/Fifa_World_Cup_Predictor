from sklearn import preprocessing
import pandas as pd

#loading the dataset
matches_path = r'WorldCupMatches.csv'

matches = pd.read_csv(matches_path)
matches = matches[:852]
#print(matches.shape)

#selecting the features to be dropped (since they don't play important role in predicting match result)
columns_to_drop = ['Datetime','City','Win conditions','Attendance','RoundID','MatchID','Home Team Initials','Away Team Initials', 'Half-time Home Goals','Half-time Away Goals','Assistant 1','Assistant 2','Referee']
matches = matches.drop(columns_to_drop,axis=1)

matches['Home Team Winner'] = matches['Home Team Goals'] > matches['Away Team Goals']
matches['Away Team Winner'] = matches['Home Team Goals'] < matches['Away Team Goals']

matches = matches.drop(['Home Team Goals','Away Team Goals'],axis=1)

y = matches.drop(['Year', 'Stage', 'Stadium', 'Home Team Name', 'Away Team Name'],axis=1)

features = matches.drop(['Home Team Winner','Away Team Winner'],axis =1)


for i in range(len(features)):
    #print(features['Stage'][i].split(" ")[0])
    if features['Stage'][i].split(' ')[0] == "Group":
        features['Stage'][i] = "Group"
#print(features['Stage'].head())

#encoding the string data into numeric
features = pd.get_dummies(features)

#standardizing the data
features = preprocessing.scale(features)

#print(features.shape)

from keras.models import Sequential
from keras.layers import Dense
n_cols = features.shape[1]

#Building the neural network
model = Sequential()
model.add(Dense(60,activation = 'relu' , input_shape=(n_cols,)))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.fit(features,y,validation_split=0.3,epochs=1)
