import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import graphviz
import numpy as np

winner_data = pd.read_csv("../dados/match_winner_data_version1.csv")
loser_data = pd.read_csv("../dados/match_loser_data_version1.csv")

def clean_data(data):
    #Eliminating rows with NaN and null values
    if data.isnull().values.any():
        data.dropna(subset = ["win", "firstBlood", "firstTower", "firstInhibitor",
        "firstBaron", "firstDragon", "firstRiftHerald"], inplace=True)
    
    return data

# Make sure that files was loaded

if (winner_data.empty or loser_data.empty):
    print("Problem reading files! Please check files path!")
else:
    print("Files read successfully!")

    print("Winner dataset shape: ", winner_data.shape)
    print("Loser dataset shape: ", loser_data.shape)

    #Replace "Win" and "Fail" with True and False
    winner_data["win"].replace({"Win": True}, inplace=True)
    loser_data["win"].replace({"Fail": False}, inplace=True)
    
    #Separate training set
    winner_training_data = winner_data.iloc[0:50000, 2:9]
    loser_training_data = loser_data.iloc[0:50000, 2:9]
    
    #Clean training set
    winner_training_data = clean_data( winner_training_data)
    loser_training_data = clean_data(loser_training_data)
    
    #Separate test set
    winner_test_data = winner_data.iloc[50000:108828, 2:9]
    loser_test_data = loser_data.iloc[50000:108828, 2:9]

    #Clean test set
    winner_test_data = clean_data(winner_test_data)
    loser_test_data = clean_data(loser_test_data)

    loser_training_data['win'] = loser_training_data['win'].astype('bool')
    loser_test_data['win'] = loser_test_data['win'].astype('bool')

    test_data = pd.concat([winner_test_data.iloc[:, 1:7], loser_test_data.iloc[:, 1:7]], ignore_index=True)
    test_set_labels = pd.concat([winner_test_data["win"], loser_test_data["win"]], ignore_index=True)

    #print("Change")
    #print(loser_training_data.dtypes)
    
    #Defining dependent variable
    Y = pd.concat([winner_training_data["win"], loser_training_data["win"]], ignore_index=True)

    #Defining independent variable
    X = pd.concat([winner_training_data.iloc[:, 1:7], loser_training_data.iloc[:, 1:7]], ignore_index=True)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(X, Y)

    new_matches_test = clf.predict(test_data)
    acc = accuracy_score(test_set_labels, new_matches_test)

    print('Accuracy Score: ', acc)

    my_tree = tree.export_graphviz(clf, out_file=None, 
                      feature_names= ["firstBlood", "firstTower", "firstInhibitor",
                                      "firstBaron", "firstDragon", "firstRiftHerald"],  
                      class_names=["Win", "Lose"],  
                      filled=True, rounded=True,  
                      special_characters=True)

    graph = graphviz.Source(my_tree)
    graph.render("ResultsTree")