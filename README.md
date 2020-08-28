# League of Legends match result prediction through Decision Tree

This repository contains the source code and data used to write [this notebook](https://www.kaggle.com/natliacarvalho/league-of-legends-decision-trees).

## What is the objective of this project?

Through the data of ranked matches, we want to train a prediction model and check the accuracy achieved by this model. For this, the data set was divided into two groups: Training data and test data.

## How to predict the results?

For this purpose, I choose the Decision Tree Classifier model and I used the its implementation available on [scikitlearn](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms). I also generated a pdf containing the graphic representation of the decision tree.

If you are interested on understanding about Decision Trees Classifier, I think [this book](http://math.ecnu.edu.cn/~lfzhou/seminar/[Joel_Grus]_Data_Science_from_Scratch_First_Princ.pdf) can be helpful!

## What part of datasets was used?

I used the informations about the firsts: FirstBlood, FirstTower, firstInhibitor, firstBaron, firstDragon and firstRiftHerald as the independent variable (X).

The column win was used as dependent variable (Y).

The first half of the data was used as training data, the other data was used as a test.

## What about the model accuracy?

The accuracy achieved was approximately 84%. Make no mistake, that doesn't mean we have a crystal ball to predict match results.

Whas this helpful for you? Wanna talk about projects? Join us on [Telegram](https://t.me/machinelearning_br) and/or [discord](https://discord.gg/GjKasYU)
