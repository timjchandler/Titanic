from abc import abstractmethod
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier        

class Predictor():
    '''
    Parent class for the predictors.
    Loads in data and provides an abstract method for
    training.
    '''
    def __init__(self, X, yy):
        '''
        Initializes the class and loads data

        Args:
            X (pd.DataFrame)    The data features
            yy (pd.DataFrame)   The data labels
        '''
        self._X = X 
        self._yy = yy

    def train(self):
        self._model.fit(self._X, self._yy)
        print(f'Score: {round(self._model.score(self._X, self._yy), 3)}')

    def save_submission(self, X_test, save_path="submission.csv"):
        predictions = self._model.predict(X_test.drop(['PassengerId'], axis=1))
        submission = pd.DataFrame({
            'PassengerId': X_test['PassengerId'],
            'Survived': predictions
        })
        submission.to_csv(save_path, index=False)


class SupportVectorMachine(Predictor):
    '''
    Support Vector Machine, inherits from predictor class to load 
    in X and yy data.
    '''
    def train(self):
        '''
        Trains the SVM and provides a score based on the training
        data used.
        '''
        self._model = SVC()
        super().train()

class RandomForest(Predictor):
    '''
    Random Forest, inherits from predictor class to load in X and
    yy data
    '''
    def train(self):
        '''
        Trains the Random Forest and provides a score based on the
        training data used.
        '''
        self._model = RandomForestClassifier(n_estimators=100)
        super().train()
