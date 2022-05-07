import pandas as pd
import numpy as np


class DataWrangling():
    '''
    Wrangles the raw data from the CSV files
    '''

    data: pd.DataFrame

    def __init__(self, path, is_test=False):
        '''
        Initializes the class and reads in the data from the csv

        Args:
            path (string):      The path at which the data is located
        '''
        assert(path[-4:] == '.csv')
        self.data = pd.read_csv(path)
        self._is_test = is_test
        self._wrangle()

    def head(self, count=5):
        '''
        Prints the first rows of the self.data in its current form

        Args:
            count (int):        The number of rows to print
        '''
        print(self.data.head(count))

    def _wrangle(self):
        '''
        Performs all data wrangling functions on the dataset
        '''
        self._get_titles()
        self._bin_ages()
        self._find_families()
        self._bin_gender()
        self._bin_embarked()
        self._bin_fare()
        self._drop_unnecessary()

    def get_xy(self):
        '''
        Getter for the dataset, returns features and labels for
        training data or just features for test
        '''
        if self._is_test:
            return self.data

        return self.data.drop('Survived', axis=1), self.data['Survived']

    def _get_titles(self):
        '''
        Extracts the titles from the Name column. Groups rare titles and
        different terms for similar or same titles. Maps the final title
        list to integer values
        
        Drops the Name column once complete
        '''
        if 'Name' in self.data.columns.values:
            self.data['Title'] = self.data.Name.str.extract(' ([A-Za-z]+)\.', \
                expand=False)
            
            self.data['Title'] = self.data['Title'].replace([
                'Capt', 
                'Col',
                'Countess', 
                'Don', 
                'Dona',
                'Dr', 
                'Jonkheer', 
                'Lady', 
                'Major',
                'Rev', 
                'Sir'
                ], 'Rare')
            
            self.data['Title'] = self.data['Title'].replace([
                'Mlle', 
                'Ms'
                ], 'Miss')
            
            self.data['Title'] = self.data['Title'].replace([
                'Mme'
                ], 'Mrs')

            self.data['Title'] = self.data['Title'].map({
                'Miss': 1,
                'Mrs': 2,
                'Master': 3,
                'Mr': 4,
                'Rare': 5
            }).astype(int)

            self.data = self.data.drop(['Name'], axis=1)
        else:
            print("Warning: Name column not present to infer titles")

    def _bin_ages(self):
        '''
        Converts the continuous age range into 5 bins split at ages 16, 32,
        48 and 64
        '''
        self.data['Age'] = self.data['Age'].fillna(0)
        self.data.loc[self.data['Age'] <= 16, 'Age'] = 0
        self.data.loc[(self.data['Age'] > 16) & \
            (self.data['Age'] <= 32), 'Age'] = 1

        self.data.loc[(self.data['Age'] > 32) & \
            (self.data['Age'] <= 48), 'Age'] = 2

        self.data.loc[(self.data['Age'] > 48) & \
            (self.data['Age'] <= 64), 'Age'] = 3

        self.data.loc[self.data['Age'] > 64,'Age'] = 4
        self.data['Age'] = self.data['Age'].astype(int)

    def _find_families(self):
        '''
        Checks whether a passenger is accompanied by family using the
        parent or child variable and the sibling or spouse variable. 
        Stores a boolean value in a new column, 0 if unacompanied, 1 if
        with family. Drops the old columns.
        '''
        if 'SibSp' in self.data.columns.values and \
                                        'Parch' in self.data.columns.values:
            self.data['Family'] = 0
            self.data.loc[self.data['Parch'] > 0, 'Family'] = 1 
            self.data.loc[self.data['SibSp'] > 0, 'Family'] = 1 
            self.data = self.data.drop(['SibSp', 'Parch'], axis=1)

    def _bin_gender(self):
        '''
        Converts sex to numerical values
        '''
        if 'Sex' in self.data.columns.values:
            self.data['Sex'] = self.data['Sex'].fillna(
                self.data.Sex.dropna().mode()[0]
            )
            self.data['Sex'] = self.data['Sex'].map({
                'male': 0,
                'female': 1
            }).astype(int)

    def _bin_embarked(self):
        '''
        Fills the missing values for embarked, converts them to 
        numerical values.
        '''
        self.data['Embarked'] = self.data['Embarked'].fillna(
            self.data.Embarked.dropna().mode()[0]
        )
        self.data['Embarked'] = self.data['Embarked'].map({
            'S': 0,
            'C': 1,
            'Q': 2
        }).astype(int)
            
    def _bin_fare(self):
        '''
        Sets the fare values to a band, split into 4 categories.
        ''' 
        self.data.loc[self.data['Fare'] <= 10.5, 'Fare'] = 0
        self.data.loc[(self.data['Fare'] > 10.5) & \
            self.data['Fare'] <= 21.7, 'Fare'] = 1
        self.data.loc[(self.data['Fare'] > 21.7) & \
            self.data['Fare'] <= 39.7, 'Fare'] = 2
        self.data.loc[self.data['Fare'] > 39.7, 'Fare'] = 3
        self.data['Fare'] = self.data['Fare'].astype(int)

    def _drop_unnecessary(self):
        '''
        Drops columns that are otherwise unused from the dataset
        '''
        if 'PassengerId' in self.data.columns.values and not self._is_test:
            self.data = self.data.drop(['PassengerId'], axis=1)
            print('Dropped PassengerId')
        if 'Ticket' in self.data.columns.values:
            self.data = self.data.drop(['Ticket'], axis=1)
            print('Dropped Ticket')
        if 'Cabin' in self.data.columns.values:
            self.data = self.data.drop(['Cabin'], axis=1)
            print('Dropped Cabin')

