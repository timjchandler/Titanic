from typing_extensions import assert_type
import pandas as pd
import numpy as np


class DataWrangling():
    '''
    Wrangles the raw data from the CSV files
    '''

    data: pd.DataFrame

    def __init__(self, path):
        '''
        Initializes the class and reads in the data from the csv

        Args:
            path (string):      The path at which the data is located
        '''
        assert(path[-4:] == '.csv')
        self.data = pd.read_csv(path)

    def head(self, count=5):
        '''
        Prints the first rows of the dataset in its current form

        Args:
            count (int):        The number of rows to print
        '''
        print(self.data.head(count))

    def GetTitles(self):
        

