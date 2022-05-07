import re
import os
import sys
import pandas as pd
from wrangling import DataWrangling as dw
from training import RandomForest, SupportVectorMachine

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        run_rf()
    elif args[0] == '-p':
        if args[1] == 'svm':
           run_svm()
        if args[1] == 'rf':
            run_rf()
        else:
            print("Unknown predictor. Please choose one of the following:\n\tsvm\t\tSupport Vector Machine\n\trf\t\tRandom Forest Classifier") 
    else:
        print("Use the flag -p to select predictor")            

def load_data(path='data/'):
    '''
    Loads in the data using the DataWrangler class to split it into
    training features, training labels and test features.

    Args:
        path (str):         The path to the directory holding the data
    '''
    train = dw(path + 'train.csv')
    test = dw(path + 'test.csv', is_test=True)
    X, yy = train.get_xy()
    X_test = test.get_xy()
    return X, yy, X_test

def run_rf():
    X, yy, X_test = load_data()
    rf = RandomForest(X, yy)
    rf.train()
    filename = check_submissions('rf')
    rf.save_submission(X_test, save_path='submissions/' + filename + '.csv')  

def run_svm():
    raise(NotImplementedError)

def check_submissions(name):
    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    
    while True:
        if os.path.exists('submissions/' + name + '.csv'):
            suffix = 1
            if re.search('[0-9]', name) != None:
                suffix = int(''.join(re.findall('[0-9]', name))) + 1
            name = re.sub('[0-9]|_', '', name) + '_' + str(suffix)
        else:
            break
    
    return name

if __name__=='__main__':
    main()