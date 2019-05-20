##### Imports   #######
#Azure
from azureml.core import Run

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#arg parsing
import argparse
import os

#pandas & numpy
import pandas as pd
import numpy as np

#save model
from joblib import dump

#####Arg Parsing#######
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', 
                    help='data folder mounting point', default='C:\Daten\OneDrive\OneDrive - Digital Ratio GmbH\Microsoft Training\AzureMLServices'
                   )

parser.add_argument('--max_depth', type=int, dest='max_depth', default=2)
parser.add_argument('--random_state', type=int, dest='random_state', default=42)
parser.add_argument('--n_estimators', type=int, dest='n_estimators', default=100)
args = parser.parse_args()           

##### load Data #######
data_folder = os.path.join(args.data_folder, 'data')
print('Data folder:', data_folder)

data = pd.read_csv(os.path.join(data_folder, "flights.csv"), index_col=0)

##### Test-Train-Split #####
X_train, X_test, y_train, y_test = train_test_split( data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"],
                                                     random_state=42, test_size=0.25)

#### to be changed ####


params = {
          'max_depth':args.max_depth, 
          'random_state':args.random_state,
          'n_estimators':args.n_estimators
}
#### Random Forrest Classifier ####
rf = RandomForestClassifier(**params)
rf.fit(X_train, y_train)


#### Evaluation ####
# get hold of the current run
run = Run.get_context()

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))
auc_ = auc(rf, X_train, X_test)

print('-- AucScore --')
print(auc_)

run.log('train aucc', np.float(auc_[0]))
run.log('test aucc', np.float(auc_[1]))

os.makedirs('outputs', exist_ok=True)
dump(rf, 'outputs/RandomForest.joblib')
