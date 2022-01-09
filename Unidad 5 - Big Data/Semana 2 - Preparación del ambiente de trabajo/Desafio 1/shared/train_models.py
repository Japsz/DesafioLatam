import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
import sys
from joblib import dump

"""
python train_models.py filename
"""

file_name = sys.argv[1]
target_name = sys.argv[2]

df = pd.read_csv(file_name)
df['delay_time_bin'] = np.where(df['delay_time'] > df['delay_time'].mean(), 1, 0) # Delay_time recoding
df.drop(columns = ['delay_time'], inplace = True)
#df['deliverer_id'] = df['deliverer_id'].astype('O')
df = pd.get_dummies(df, drop_first = True)

x_train, x_val, y_train, y_val = train_test_split(df.loc[:, df.columns != 'delay_time_bin'], 
                                                df['delay_time_bin'],
                                                test_size = .33, 
                                                random_state = 11238)

logit_clf = LogisticRegression().fit(x_train, y_train)
tree_clf = DecisionTreeClassifier().fit(x_train, y_train)
forest_clf = RandomForestClassifier().fit(x_train, y_train)
gboost_clf = GradientBoostingClassifier().fit(x_train, y_train)
nbayes_clf = BernoulliNB().fit(x_train, y_train)

clf_list = [logit_clf, tree_clf, gboost_clf, forest_clf, nbayes_clf]

file = open('candidate_models.txt', 'w')


best_accuracy= 0
for model in clf_list:
    model_preds = model.predict(x_val)
    print('\n########### {} ###########\n'.format(type(model).__name__))
    clf_report = classification_report(y_val, model_preds)
    print(clf_report)

    report_dict = classification_report(y_val, model_preds, output_dict = True)

    if(report_dict['accuracy'] > best_accuracy):
        best_accuracy = report_dict['accuracy']
        best_model = model

    # write to file
    file.write('\n########### {} ###########\n'.format(type(model).__name__))
    file.write(clf_report)
file.close()

dump(best_model, str(type(best_model).__name__)+'_BestModel.joblib')
