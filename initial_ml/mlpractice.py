
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
#from six.moves import urllib
from tensorflow import feature_column as fc
import tensorflow as tf



dftrain=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  #we read the data file
dfeval=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  #we read the data file
y_train = dftrain.pop('survived') #we removed the 'survived' column
y_eval = dfeval.pop('survived')  #we removed the survived column
#print(dftrain.head())
#print(dftrain.shape)

dftrain.age.hist(bins=20)

CATEGORICAL_COLUMN=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
NUMERICAL_COLUMN=['age','fare']
feature_columns=[]

for feature_name in CATEGORICAL_COLUMN:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))


for feature_name in NUMERICAL_COLUMN:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
#print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): #This is the inner function that will returned
        ds= tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #This creates object of the data and its label
        if shuffle:
            ds=ds.shuffle(1000) #randomize the order of data
        ds=ds.batch(batch_size).repeat(num_epochs) #This seperates the datasets into the number of batches and then runs it with the number of epochs
        return ds #return a batch of the dataset
    return input_function #returns the input function
train_input_fn=make_input_fn(dftrain, y_train)
eval_input_fn=make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)  #creating the model

linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on tetsing data

clear_output()  # clears consoke output
print(result['accuracy'])  # the result variable is simply a dict of stats about our model


pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')
