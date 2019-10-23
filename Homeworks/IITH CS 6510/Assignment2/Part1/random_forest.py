# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:13:38 2018

@author: Nageswara Rao Gurram
Student ID: CS17EMDS11013
"""

import decision_tree as dt

import numpy as np
import pandas as pd
import scipy.stats as sp
import sys, getopt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import time
import math

INPUT_FILE_NAME = "spam.data"

'''
Feature selection set param (Supported : sqrt, log2 and fraction b/w 0 and 1)
'''
dt.FEATURE_SELECTION_SET = "sqrt"

'''
Number of trees in random forest
'''
NUMBER_OF_TREES = 10


class RandomForest():
  '''
  Build a random forest 
  '''
  
  def __init__(self):
    self.n_trees = NUMBER_OF_TREES
    self.trees = [] # create decision trees
    
    for i in range(NUMBER_OF_TREES):
      self.trees.append(dt.DecisionTree())
    
  def build(self, train_samples, train_classes):
    '''
    Build random forest using training samples and classes
    '''
    total_training_samples = train_samples.shape[0]
    total_features = train_samples.shape[1]
    train_classes = np.reshape(train_classes, (total_training_samples, 1))
    training_set = np.hstack((train_samples, train_classes))
    
    for i in range(self.n_trees):
      # create a training set by selecting random samples with replacement
      selected_samples_idx = np.random.randint(total_training_samples, size=total_training_samples)
      rand_train_set = training_set[selected_samples_idx, :]
      self.trees[i].learn(rand_train_set)
    
  def classify(self, test_sample):
    '''
    Classify a test sample by taking the mode of the classes
    '''
    predicted_classes = np.empty((self.n_trees,), dtype=int)
    for i in range(self.n_trees):
      predicted_classes[i] = self.trees[i].classify(test_sample)
    
    return sp.mode(predicted_classes).mode[0]

def printParams(p_trees, p_features, total_features):
    if p_features == "sqrt":
      print("\n NUMBER_OF_TREES : " + str(p_trees) + " FEATURE_SELECTION_SET: " + str(round(math.sqrt(total_features))))
    elif p_features == "log2":
      print("\n NUMBER_OF_TREES : " + str(p_trees) + " FEATURE_SELECTION_SET: " + str(round(math.log2(total_features))))
    elif float(p_features) > 0 and float(p_features) <= 1:
      print("\n NUMBER_OF_TREES : " + str(p_trees) + " FEATURE_SELECTION_SET: " + str(round(total_features*float(p_features))))
    else:
      return total_features

def run_random_forest():
  # read input data
  file_data = pd.read_csv(INPUT_FILE_NAME, header=None, delimiter=r"\s+")
  
  # Split input into train and test data
  (train_samples, test_samples, train_classes, test_classes) = train_test_split(file_data.iloc[:, :-1].values, file_data.iloc[:,-1].values, test_size=0.33, stratify=file_data.iloc[:,-1].values)
  
  printParams(NUMBER_OF_TREES, dt.FEATURE_SELECTION_SET, train_samples.shape[1])
  # run random forest algorithm on custom implementation
  c_start = time.time()
  rd = RandomForest()
  rd.build(train_samples, train_classes)
  
  pred_classes = np.zeros(test_classes.shape, dtype=int)
  for i in range(test_samples.shape[0]):
    pred_classes[i] = rd.classify(test_samples[i])
  
  # Calculateout of bag error

  print("\n Accuracy with custom implementation %.4f" % accuracy_score(test_classes, pred_classes))
  print("\n Time taken for custom implementation %.2f secs "  % (time.time() - c_start))

  l_start = time.time()
  clf = RandomForestClassifier(n_estimators=NUMBER_OF_TREES, random_state=0, max_features=dt.FEATURE_SELECTION_SET)
  clf.fit(train_samples, train_classes)
  clf_pred = clf.predict(test_samples)
  
  print("\n\n Accuracy with library implementation %.4f" %  accuracy_score(test_classes, clf_pred))
  print("\n Time taken for library implementation %.2f secs "  % (time.time() - l_start))

  #plot_oob_test_errors(train_samples, test_samples, train_classes, test_classes)
  
def plot_oob_test_errors(train_samples, test_samples, train_classes, test_classes):
  total_features = train_samples.shape[1]
  feature_set_fractions = np.arange(1, total_features, 5) # total features to consider at each split
  oob_errors = []
  test_errors = []
  
  for fraction in feature_set_fractions:
    clf = RandomForestClassifier(n_estimators=50, random_state=0, max_features=fraction, oob_score=True)
    clf.fit(train_samples, train_classes)
    clf_pred = clf.predict(test_samples)
    oob_errors.append(1 - clf.oob_score_)
    test_errors.append(1- accuracy_score(test_classes, clf_pred))
  
  plt.plot(feature_set_fractions, oob_errors, label="oob error rate")
  plt.plot(feature_set_fractions, test_errors, label="test error rate")
  plt.xlabel("Max features")
  plt.ylabel("error rates")
  plt.legend(loc="lower right")
  plt.show()
  
def is_number(pstr):
  try:
    float(pstr)
  except ValueError:
    return False
  return True

def set_parameters(params):
  '''
  Set parametes from command line arguments
  '''
  if len(params) < 1:
    return
  if params[0] in ("-h", "--help"):
    print("Arguments supported: \n \
          -t (--number-of-trees )  number of trees to construct in random forest (default : 10) \
          -f (--max-features ) feature selection set to take the best split feature (default : sqrt) \
            sqrt : max features will be sqrt(total_features) \
            log2 : max features will be log2(total_features) \
            float : if its fraction b/w 0 and 1 then features will be that fraction of total features \
           ")
    sys.exit(0)
  if len(params) < 2:
    return
  global NUMBER_OF_TREES
  try:
    pdict, args = getopt.getopt(params,  "t:f:", ["help=", "--number-of-trees", "--max-features"])
  except getopt.GetoptError:
    print("options are -t (--number-of-trees) and -f (--max-features) \n")
    sys.exit(2)
  for pname, pvalue in pdict:
    if pname in ("-t", "--number-of-trees"):
      NUMBER_OF_TREES = int(pvalue)
    elif pname in ("-f", "--max-features"):
      if is_number(pvalue) and float(pvalue) > 0 and float(pvalue) <= 1:
        dt.FEATURE_SELECTION_SET = float(pvalue)
      else:
        dt.FEATURE_SELECTION_SET = str(pvalue)
    else:
      print("Invalid option, please specify in \n -t (--number-of-trees ) or  f (--max-features ) \n ")
      sys.exit(2)

if __name__ == "__main__":
  set_parameters(sys.argv[1:])
  run_random_forest()