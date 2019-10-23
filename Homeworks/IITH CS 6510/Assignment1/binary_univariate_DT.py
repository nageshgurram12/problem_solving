# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:18:35 2018

@author: nrgurram
"""
import scipy.stats as sp
import numpy as np

import csv
import sys, getopt

# Enter You Name Here
myname = "nag-" # or "Doe-Jane-"

LEAF_THRESHOLD_IMPURITY = 0.1

LEAF_DATASET_THRESHOLD_COUNT = 10

PRUNE_DEPTH = 10

KFOLD_CV = 1

IMPURITY_MEASURE = "entropy"

class Node():
  '''
  Represents node in decision tree
  '''
  def __init__(self, feature, examples=None, is_leaf=False):
    '''
    Initialization of node in tree
    '''
    #signifies whether the node is leaf or not
    self.is_leaf = is_leaf
    
    '''
    The feature on which decision is to be taken
    This is determined by calculating information gain on all features and  by 
    selecting the feature which maximizes it 
    '''
    self.feature = feature
    
    #feature breakpoint
    self.feature_bp = None
    
    #Examples in leaf node
    self.examples = examples
    
    #left node of tree
    self.left_node = None
    
    #right node oftree
    self.right_node = None

  def get_majority_class(self):
    if self.examples is None or self.examples.shape[0] == 0:
      return 0;
    return sp.mode(self.examples[:,-1]).mode[0]

  def print(self, level=0):
    if self.is_leaf:
      print("  "*level, "leaf :", self.examples.shape)
    else:
      print("  "*level, "logic node: ", (self.feature, self.feature_bp))

class DecisionTree():
  '''
  The built decision tree based on training examples
  '''

  def __init__(self):
    '''
    Tree data structure
    '''
    self.root = None

  def classify(self, test_instance):
    '''
    Classify a test instance with decision tree
    '''
    result = 0 # baseline: always classifies as 0
    if self.root == None:
      return result
    result = self.test_for(test_instance)
    return result

  def test_for(self, test_instance):
    '''
    Run the instance against the tree
    '''
    node = self.root
    while not node.is_leaf:
      if test_instance[node.feature] <= node.feature_bp:
        node = node.left_node
      else:
        node = node.right_node
    return node.get_majority_class()

  def learn(self, training_set):
    '''
    Build the decision tree by learning on training_set
    '''
    if training_set.size == 0 or training_set.shape[0] == 0:
        raise ValueError("Training set can't be empty")
    features = np.arange(training_set.shape[1]-1)
    self.root = self.build_DT(features, training_set)

  def build_DT(self, features, dataset):
    '''
    Build the decision tree
    '''
    if dataset.shape[0] == 0:
      return Node(None, is_leaf=True)
    
    initial_impurity = self.get_impurity_of_dataset(dataset)
    if self.stop_criteria(features, dataset, initial_impurity):
      return Node(None, dataset, True)

    # Find the best split feature by calcualting information gain for each one
    (best_split_impurity, best_split_feature, feature_bp) = self.get_best_split_feature(features, dataset)
    
    # Create a internal node
    logic_node = Node(best_split_feature, is_leaf=False)
    logic_node.feature_bp = feature_bp
    # divide data set based on best_split_feature
    (left_dataset, right_dataset) = self.split_dataset(best_split_feature, feature_bp, dataset)
    
    if left_dataset.shape[0] != 0:
      logic_node.left_node = self.build_DT(features, left_dataset)
    
    if right_dataset.shape[0] != 0:
      logic_node.right_node = self.build_DT(features, right_dataset)
    return logic_node


  def stop_criteria(self, features, dataset, initial_impurity):
    '''
    Stop splitting the node further and create a leaf node when 
    1) initial_impurity <= LEAF_THRESHOLD_IMPURITY
    2) dataset < LEAF_DATASET_THRESHOLD_COUNT
    '''
    if initial_impurity <= LEAF_THRESHOLD_IMPURITY:
      return True
    if dataset.shape[0] <= LEAF_DATASET_THRESHOLD_COUNT:
      return True
    
    return False

  def get_best_split_feature(self, features, dataset):
    '''
      Return the best split attribute based on information gain
    '''
    min_impurity = 1
    for feature in features:
      (imp_on_feat, min_imp_bp) = self.get_impurity_on(feature, dataset)
      if imp_on_feat < min_impurity:
        min_impurity = imp_on_feat
        min_imp_feat = feature
        feature_bp = min_imp_bp
    return (min_impurity, min_imp_feat, feature_bp)

  def split_dataset(self, feature, feature_bp, dataset):
    '''
    Split the dataset on feature
    '''
    return (dataset[dataset[:,feature] <= feature_bp], dataset[dataset[:, feature] > feature_bp])

  def get_impurity_on(self, feature, dataset):
    '''
    Calculate impurity on this feature at breakpoints and return minimum of them
    '''
    if dataset.size == 0:
      return (0, None)
    breakpoints = self.get_breakpoints(feature, dataset)
    min_impurity = 1;
    min_imp_bp = breakpoints[0]

    for breakpoint in breakpoints:
      imp_at_bp = self.get_impurity_at_bp(breakpoint, feature, dataset)
      if imp_at_bp < min_impurity:
        min_impurity = imp_at_bp
        min_imp_bp = breakpoint
        
    return (min_impurity, min_imp_bp)

  def get_breakpoints(self, feature, dataset):
    '''
    a) Sort the feature
    b) Find all the "breakpoints" where the class labels associated with them change.
    '''
    # get array with only feature and class
    feat_and_class = np.copy(dataset[:, [feature, -1]])
    # Sort the feature vector
    feat_and_class = feat_and_class[feat_and_class[:,0].argsort()]
    # Search  for the breakpoints where classes change in order 
    # Ex : {2.4, 0}, {2.6, 0}, {3.1, 1}, {3.2, 0} then return 3.1 and 3.2
    class_labels = feat_and_class[:,-1]
    breakpoints = feat_and_class[np.where(class_labels[1:] != class_labels[:-1])]
    breakpoints = np.vstack((feat_and_class[0,:],breakpoints))
    return np.unique(breakpoints[:,0])

  def get_impurity_at_bp(self, breakpoint, feature, dataset):
    '''
    Return impurity at the breakpoint 
    '''
    left_data = np.copy(dataset[np.where(dataset[:, feature] <= breakpoint)])
    right_data = np.copy(dataset[np.where(dataset[:, feature] > breakpoint)])
    total_rows = dataset.shape[0]
    left_imp = self.get_impurity_of_dataset(left_data)
    right_imp = self.get_impurity_of_dataset(right_data)
    return ((left_data.shape[0] / total_rows) * left_imp + (right_data.shape[0] / total_rows) * right_imp)

  def get_impurity_of_dataset(self, dataset):
    '''
    Measures the impurity in dataset quality class by using entropy 
    '''
    total_rows = dataset.shape[0]
    total_0s = np.sum(dataset[:,-1] == 0)
    total_1s = np.sum(dataset[:,-1] == 1)
    return self.impurity_func([round(total_0s/total_rows,5), round(total_1s/total_rows,5)])

  def impurity_func(self, probs):
    '''
    Calculate impurity using entropy
    '''
    return round(sp.entropy(probs, base = 2.0),5)
  
  def print(self, node, level=0):
    '''
    Print the tree
    '''
    if not node:
      return
    node.print(level)
    if node.left_node:
      self.print(node.left_node, level+1)
    if node.right_node:
      self.print(node.right_node, level+1)


def run_decision_tree():
  '''
  Runs the decision tree algorithm on wine-dataset.csv with binary splits on single feature
  This programs trains only on one training set and executes on test set
  '''
    # Load data set
  with open("wine-dataset.csv") as f:
    next(f, None)
    data = [tuple(line) for line in csv.reader(f, delimiter=",")]
  print("Number of records: %d" % len(data))

    # Split training/test sets
    # You need to modify the following code for cross validation.
  K = 10
  training_set = np.array([x for i, x in enumerate(data) if i % K != 9], dtype=float)
  test_set = np.array([x for i, x in enumerate(data) if i % K == 9], dtype=float)
    
  tree = DecisionTree()
  # Construct a tree using training set
  tree.learn( training_set )

    # Classify the test set using the tree we just constructed
  results = []
  for instance in test_set:
    result = tree.classify( instance[:-1] )
    results.append( result == instance[-1])

    # Accuracy
  accuracy = float(results.count(True))/float(len(results))
  print("accuracy: %.4f" % accuracy)
    

    # Writing results to a file (DO NOT CHANGE)
  f = open(myname+"result.txt", "w")
  f.write("accuracy: %.4f" % accuracy)
  f.close()

def set_parameters(params):
  '''
  Set parametes from command line arguments
  '''
  global LEAF_THRESHOLD_IMPURITY, LEAF_DATASET_THRESHOLD_COUNT, PRUNE_DEPTH, KFOLD_CV, IMPURITY_MEASURE
  try:
    pdict = getopt.getopt(params, "ildkf", ["max_leaf_impurity=", "max_samples_in_leaf=", "min_pruning_node_depth=", "kfold=", "impurity_func="])
  except getopt.GetoptError:
    print("options are --max_leaf_impurity, --max_samples_in_leaf, --min_pruning_node_depth, --kfold, --impurity_func \n")
    sys.exit(2)
  for pname, pvalue in pdict:
    if pname in ("-i", "--max_leaf_impurity"):
      LEAF_THRESHOLD_IMPURITY = float(pvalue)
    elif pname in ("-l", "--max_samples_in_leaf"):
      LEAF_DATASET_THRESHOLD_COUNT = int(pvalue)
    elif pname in ("-d", "--min_pruning_node_depth"):
      PRUNE_DEPTH = int(pvalue)
    elif pname in ("-k", "--kfold"):
      KFOLD_CV = int(pvalue)
    elif pname in ("-f", "--impurity_func"):
      IMPURITY_MEASURE = str(pvalue)
    else:
      print("Invalid option, please specify in \n--max_leaf_impurity, --max_samples_in_leaf, --min_pruning_node_depth, --kfold, --impurity_func \n ")
      sys.exit(2)


if __name__ == "__main__":
  set_parameters(sys.argv[1:])
  run_decision_tree()