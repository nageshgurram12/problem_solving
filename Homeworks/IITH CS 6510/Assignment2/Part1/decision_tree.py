# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:18:35 2018

@author: Nageswara rao Gurram
ID: CS17EMDS11013

"""
import scipy.stats as sp
import numpy as np
import math 

import csv
import sys, getopt

# Enter You Name Here
myname = "cs17emds11013-"

# Specifies the cut off impurity level to create a leaf node
LEAF_THRESHOLD_IMPURITY = 0

# Specifies the cut off samples count in leaf
LEAF_DATASET_THRESHOLD_COUNT = 5

# Min depth of node in tree to consider in post pruning
PRUNE_DEPTH = 10

# K-fold validation
KFOLD_CV = 1

# Impurity measure 
IMPURITY_MEASURE = "gini"

# feature selection set, if its ALL then select best feature from all 
# Else support these SQRT = sqrt(M), LOG = log(M)
FEATURE_SELECTION_SET = "ALL"

class Node():
  '''
  Represents node in decision tree
  '''
  def __init__(self, feature, mode=0, is_leaf=False):
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
    
    #mode in leaf node
    self.mode = mode
    
    #left node of tree
    self.left_node = None
    
    #right node of tree
    self.right_node = None
    
    # Store accuracies when node is logic node and when its leaf(For post prunning)
    self.accuracy_count_as_leaf = 0
    
    self.accuracy_count_as_logic = 0

  def print(self, level=0):
    if self.is_leaf:
      print("  "*level, "leaf :", self.mode)
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
    return node.mode

  def post_prune(self, validation_set):
    '''
    Do post pruning and remove useless subtrees and convert them to leaves
    '''
    def update_accuracies(node, val_sample, level=1):
      '''
      Update accuracies for all internal nodes in tree by traversing with the validation sample when 
      1) node acts is leaf 
      2) node acts logic node
      '''
      if node.is_leaf:
        return node.mode
      
      if level <= PRUNE_DEPTH:
        update_accuracies(node.left_node, val_sample, level+1)
        update_accuracies(node.right_node, val_sample, level+1)
      else:
        # Increment accuracy as leaf count when val_sample class is matching node mode
        if val_sample[-1] == node.mode:
          node.accuracy_count_as_leaf += 1
        # Now get accuracy when node acts as logic node
        # If either left or right branch predicts class correct for this sample then 
        if val_sample[node.feature] <= node.feature_bp:
          left_mode = update_accuracies(node.left_node, val_sample, level+1)
          if left_mode == val_sample[-1]:
            node.accuracy_count_as_logic += 1
        else:
          right_mode = update_accuracies(node.right_node, val_sample, level+1)
          if right_mode == val_sample[-1]:
            node.accuracy_count_as_logic += 1

    for val_sample in validation_set:
      update_accuracies(self.root, val_sample)
    
    # Now traverse the entire tree and make the nodes as leaves whose accuracies are better as leaf
    def remove_unncessary_nodes(node):
      if node.is_leaf:
        return
      if node.accuracy_count_as_leaf > node.accuracy_count_as_logic:
        node.is_leaf = True
        return
      remove_unncessary_nodes(node.left_node)
      remove_unncessary_nodes(node.right_node)

    remove_unncessary_nodes(self.root)

  def learn(self, training_set):
    '''
    Build the decision tree by learning on training_set
    '''
    if training_set.size == 0 or training_set.shape[0] == 0:
        raise ValueError("Training set can't be empty")
    features = np.arange(training_set.shape[1]-1)
    self.root = self.build_DT(features, training_set)
    #self.print(self.root, level=0)

  def build_DT(self, features, dataset, level=0):
    '''
    Build the decision tree
    '''
    if dataset.shape[0] == 0:
      return Node(None, is_leaf=True)
    
    initial_impurity = self.get_impurity_of_dataset(dataset)
    if self.stop_criteria(features, dataset, initial_impurity):
      return Node(None, self.get_majority_class(dataset), True)

    # Filter feature selection set by using param FEATURE_SELECTION_SET
    rand_subset_features = self.filter_feature_selection_set(features)
    # Find the best split feature by calcualting information gain for each one
    (best_split_impurity, best_split_feature, feature_bp) = self.get_best_split_feature(rand_subset_features, dataset)
    
    # Create a internal node
    logic_node = Node(best_split_feature, is_leaf=False)
    logic_node.feature_bp = feature_bp
    # If level reached prune depth, store the mode for future purpose
    if level > PRUNE_DEPTH:
      logic_node.mode =  self.get_majority_class(dataset)
    # divide data set based on best_split_feature
    (left_dataset, right_dataset) = self.split_dataset(best_split_feature, feature_bp, dataset)
    
    if left_dataset.shape[0] != 0:
      logic_node.left_node = self.build_DT(features, left_dataset, level+1)
    else:
      logic_node.left_node = Node(None, is_leaf=True)
      
    if right_dataset.shape[0] != 0:
      logic_node.right_node = self.build_DT(features, right_dataset, level+1)
    else:
      logic_node.right_node = Node(None, is_leaf=True)
      
    return logic_node

  def filter_feature_selection_set(self, features):
    if FEATURE_SELECTION_SET == "sqrt":
      return np.random.choice(features, size=round(math.sqrt(features.size)), replace=False)
    elif FEATURE_SELECTION_SET == "log2":
      return np.random.choice(features, size=round(math.log2(features.size)), replace=False)
    elif float(FEATURE_SELECTION_SET) > 0 and float(FEATURE_SELECTION_SET) <= 1:
      return np.random.choice(features, size=round(features.size*float(FEATURE_SELECTION_SET)), replace=False)
    else:
      return features

  def get_majority_class(self, dataset):
    if dataset is None or dataset.shape[0] == 0:
      return 0;
    return sp.mode(dataset[:,-1]).mode[0]

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
    min_impurity = 1.1
    min_imp_feat = features[0]
    feature_bp = dataset[[0,0]]
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
    Calculate impurity using passed argument
    '''
    if IMPURITY_MEASURE == "entropy":
      return round(sp.entropy(probs, base = 2.0),5)
    elif IMPURITY_MEASURE == "gini":
      return round(1-sum([x**2 for x in probs]),5)
    else:
      print("Other impurity measures are not supported, So using default measure - entropy \n")
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

  # Split the data into KFOLd_CV parts 
  accuracies = np.empty(KFOLD_CV)
  for itr in range(KFOLD_CV):
    training_set = np.empty((0,len(data[0])), dtype=float)
    test_set = np.empty((0,len(data[0])), dtype= float)
    validation_set = np.empty((0,len(data[0])), dtype = float)
    
    for i,x in enumerate(data):
      if i%10 == itr:
        test_set = np.vstack((test_set, np.array(x, dtype=float)))
      elif i%10 == itr+1:
        validation_set = np.vstack((validation_set, np.array(x, dtype=float)))
      else:
        training_set = np.vstack((training_set, np.array(x, dtype=float)))
    
    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn( training_set )
    # Post prune the tree using validation set
    tree.post_prune(validation_set)
  
      # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
      result = tree.classify( instance[:-1] )
      results.append( result == instance[-1])
    # Accuracy
    accuracies[itr] = float(results.count(True))/float(len(results))

  print("accuracy: %.4f" % np.average(accuracies))

    # Writing results to a file (DO NOT CHANGE)
  f = open(myname+"result.txt", "w")
  f.write("accuracy: %.4f" % np.average(accuracies))
  f.close()

def set_parameters(params):
  '''
  Set parametes from command line arguments
  '''
  if len(params) < 1:
    return
  if params[0] in ("-h", "--help"):
    print("Arguments supported: \n \
            -i, --max_leaf_impurity : If internal node reaches below this impurity level, it'll be made as leaf \n \
            -l,--max_samples_in_leaf : If internal node gets below this samples size, it'll be made as leaf  \n \
            -d, --min_pruning_node_depth : Minimum depth of node in tree to consider in post pruning. After this depth, if node is not giving any better accuracy, make it as leaf node \n \
            -k --kfold: K-Fold cross validation \n \
            -f, --impurity_func: Impurity fucntion (entropy, gini) \n \
           ")
    sys.exit(0)
  if len(params) < 2:
    return
  global LEAF_THRESHOLD_IMPURITY, LEAF_DATASET_THRESHOLD_COUNT, PRUNE_DEPTH, KFOLD_CV, IMPURITY_MEASURE
  try:
    pdict, args = getopt.getopt(params, "i:l:d:k:f:", ["help=","max_leaf_impurity=", "max_samples_in_leaf=", "min_pruning_node_depth=", "kfold=", "impurity_func="])
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
      IMPURITY_MEASURE = str.lower(pvalue)
    else:
      print("Invalid option, please specify in \n--max_leaf_impurity, --max_samples_in_leaf, --min_pruning_node_depth, --kfold, --impurity_func \n ")
      sys.exit(2)

if __name__ == "__main__":
  set_parameters(sys.argv[1:])
  run_decision_tree()