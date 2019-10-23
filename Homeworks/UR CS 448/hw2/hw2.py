#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Sun Sep  1 16:59:44 2019

@author: nageswara
"""
import numpy as np

train_set_file = './data/train'
dev_set_file = './data/dev'
test_set_file = './data/test'

trans_weights = {}  # key: (tag, tag)
emission_weights = {}  # key: (tag, token)
all_tags = set()
all_tokens = set()

all_train_observations = []
all_train_tag_seqs = []

all_test_observations = []
all_test_tag_seqs = []

NEG_INF = float('-inf')
START_SYM = 'START'
RANDOM_SAMPLE_SIZE = 1000 # Bootstrap sample size 

TOTAL_ITERATIONS = 100 # Total number of iterations to run on train data
lr = 0.1 # learning rate

# Read the train data and store in local memory

with open(train_set_file, 'r') as train_set:
    for example in train_set:
        (seq_len, *obs_tag_seq) = example.rstrip().split(' ')
        seq_len = int(seq_len)
        observation = []; tag_seq = []
        for i in range(0, seq_len * 2, 2):
            all_tokens.add(obs_tag_seq[i])
            all_tags.add(obs_tag_seq[i + 1])
            observation.append(obs_tag_seq[i])
            tag_seq.append(obs_tag_seq[i + 1])
        all_train_observations.append(observation)
        all_train_tag_seqs.append(tag_seq)

# Read validation set
    
with open(test_set_file, 'r') as test_set:
    for example in test_set:
        (seq_len, *obs_tag_seq) = example.rstrip().split(' ')
        seq_len = int(seq_len)
        observation = []; tag_seq = []
        for i in range(0, seq_len * 2, 2):
            observation.append(obs_tag_seq[i])
            tag_seq.append(obs_tag_seq[i + 1])
        all_test_observations.append(observation)
        all_test_tag_seqs.append(tag_seq)
    
def set_default(weight_map, key1, key2):
    #initialize weights with gussian noise
    if (key1, key2) not in weight_map:
        weight_map[(key1, key2)] = np.random.normal(0,1,1)[0] 

def increment(weight_map, key1, key2):
    set_default(weight_map, key1, key2)
    weight_map[(key1, key2)] += lr * 1

def decrement(weight_map, key1, key2):
    set_default(weight_map, key1, key2)
    weight_map[(key1, key2)] -= lr * 1
    
# Initialize all weights with 0
for tag1 in all_tags:
    for tag2 in all_tags:
        set_default(trans_weights, tag1, tag2)
    set_default(trans_weights, START_SYM, tag1)
    for token in all_tokens:
        set_default(emission_weights, tag1, token)

default_tag = next(iter(all_tags))


def decode(observations, expected_tag_seq):

   # For each sequence, intialize a matrix to run Viterbi algorithm

    viterbi = {}
    backpointer = {}
    seq_len = len(observations)
    for i in range(0, seq_len):
        for tag in all_tags:
            viterbi[(i, tag)] = NEG_INF

   # Initalize first token's probabilities in observation

    for tag in all_tags:
        set_default(emission_weights, tag, observations[0])
        set_default(trans_weights, START_SYM, tag)
            
        viterbi[(0, tag)] = trans_weights[(START_SYM, tag)] \
            * emission_weights[(tag, observations[0])]
        backpointer[(0, tag)] = 0

    for i in range(1, seq_len):
        cur_obs = observations[i]
        best_prev_tag = default_tag
        for cur_tag in all_tags:
            for prev_tag in all_tags:
                set_default(trans_weights, prev_tag, cur_tag)
                set_default(emission_weights, cur_tag, cur_obs)

                current_weight = viterbi[(i - 1, prev_tag)] \
                    + emission_weights[(cur_tag, cur_obs)] \
                    + trans_weights[(prev_tag, cur_tag)]
                if current_weight > viterbi[(i, cur_tag)]:
                    viterbi[(i, cur_tag)] = current_weight
                    best_prev_tag = prev_tag
            backpointer[(i, cur_tag)] = best_prev_tag

   # get the best backpointer for the last token to track back the best path

    best_score = NEG_INF
    bestbackpointer = default_tag
    for tag in all_tags:
        if best_score < viterbi[(seq_len - 1, tag)]:
            best_score = viterbi[(seq_len - 1, tag)]
            bestbackpointer = tag

   # trace back to get the best path

    estimated_tag_seq = [None] * seq_len
    estimated_tag_seq[seq_len - 1] = bestbackpointer
    for i in reversed(range(seq_len)):
        if i == 0:
            break
        estimated_tag_seq[i - 1] = backpointer[(i,
                estimated_tag_seq[i])]
    return estimated_tag_seq

def get_accuracy(expected_seq, estimated_seq):
    obsr_accuracy = 0
    seq_len = len(expected_seq)
    for tag_it in range(seq_len):
        if(expected_seq[tag_it] == estimated_seq[tag_it]):
            obsr_accuracy += 1
    obsr_accuracy /= seq_len
    return obsr_accuracy
    
def train():
    global trans_weights, emission_weights, lr
    trans_weights = {}  # clear all weights
    emission_weights = {}
    for it in range(TOTAL_ITERATIONS):
        train_set_len = len(all_train_observations)
        accuracy = 0
        rand_sample = np.random.randint(0, high=train_set_len, \
                                        size=RANDOM_SAMPLE_SIZE)
        if it%10 == 1:
            lr *= 0.5
            
        for i in np.nditer(rand_sample):
            observations = all_train_observations[i]
            expected_tag_seq = all_train_tag_seqs[i]
            estimated_tag_seq = decode(observations, expected_tag_seq)
            accuracy += get_accuracy(expected_tag_seq, estimated_tag_seq)
            
            # Update the weights 
            
            for i in range(len(observations)):
                obsr = observations[i]
                expected_tag = expected_tag_seq[i]
                estimated_tag = estimated_tag_seq[i]
                if i == 0:
                    prev_expt_tag = START_SYM
                    prev_esti_tag = START_SYM
                else:
                    prev_expt_tag = expected_tag_seq[i-1]
                    prev_esti_tag = estimated_tag_seq[i-1]
                
                if expected_tag != estimated_tag:
                    increment(emission_weights, expected_tag, obsr)
                    decrement(emission_weights, estimated_tag, obsr)
                    
                    increment(trans_weights, prev_expt_tag, expected_tag)
                    decrement(trans_weights, prev_esti_tag, estimated_tag)
                    
                    if i != len(observations)-1:
                        increment(trans_weights, expected_tag, \
                                  expected_tag_seq[i+1])
                        decrement(trans_weights, estimated_tag, \
                                  estimated_tag_seq[i+1])
            
        # print train accuracy with every iteration
        print("Train Accuracy: %.4f \
              Iteration : %d" % (accuracy/RANDOM_SAMPLE_SIZE, it))

def test():
    total_obs = 0
    accuracy = 0
    
    for i in range(len(all_test_observations)):
        observation = all_test_observations[i]
        expected_tag_seq = all_test_tag_seqs[i]
        estimated_tag_seq = decode(observation, expected_tag_seq)
        accuracy += get_accuracy(expected_tag_seq, estimated_tag_seq)
        total_obs += 1
        
        # Print some test predictions to console
        if(i%500 == 0):
            print("\n Observation : {} \n, Expected Tag Sequence : {} \n, \
                  Estimated Tag Sequence : {}".\
                  format(observation, expected_tag_seq, estimated_tag_seq))
            
    return accuracy/total_obs

if __name__ == '__main__':
    train()
    print("\nTest accuracy : %.4f" % (test()))