#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Sun Sep  1 16:59:44 2019

@author: nageswara
"""

weights_file = './data/train.weights'
dev_set_file = './data/dev'
test_set_file = './data/test'

trans_weights = {}  # key: (tag, tag)
emission_weights = {}  # key: (tag, token)
all_tags = set()
default_tag = ''

SMALL_DEFAULT_WEIGHT = -1000
# Read the weights

with open(weights_file, 'r') as weights:
    for weight in weights:
        (weight_desc, value) = weight.rstrip().split(' ')
        weight_type = weight_desc.split('_')
        if weight_type[0] == 'T':
            trans_weights[(weight_type[1], weight_type[2])] = \
                float(value)
            all_tags.update([weight_type[1], weight_type[2]])
        if weight_type[0] == 'E':
            emission_weights[(weight_type[1], weight_type[2])] = \
                float(value)
            all_tags.add(weight_type[1])

default_tag = next(iter(all_tags))

def decode(observations, expected_tag_seq):
   # For each sequence, intialize a matrix to run Viterbi algorithm
        seq_len = len(observations)
        viterbi = {}
        backpointer = {}
        
        for i in range(0, seq_len):
            for tag in all_tags:
                viterbi[(i, tag)] = float('-inf')

   # Initalize first token's probabilities in observation
        first_token = observations[0]
        for tag in all_tags:
            if (tag, first_token) not in emission_weights:
                emission_weights[(tag, first_token)] = SMALL_DEFAULT_WEIGHT
            viterbi[(0, tag)] = emission_weights[(tag, first_token)]
            backpointer[(0, tag)] = 0

    # Run dynamic programming algo to get the best path
        for i in range(1, seq_len):
            cur_obs = observations[i]
            best_prev_tag = default_tag
            for cur_tag in all_tags:
                for prev_tag in all_tags:
                    if (prev_tag, cur_tag) not in trans_weights:
                        trans_weights[(prev_tag, cur_tag)] = \
                            SMALL_DEFAULT_WEIGHT
                    if (cur_tag, cur_obs) not in emission_weights:
                        emission_weights[(cur_tag, cur_obs)] = \
                            SMALL_DEFAULT_WEIGHT

                    current_weight = viterbi[(i - 1, prev_tag)] \
                        + (emission_weights[(cur_tag, cur_obs)] \
                        + trans_weights[(prev_tag, cur_tag)])
                    if current_weight >= viterbi[(i, cur_tag)]:
                        viterbi[(i, cur_tag)] = current_weight
                        best_prev_tag = prev_tag
                backpointer[(i, cur_tag)] = best_prev_tag

   # get the best backpointer for the last token to track back the best path

        best_score = float('-inf')
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

    # Get the accuracy of curret observation by matching expected & estimated
        
        obsr_accuracy = 0
        for tag_it in range(seq_len):
            if(estimated_tag_seq[tag_it] == expected_tag_seq[tag_it]):
                obsr_accuracy += 1
        obsr_accuracy /= seq_len
        
        return (estimated_tag_seq, obsr_accuracy)

# Read each test observation sequence and predict the tag sequence
def get_test_set_accuracy():
    accuracy = 0
    total_obsr = 0
    with open(test_set_file, 'r') as test_set:
        for test_obs in test_set:
            (seq_len, *obs_tags) = test_obs.rstrip().split(' ')
            seq_len = int(seq_len)
            observations = []
            expected_tag_seq = []
            for i in range(0, seq_len * 2, 2):
                observations.append(obs_tags[i])
                expected_tag_seq.append(obs_tags[i + 1])
    
            (estimated_tag_seq, obsr_accuracy) = decode(observations, expected_tag_seq)
            accuracy += obsr_accuracy
            total_obsr += 1
    return accuracy/total_obsr

if __name__ == "__main__":
    obsr_sentence = input("Enter input observation sentence: ")
    expected_tags_seq = input("Enter expected tag sequence: ")
    (estimated_tag_seq, obsr_accuracy) = decode(obsr_sentence.rstrip().split(" "), \
     expected_tags_seq.rstrip().split(" "))
    print("Estimated tag sequence: {}, \n Accuracy for the senetence: {:.2f} % ".format \
          (estimated_tag_seq, obsr_accuracy*100))
    
    accuracy = get_test_set_accuracy()
    print("\n Overall accuracy on test set: {:.2f} %".format(accuracy * 100))