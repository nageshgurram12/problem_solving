# -*- coding: utf-8 -*-
import argparse
import re

'''
Class that represents a feature T_A_a 1
left and right are values of the same in rule (not tree)
'''
class Feature():
    def __init__(self, type, left, right):
        self.type = type
        self.left = left
        self.right = right
        
'''
Take a feature string in format T_A_a 1 and create a feature object
'''
def create_feature(feature_str):
    feat_parts = re.split(r'[_\s]', feature_str)
    if len(feat_parts) not in (4, 5):
        raise ValueError
    if len(feat_parts) == 5:
        feat_parts[2] = [feat_parts[2], feat_parts[3]]
        del feat_parts[3]
    return Feature(*feat_parts)
    
'''
Parse the given feature weights file and generate feature objects
'''
def parse_feat_weights(weights_file):
    # Pre terminals appear just above terminals in tree
    pre_terms = set()
    
    # set of all non terminals including pre-terminals
    all_non_terms = set()
    
    # all features given
    features = {}
    
    with open(weights_file) as f:
        feat_lines = f.readlines()
        for feat_line in feat_lines:
            #feat = create_feature(feat_line.strip())
            feat_line = feat_line.strip()
            feat_parts = re.split(r'[_\s]', feat_line)
        
            if feat_parts[0] == 'T':
                pre_terms.add(feat_parts[1])
            
            if feat_parts[0] == 'R':
                 feat_parts[2] = feat_parts[2] + "_" + feat_parts[3]
                 del feat_parts[3]
            
            all_non_terms.add(feat_parts[1])
            weight = feat_parts[3]
            features["_".join(feat_parts[0:3])] = weight
    
    return (pre_terms, all_non_terms, features)

def prepare_table_key(non_term, start_ix, end_ix):
    return non_term + "_" + str(start_ix) + "_" + str(end_ix)

def prepare_feature_key(*args):
    return "_".join(args)

def get_val_default(key, dictr):
    if key in dictr:
        return float(dictr[key])
    else:
        return 0

'''
Generate a string version of the best parse tree 
by traversing back pointers
''' 
def stringify(back_ptrs, best_scr_node):
    node_details = best_scr_node.split("_") # S_0_7
     
    # B_A_6:506 or a:1
    node_children = back_ptrs[best_scr_node].split(":")[0].split("_")
    if((int(node_details[2]) - int(node_details[1])) == 1):
        return "(" + node_details[0] + " " + node_children[0] + ")" 
    else:
        split_nodes = (node_children[0] + "_" + \
                       node_details[1] + "_" + node_children[2], \
                       node_children[1] + "_" + \
                       node_children[2] + "_" + node_details[2]) # (B_0_6, A_6_7)
        return "(" + \
            node_details[0] + " " + \
            stringify(back_ptrs, split_nodes[0]) + " " + \
            stringify(back_ptrs, split_nodes[1]) + ")"
    
    
def decode(sentence, pre_terms, all_non_terms, feat_weigts):    
    
    table = {} # key = NT_si_ei
    back = {}
    sentence = sentence.split(" ")
    sen_len= len(sentence)
    
    for i in range(0, sen_len):
        for pre_term in all_non_terms:
            table_key = prepare_table_key(pre_term, i, i+1)
            pre_term_feat_key = prepare_feature_key('T', pre_term, sentence[i])
            pre_term_score = get_val_default(pre_term_feat_key, feat_weigts)
            
            # Create leaf nodes with preterminals for each word 
            table[table_key] = pre_term_score
            back[table_key] = sentence[i] + ":" + str(pre_term_score)
    
    # Notation is [i, k) word at index i is included and at k is not
    # For each substring [i, k) search for best subtree that can be constructed
    # with any non-terminal
    for span in range(2, sen_len+1):  # span = 2 to N
        for i in range(0, sen_len-span+1): # i = 0 to N-span
            k = i + span # k= i+span
            for j in range(i+1, k): # j from i+1 to k-1
                for nt_A in all_non_terms:
                    A_key = prepare_table_key(nt_A, i, k)
                    exist_val = get_val_default(A_key, table)
                    for nt_B in all_non_terms:
                        for nt_C in all_non_terms:
                            ft_r_key = prepare_feature_key('R', nt_A, nt_B, nt_C)
                            ft_fw_key = prepare_feature_key('F', nt_A, sentence[i])
                            ft_lw_key = prepare_feature_key('L', nt_A, sentence[k-1])
                            new_val = get_val_default(ft_r_key, feat_weigts) + \
                            get_val_default(ft_fw_key, feat_weigts) + \
                            get_val_default(ft_lw_key, feat_weigts)
                            
                            B_key = prepare_table_key(nt_B, i, j)
                            C_key = prepare_table_key(nt_C, j, k)
                            new_val += get_val_default(B_key, table) + \
                            get_val_default(C_key, table)
                            #new_val = table[B_key] + table[C_key]
                            
                            if new_val >= exist_val:
                                exist_val = new_val
                                table[A_key] = new_val
                                back[A_key] = nt_B + "_" + nt_C \
                                + "_" + str(j) + ":" + str(new_val)
    
    best_scr_key = "S_0_" + str(sen_len)
    
    print("\n Sentence: " + " ".join(sentence))
    print("Best tree score: " + best_scr_key + " -> " + \
          str(back[best_scr_key]))
    
    print("Parse tree: " + stringify(back, best_scr_key))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKY parsing")
    
    parser.add_argument('-T', '--test-sentences', type=str,\
                        default='./data/test_sentences', \
                        help='Provide test sentences file')
        
    parser.add_argument('-W', '--weights', type=str, metavar='W', \
                        default='./data/weights', \
                        help='Provide weights for features, \
                        (otherwise training will be run to get them )')
    args = parser.parse_args()
    
    if args.weights:
        (pre_terms, all_non_terms, features) = \
        parse_feat_weights(args.weights)
        
    with open(args.test_sentences) as f:
        test_sentences = [line.strip() for line in f.readlines()]
        
    for sentence in test_sentences:
        decode(sentence = sentence,
               pre_terms = pre_terms,
               all_non_terms = all_non_terms,
               feat_weigts = features)