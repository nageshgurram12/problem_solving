# -*- coding: utf-8 -*-
import argparse

class TreeNode():
    def __init__(self, node, left=None, right=None, first_term=None, last_term=None):
        self.node = node
        self.left = left
        self.right = right
        self.first_term = first_term
        self.last_term = last_term
        self.is_pre_term = False
        self.is_term = False
        self.features = []
        
    '''
    Add feature for this node in tree
    '''
    def add_feature(self, feature):
        self.features.append(feature)
     
    '''
    Create features for non terminal inside the tree by 
    getting first and last terminals and also left & right children
    '''
    def create_non_term_features(self):
        if self.is_term or self.is_pre_term:
            return
        
        # add rule feature
        self.features.append("R_" + self.node + "_" + self.left.node + \
                             "_" + self.right.node)
        
        # for first and last word features, traverse down
        # TODO: can optimize further
        first_term = self
        while first_term != None and not first_term.is_pre_term:
            first_term = first_term.left
        
        self.features.append("F_" + self.node + "_" + first_term.left.node)
        
        last_term = self
        while last_term != None and not last_term.is_pre_term:
            last_term = last_term.right
        
        # since all terminals are to left of pre-term
        self.features.append("L_" + self.node + "_" + last_term.left.node)
            

'''
Return strigified version of tree
'''
def stringify_tree(root):
    if root.is_term:
        return root.node
    elif root.is_pre_term:
        return "("+ root.node + " " + root.left.node +")"
    else:
        return "(" + root.node + " " + stringify_tree(root.left) + \
        " " + stringify_tree(root.right) + ")"
        
  
def isTerm(str1):
    return str1.islower()

'''
Create pre-terminal node in parse tree by taking terminal of it
'''
def create_pre_term_node(pre_term, term):
    term_node = TreeNode(term)
    term_node.is_term = True
                
    pre_term_node = TreeNode(pre_term, left=term_node)
    pre_term_node.is_pre_term = True
    pre_term_node.add_feature('T_' + pre_term + "_" + term)
    
    return pre_term_node
    
'''
Parse line and create a given parse tree 
'''
def get_sentence_parser(line):
    line_stack = [] # stack for parsing the line
    node_stack = [] # stack to store already built nodes
    
    root = None # Parsed tree root
    sentence = '' # parsed sentence
    non_terminals = []
    
    # TODO - assumption
    # Iterating over line with each char, 
    # assuming all terminals and non terminals are only characters
    for part in line:
        if part == ' ':
            continue
        
        if part == "(":
            line_stack.append("(")
        elif part == ")":
            # Pop term and non term from stack if top of stack is term
            if isTerm(line_stack[-1]):
                term = line_stack.pop()
                pre_term = line_stack.pop()
                
                pre_term_node = create_pre_term_node(pre_term, term)
                node_stack.append(pre_term_node)
                
            # pop two nodes from node stack and add as right and left children
            # for non term in top of the line stack 
            else:
                root = non_term_node = TreeNode(line_stack.pop())
                non_term_node.right = node_stack.pop() # order is important
                non_term_node.left = node_stack.pop()
                non_term_node.create_non_term_features()
                
                node_stack.append(non_term_node)
                
            line_stack.pop() # remove (
        else:
            line_stack.append(part)
            
            if isTerm(part):
                sentence += part + " "
            else:
                non_terminals.append(part)
    
    return (root, sentence.rstrip(), non_terminals)

'''
Run decoding to get best parse tree
'''
def decode(sentence, all_non_terms, feat_weights):    
    
    # utilities 
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
    Generate best parsed tree based on back pointers
    '''
    def construct_tree(node_info, back_ptrs):
        node_details = node_info.split("_") # S_0_7
     
        # B_A_6:506 or a:1
        node_children = back_ptrs[node_info].split(":")[0].split("_")
        if((int(node_details[2]) - int(node_details[1])) == 1): # a:1
            return create_pre_term_node(node_details[0], node_children[0])
        else:
            # create a non terminal node in tree with left and right children
            # add features while creating the node
            node_nt = node_details[0] # S
            left_nt = node_children[0] # B
            right_nt = node_children[1] # A
            node = TreeNode(node_nt)
            
            # add all features to this node in tree
            rule_ft = "R_" + node_nt + "_" + left_nt + "_" + right_nt
            node.add_feature(rule_ft)
            
            first_word_ft = "F_" + node_nt + "_" + sentence[int(node_details[1])]
            node.add_feature(first_word_ft)
            
            last_word_ft = "L_" + node_nt + "_" + sentence[int(node_details[2])-1]
            node.add_feature(last_word_ft)
            
            # Do recursion to construct left and right subtrees
            left_node_info = node_children[0] + "_" + \
                       node_details[1] + "_" + node_children[2]
            node.left = construct_tree(left_node_info, back_ptrs)
            
            right_node_info = node_children[1] + "_" + \
                       node_children[2] + "_" + node_details[2]
            node.right = construct_tree(right_node_info, back_ptrs)
            
            return node
        
        
    table = {} # key = NT_si_ei
    back = {}
    sentence = sentence.split(" ") # TODO: assumption nt are always chars
    sen_len= len(sentence)
    
    for i in range(0, sen_len):
        for pre_term in all_non_terms:
            table_key = prepare_table_key(pre_term, i, i+1)
            pre_term_feat_key = prepare_feature_key('T', pre_term, sentence[i])
            pre_term_score = get_val_default(pre_term_feat_key, feat_weights)
            
            # Create leaf nodes with preterminals for each word 
            table[table_key] = pre_term_score
            back[table_key] = sentence[i] + ":" + str(pre_term_score)
    
    # Notation is [i, k) word at index i is included and word at k is not
    # For each substring [i, k) search for best subtree that can be constructed
    # with any non-terminal
    for span in range(2, sen_len+1):  # span = 2 to N
        for i in range(0, sen_len-span+1): # i = 0 to N-span
            k = i + span # k= i+span
            for j in range(i+1, k): # j from i+1 to k-1
                for nt_A in all_non_terms:
                    A_key = prepare_table_key(nt_A, i, k)
                    # initialize table values with -inf
                    if A_key in table:
                        exist_val = table[A_key]
                    else:
                        exist_val = float('-inf')
                    for nt_B in all_non_terms:
                        for nt_C in all_non_terms:
                            # feature keys for rule, first and last word
                            ft_r_key = prepare_feature_key('R', nt_A, nt_B, nt_C)
                            ft_fw_key = prepare_feature_key('F', nt_A, sentence[i])
                            ft_lw_key = prepare_feature_key('L', nt_A, sentence[k-1])
                            
                            # summation of these 
                            new_val = get_val_default(ft_r_key, feat_weights) + \
                            get_val_default(ft_fw_key, feat_weights) + \
                            get_val_default(ft_lw_key, feat_weights)
                            
                            # left and right nodes scores
                            B_key = prepare_table_key(nt_B, i, j)
                            C_key = prepare_table_key(nt_C, j, k)
                            new_val += get_val_default(B_key, table) + \
                            get_val_default(C_key, table)
                            
                            # get best sub tree structure for current node A
                            if new_val > exist_val:
                                exist_val = new_val
                                table[A_key] = new_val
                                back[A_key] = nt_B + "_" + nt_C \
                                + "_" + str(j) + ":" + str(new_val)
    
    root_key = "S_0_" + str(sen_len)
    
    return construct_tree(root_key, back)
    
'''
Update feature weights in given tree with either +1 or -1
'''
def update_feature_weights(root, weights, val):
    if root.is_term or root == None:
        return
    
    # update feature scores of current node
    for feature in root.features:
        if feature in weights:
            weights[feature] += val
        else:
            weights[feature] = val
    
    if root.left != None:
        update_feature_weights(root.left, weights, val)
        
    if root.right != None:
        update_feature_weights(root.right, weights, val)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptron learning for CFG parsers")
    
    parser.add_argument('-R', '--train-sentences', type=str,\
                        default='./data/train_sentences', \
                        help='Provide training sentences file')
    
    parser.add_argument('-L', '--weights-learned', type=str, metavar='W',\
                        default='./data/weights_learned', \
                        help='Learned weights after perceptron')
    
    args = parser.parse_args()
    

    weights = {} # all weights are initialized to 0
    itr = 0
    with open(args.train_sentences) as f:
        for line in f.readlines():
            print("\n iter : " + str(itr)); itr += 1
            (parse_tree, sentence, non_terminals) = get_sentence_parser(line.strip())
            print("Gold tree: " + stringify_tree(parse_tree))
            
            # decode a tree for sentence with current weights
            decoded_tree = decode(sentence, non_terminals, weights)
            print("Decoded tree: " + stringify_tree(decoded_tree))
            
            # Update feature weights by comparing parse tree and decode tree features
            update_feature_weights(parse_tree, weights, 1)
            update_feature_weights(decoded_tree, weights, -1)
            
            print("weights after update: " + str(weights))
            
    with open(args.weights_learned, "w") as f:
        for weight in weights.items():
            f.write("%s %s\n" % weight)