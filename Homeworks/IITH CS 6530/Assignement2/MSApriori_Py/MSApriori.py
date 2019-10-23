# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 23:07:31 2018

@author: nrgurram
"""

# Imports here
import os
import sys
import itertools
import time
import math

#Initialize constants 
# SUPPORT AND MIS values are in fractions
LAMDA = 0.5 # MIS = sup(i) * lamda
MAX_FREQ_ITEM_SIZE = 2 # Maximum frequest item set size
LEAST_MIS = 0.02 # The least MIS an item can get
SPD = 1 # support difference constraint to avoid mixing of frequent and rare items

# Initialize variables here
MIS = dict() # minimum item support of each item
database = list() # complete database of transactions, each transaction is set of items
item_count_map = dict() # Map with item,count as key value pair
database_size = 0 # total number of items
items = list() # items sorted based on their MIS values
freq_itemsets = list() # frequent item sets for level k , k is itemset size
candidate_item_sets = list() # candidate frequent item sets for level k-1
candidate_itemsets_count = dict() # count of each candidate set in transactions

can_not_be_together_itemsets = list()
current_dir = os.path.dirname(__file__)
output_file = open(os.path.join(current_dir, "output.txt"), "w")

def main():
    start_time = time.time()
    
    read_input_file()
    create_MIS()
    sort_items_transactions()
    initial_pass()
    MSApriori_for_size2_and_above()

    end_time = time.time()
    print("Total time taken " + str(round(end_time - start_time, 4)))
    #TODO: remove these
    #print(database)
    #print(candidate_item_sets)
    #print(str(freq_itemsets))
    output_file.close()
"""
Read input file and create a transaction database
"""
def read_input_file():
    global current_dir,database,item_count_map,database_size
    rel_path_to_data_file = "../dataset/retail1/retail1.txt"
    path_to_data_file = os.path.join(current_dir, rel_path_to_data_file)
    data_file = open(path_to_data_file)
    
    for transaction in data_file:
        transaction = transaction.rstrip('\n\r')
        if transaction:
            t_items = transaction.split(' ')
        database.append(list())
        for item in t_items:
            item = int(item)
            database[len(database)-1].append(item)
            if(item_count_map.get(item) != None):
                item_count_map[item] = item_count_map[item] + 1
            else:
                item_count_map[item] = 1
    
    database_size = len(database)
    data_file.close()
    
"""
Create MIS based on items support
MIS(i) = item.count/n * lamda
"""
def create_MIS():
    global MIS,item_count_map, LAMDA, database_size, SPD
    LS_relative = int(math.ceil(LEAST_MIS * database_size))
    SPD = int(math.ceil(SPD * database_size))
    for item in item_count_map:
        MIS[item] = int(item_support(item) * LAMDA)
        if MIS[item] < LS_relative:
            MIS[item] = LS_relative
    
"""
Sort items based on MIS values
sort transactions based on MIS values
"""
def sort_items_transactions():
    global MIS,items,item_count_map, database
    items = sorted(item_count_map.keys(), key=lambda x : (MIS[x], x))
    for transaction in database:
        transaction.sort(key=lambda x : (MIS[x], x))
        
"""
In initial pass if item support is greater than MIS(l) then add to 1-candidates set
l - item with least MIS value
In 1-candidates set if each item support is greater than its MIS, then add to 1-frequent itemsets
"""
def initial_pass():
    global items,item_count_map,MIS,candidate_item_sets,freq_itemsets
    minMIS = -1
    firstMinMISItemIndex = 0
    for i in range(0, len(items)):
        item = items[i]
        if item_support(item) >= MIS[item]:
            minMIS = MIS[item]
            firstMinMISItemIndex = i
            candidate_item_sets.append(item)
            break
    
    for i in range(firstMinMISItemIndex+1, len(items)):
        item = items[i]
        if item_support(item) >= minMIS:
            candidate_item_sets.append(item)
    
    if len(candidate_item_sets) == 0:
        print >> sys.stderr, "NO FREQUENT ITEMSETS"
        sys.exit(1)
    
    for item in candidate_item_sets:
        if item_support(item) >= MIS[item]:
            freq_itemsets.append(item)
            write_to_file([item], item_support(item))


"""
Generate frequent itemsets of size > 1 
In this we generate candidate item sets of size 2 by calling
i) if size = 2, generate_candidates_of_size2
ii) if size > 2, generate_candidates_of_size2_above
Once we have candidates, do a pass over database and filter the itemsets that doesn't satisfy MIS
"""
def MSApriori_for_size2_and_above():
    global database,candidate_item_sets,freq_itemsets,MAX_FREQ_ITEM_SIZE
    set_size = 2
    while(True):
        if(set_size > MAX_FREQ_ITEM_SIZE or len(freq_itemsets) < 2):
            break
        if set_size == 2:
            candidate_item_sets = generate_candidates_of_size2()
        else:
            candidate_item_sets = generate_candidates_of_size2_above()
        
        # filter candidates based on support count
        filter_candidates()
        
        # clear frequent itemsets
        freq_itemsets = list()
        # check if candidate support is atleast as much as MIS[itemset[0]] i.e first item,
        # then only candidate itemset satisifies minimum support rule
        print(candidate_item_sets)
        for itemset in candidate_item_sets:
            if itemset_support(itemset) >= MIS[itemset[0]]:
                freq_itemsets.append(itemset)
                write_to_file(itemset, itemset_support(itemset))
        
        #print(freq_itemsets)
        # Increment set_size
        set_size = set_size + 1

"""
Filter candidate itemsets by scanning database in MIS order
and by ensuring each candidate itemset is in MIS order
"""
def filter_candidates():
    global database, candidate_item_sets, MIS
    # Fitler the candidates by scaning the database
    for transaction in database:
       for candidate_itemset in candidate_item_sets:
           pos = 0
           for item in transaction:
               if item == candidate_itemset[pos]:
                   pos = pos + 1
                   # If candidate is in transaction then increment its count
                   if pos == len(candidate_itemset):
                       incerement_candidate_count(candidate_itemset)
                       break
               # If item in transaction and candidate item set is not equals, then decide to search 
               # further based on MIS order, if item in transaction has greater MIS then break the loop
               # and take another candidate
               elif MIS_comparator(item, candidate_itemset[pos]) > 0:
                   break


"""
Generate candidates of size 2 for each pair in MIS order
i) {l, h} : l - item with less MIS, h - item with MIS(h) > MIS(l)
ii) conisder supprort difference constraint i.e |sup(h) - sup(l)| <= SPD
"""
def generate_candidates_of_size2():
    global candidate_item_sets,MIS,SPD
    candidate_item_sets_size2 = list()
    for l in range(0,len(candidate_item_sets)):
        lower_MIS_item = candidate_item_sets[l]
        sup_lower_MIS_item = item_support(lower_MIS_item)
        for h in range(l+1, len(candidate_item_sets)):
            higher_MIS_item = candidate_item_sets[h]
            sup_higher_MIS_item = item_support(higher_MIS_item)
            if(abs(sup_higher_MIS_item - sup_lower_MIS_item) <= SPD):
                candidate_item_sets_size2.append([lower_MIS_item, higher_MIS_item])
    return candidate_item_sets_size2
    
"""
Generate candidate sets of size K from K-1 size
"""
def generate_candidates_of_size2_above():
    global candidate_item_sets,MIS,SPD
    
    candidate_item_sets_sizeK = list()
    for i in range(0, len(candidate_item_sets)):
        itemset1 = candidate_item_sets[i]
        for j in range(i+1, len(candidate_item_sets)):
            itemset2 = candidate_item_sets[j]
            # compare itemset1, itemset2 if they have same items except last item then join
            for k in range(0, len(itemset1)):
                break_iloop = False
                break_jloop = False
                if(k == (len(itemset1) - 1)):
                    if(MIS_comparator(itemset1[k], itemset2[k]) > 0):
                       break_iloop = True
                       break
                    else:
                       # prepare new itemset size k from k-1 itemsets by joining last items
                       if(abs(itemset_support(itemset1) - itemset_support(itemset2)) <= SPD):
                           new_itemset = list(itemset1[0:k])
                           new_itemset.extend([itemset1[k], itemset2[k]])
                           candidate_item_sets_sizeK.append(new_itemset)
                elif(MIS_comparator(itemset1[k], itemset2[k]) < 0):
                    break_jloop = True
                    break
                elif(MIS_comparator(itemset1[k], itemset2[k]) > 0):
                    break_iloop = True
                    break
            if(break_jloop):
                break
        if(break_iloop):
            break
    
    # reset candidate itemsets and prepare it with new itemsets of size k
    candidate_item_sets = list()
    # Now for each itemset of size k generated, check all k-1 subsets are frequent
    for candidate_itemset in candidate_item_sets_sizeK:
        if(all_subsets_of_k_1_size_are_frequent(candidate_itemset)):
            candidate_item_sets.append(candidate_itemset)
    return candidate_item_sets_sizeK

"""
Compare MIS of two items and if they are equal, follow lexi order
"""
def MIS_comparator(item1,item2):
    global MIS
    m1 = MIS[int(item1)]; m2 = MIS[int(item2)]
    if(m1 == m2):
        return (item1 > item2) - (item1 < item2)
    else:
        return (m1 > m2) - (m1 < m2)
    
"""
Check if all subsets of size k-1 are frequent for the given itemset
"""
def all_subsets_of_k_1_size_are_frequent(itemset):
    global MIS, freq_itemsets
    
    # generate all subsets by always each item from the candidate, one by one
    all_k_1_subsets = list(itertools.combinations(itemset, len(itemset)-1))
    total_subsets = len(all_k_1_subsets)
    for k in range(0, total_subsets):
        k_1_itemset = list(all_k_1_subsets[k])
        # Dont exclude subset without first item unless MIS of first item and second are same
        if((k == total_subsets-1) and (MIS[itemset[0]] != MIS[itemset[1]])):
            continue
        elif(k_1_itemset not in freq_itemsets):
            return False
    return True

"""
Calculate support of an itemset
"""         
def itemset_support(itemset):
    global database_size
    return get_candidate_count(itemset)

"""
Calculate support of an item
"""         
def item_support(item):
    global item_count_map,database_size
    return item_count_map[item]
"""
Increment support count of candidate itemset if exists 
initialize to 1 otherwise
"""
def incerement_candidate_count(itemset):
    global candidate_itemsets_count
    count = get_candidate_count(itemset)
    count = count + 1
    candidate_itemsets_count[tuple(itemset)] = count

def get_candidate_count(itemset):
    global candidate_itemsets_count
    return candidate_itemsets_count.get(tuple(itemset), 0)

"""
Write item set to file with support count
"""
def write_to_file(itemset, support):
    global output_file
    itemset_string = ""
    for item in itemset:
        itemset_string = itemset_string + str(item)
        if itemset[-1] != item:
            itemset_string = itemset_string + ", "
    itemset_string = itemset_string + " #SUP: " + str(support)
    #itemset_string = itemset_string + " #MIS: " + str(mis_item)
    output_file.write(itemset_string + "\n")

"""
def group_filters():
    global can_not_be_together_itemsets, freq_itemsets, must_be_together_itemsets
    for i in range(0, len(freq_itemsets)):
        freq_itemset = freq_itemsets[i]
        for j in range(0, len(can_not_be_together_itemsets)):
            can_not_be_together_pair = can_not_be_together_itemsets[j]
            if(can_not_be_together_pair[0] in )
"""
if __name__ == "__main__": main()