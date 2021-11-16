# Mid Term Project - CS634 - Data Mining
# NJIT ID - rrg6
# Implementation of Apriori Algorithm in Python


import numpy as np
import pandas as pd
from itertools import combinations, chain

# User input for data
dataset_input = int(input(
    "Please Select the Dataset from the menu: \n 1. Amazon \n 2. Best Buy \n 3. K-Mart \n 4. Nike \n 5. Generic \n 6. Exit \n Your Selection is: "))

if dataset_input == 6:
    quit()

# load csv
datasets_list = ('Amazon', 'BestBuy', 'K-Mart', 'Nike', 'Generic')
df_tr = pd.read_csv("Dataset_" + datasets_list[dataset_input - 1] + ".csv")
df_itemset = pd.read_csv("Itemset_" + datasets_list[dataset_input - 1] + ".csv")
print("\nYou have selected dataset located in Dataset_" + datasets_list[dataset_input - 1] + ".csv \n")

# set order
order = sorted(df_itemset['Item Name'])

dataset = []
for lines in df_tr['Transaction']:
    trans = list(lines.strip().split(', '))
    trans_1 = list(np.unique(trans))
    trans_1.sort(key=lambda x: order.index(x))
    dataset.append(sorted(trans_1))

trans_num = len(dataset)

# User Input for Support and Confidence
minimum_support = int(input("Enter Minimum Support in % (value from 1 to 100): "))
minimum_confidence = int(input("Enter Minimum Confidence in % (value from 1 to 100): "))


# Define Functions Required

def count_items(itemset, dataset):
    count = 0
    for i in range(0, len(dataset)):
        if set(itemset).issubset(set(dataset[i])):
            count += 1
    return count


def frequent_itemsets(itemsets, dataset, minimum_support, non_frequent):
    L = []
    sup_count = []
    new_non_frequent = []
    trans_num = len(dataset)
    K = len(non_frequent.keys())
    for i in range(0, len(itemsets)):
        temp = 0
        if K > 0:
            for j in non_frequent[K]:
                if set(j).issubset(set(itemsets[i])):
                    temp = 1
                    break
        if temp == 0:
            freq_count = count_items(itemsets[i], dataset)
            if freq_count >= (minimum_support / 100) * trans_num:
                L.append(itemsets[i])
                sup_count.append(freq_count)
            else:
                new_non_frequent.append(itemsets[i])
    return L, sup_count, new_non_frequent


def print_table(table, sup_count):
    print("Itemset | Count")
    for i in range(0, len(table)):
        print("{} : {}".format(table[i], sup_count[i]))
    print("\n")


def join_itemsets(item_1, item_2, order):
    item_1.sort(key=lambda x: order.index(x))
    item_2.sort(key=lambda x: order.index(x))

    for i in range(0, len(item_1) - 1):
        if item_1[i] != item_2[i]:
            return []
    if order.index(item_1[-1]) < order.index(item_2[-1]):
        return item_1 + [item_2[-1]]
    return []


def get_candidate_set(items, order):
    Temp = []
    for i in range(0, len(items)):
        for j in range(i + 1, len(items)):
            items_1 = join_itemsets(items[i], items[j], order)
            if len(items_1) > 0:
                Temp.append(items_1)
    return Temp


def all_subsets(x):
    s = list(x)
    subsets = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))
    return subsets


def write_assoc_rules(item_2, item_1, conf, support, trans_num, rule_no):
    assoc_rules = ""
    assoc_rules += "Rule {}: {} -> {} \n".format(rule_no, list(item_1), list(item_2))
    assoc_rules += "Confidence: {0:2.2f}% \n".format(conf * 100)
    assoc_rules += "Support: {0:2.2f}% \n\n".format((support / trans_num) * 100)
    return assoc_rules


# Candidate Set and Frequent Set
C = {}
L = {}
sup_count_L = {}
itemset_size = 1
non_frequent = {itemset_size: []}

# For K = 1 Generate and Print Candidate Set and Frequent Set
C.update({itemset_size: [[f] for f in order]})
freq_set, support, new_non_frequent = frequent_itemsets(C[itemset_size], dataset, minimum_support, non_frequent)
L.update({itemset_size: freq_set})
non_frequent.update({itemset_size: new_non_frequent})
sup_count_L.update({itemset_size: support})
print("\nTable C1: \n")
print_table(C[1], [count_items(item, dataset) for item in C[1]])
print("\nTable L1: \n")
print_table(L[1], sup_count_L[1])

# For K > 1 Generate and Print Candidate Set and Frequent Set
K = itemset_size + 1
temp = 0
while temp == 0:
    C.update({K: get_candidate_set(L[K - 1], order)})
    print("\nTable C{}: \n".format(K))
    print_table(C[K], [count_items(item, dataset) for item in C[K]])
    freq_set, support, new_non_frequent = frequent_itemsets(C[K], dataset, minimum_support, non_frequent)
    L.update({K: freq_set})
    non_frequent.update({K: new_non_frequent})
    sup_count_L.update({K: support})
    if len(L[K]) == 0:
        temp = 1
    else:
        print("\nTable L{}: \n".format(K))
        print_table(L[K], sup_count_L[K])
    K += 1

# Generate and Print Association Rules
assoc_rules = ""
rule_no = 1
for i in range(1, len(L)):
    for j in range(0, len(L[i])):
        subsets = list(all_subsets(set(L[i][j])))
        subsets.pop()
        for k in subsets:
            item_1 = set(k)
            freq_set_1 = set(L[i][j])
            item_2 = set(freq_set_1 - item_1)
            support_freq_set_1 = count_items(freq_set_1, dataset)
            support_item_1 = count_items(item_1, dataset)
            support_item_2 = count_items(item_2, dataset)
            conf = support_freq_set_1 / support_item_1
            if conf >= (minimum_confidence / 100) and support_freq_set_1 >= (minimum_support / 100) * trans_num:
                assoc_rules += write_assoc_rules(item_2, item_1, conf, support_freq_set_1, trans_num, rule_no)
                rule_no += 1

print("Final Association Rules: \n")
print(assoc_rules)