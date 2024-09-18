import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from MDLP.discretization import *
from gnb_functions import quantileDiscretize, create_score_matrix, 
                          get_triplet_list_from_chu_liu, get_sorted_triple_list, 
                          test_result, test_results

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--algorithm", type=str, default="GNBO")
parser.add_argument("--num_of_intervals", type=int, default=5)

args = parser.parse_args()

data_path = args.data_path
algorithm = args.algorithm
assert algorithm in ["GNBO", "GNBA"]

################################################
# read and preprocess data
################################################
data = pd.read_csv(data_path, sep=',', header=None, 
                   names = ['index', 'Y']+['X{}'.format(i) for i in range(1,31)])
data.drop(columns=['index'], inplace=True)
data.replace({'M': 1, 'B': 0}, inplace=True)
num_cols = data.shape[1]
label1 = data['Y'].unique()[0]
label2 = data['Y'].unique()[1]
label_pos = label1

var_to_index_dict = {}
for i in range(1,num_cols): var_to_index_dict['X{}'.format(i)] = i-1

################################################
# discretize data
################################################
data_discr = quantileDiscretize(data, args.num_of_intervals)

################################################
# split data
################################################
data_train, data_test = train_test_split(data_discr, test_size=0.15, random_state=20)

################################################
# run algorithm
################################################
if algorithm == "GNBO":
  score_matrix, pair1, v1, pair2, v2, max_triplet, v3 = create_score_matrix(data_train, num_cols-1)
  heads, tree_score = chu_liu_edmonds(score_matrix)
  triplet_list, pair_list, ic_list = get_triplet_list_from_chu_liu(heads, int(max_triplet[1][1:]), int(max_triplet[2][1:]))
elif algorithm == "GNBA":
  max_ic, max_ic_diff_dict, triplet_list = get_sorted_triple_list(data_train)
  pair_list = None

################################################
# test with all the triplets
################################################
data_test_result, accuracy, precision, recall, auc = test_result(data_test, 
                                                                data_train, 
                                                                triplet_list, 
                                                                pair_list, 
                                                                label_pos,
                                                                algorithm)
print('accuracy:', round(accuracy, 4))
print('precision:', round(precision, 4)) 
print('recall:', round(recall, 4)) 
print('f1 score:', round(2*precision*recall/(precision+recall), 4))
print('AUC score:', round(auc, 4))

################################################
# test and plot with triplets added one by one
################################################
data_test_results, accuracy_list, precision_list, recall_list, auc_list = test_results(X_test, 
                                                                                      X_train, 
                                                                                      triplet_list, 
                                                                                      pair_list, 
                                                                                      label_pos,
                                                                                      algorithm)
plt.plot(list(range(1,num_cols-1)), np.array(accuracy_list), label='accuracy')
plt.plot(list(range(1,num_cols-1)), np.array(precision_list), label='precision')
plt.plot(list(range(1,num_cols-1)), np.array(recall_list), label='recall')
plt.plot(list(range(1,num_cols-1)), 2*(np.array(precision_list)*np.array(recall_list))\
/(np.array(precision_list)+np.array(recall_list)), label='F1 score')
plt.plot(list(range(1,num_cols-1)), np.array(auc_list), label='AUC score')
plt.xlabel('Number of triplets used')
plt.legend()
plt.show()

