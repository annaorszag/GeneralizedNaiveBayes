import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import math
import operator
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ufal.chu_liu_edmonds import chu_liu_edmonds


def quantileDiscretize(df,numOfIntervals):
    cols = list(df.columns)
    colList = [list(df[c]) for c in cols]
    elementsPerCol = [len(set(df[c])) for c in cols]
    
    def discrCol(L):
        Q = np.quantile(L, np.linspace(0,1,numOfIntervals+1)[1:-1])
        intervals = [[l for l in L if l < Q[0]]] + [[l for l in L if Q[i] <= l < Q[i+1]] for i in range(0,numOfIntervals-2)] + [[l for l in L if Q[-1] <= l]]
        intervalAverages = [np.mean(I) for I in intervals]
        correctIntervals = [min([q if l < Q[q] else len(Q) for q in range(len(Q))]) for l in L]
        return [intervalAverages[cI] for cI in correctIntervals]    
    
    newCols = [discrCol(colList[c]) if elementsPerCol[c] > numOfIntervals else colList[c] for c in range(len(colList))]
    newdata = dict(zip(cols,newCols))
    return pd.DataFrame(data=newdata)

def get_information_content(vector, dataframe):
    if len(vector)==2:
        vector_r_dict = dict(zip(dataframe[vector].value_counts().keys().tolist(), 
                      (dataframe[vector].value_counts()/dataframe.shape[0]).tolist()))
        vector_0_r_dict = dict(zip(dataframe[vector[0]].value_counts().keys().tolist(), 
                              (dataframe[vector[0]].value_counts()/dataframe.shape[0]).tolist()))
        vector_1_r_dict = dict(zip(dataframe[vector[1]].value_counts().keys().tolist(), 
                              (dataframe[vector[1]].value_counts()/dataframe.shape[0]).tolist()))
        ic = 0
        for r in vector_r_dict.keys():
            ic += vector_r_dict[r]*math.log(vector_r_dict[r]/
                                    (vector_0_r_dict[r[0]]*vector_1_r_dict[r[1]]), 2)
    elif len(vector)==3:
        vector_r_dict = dict(zip(dataframe[vector].value_counts().keys().tolist(), 
                      (dataframe[vector].value_counts()/dataframe.shape[0]).tolist()))
        vector_0_r_dict = dict(zip(dataframe[vector[0]].value_counts().keys().tolist(), 
                              (dataframe[vector[0]].value_counts()/dataframe.shape[0]).tolist()))
        vector_1_r_dict = dict(zip(dataframe[vector[1]].value_counts().keys().tolist(), 
                              (dataframe[vector[1]].value_counts()/dataframe.shape[0]).tolist()))
        vector_2_r_dict = dict(zip(dataframe[vector[2]].value_counts().keys().tolist(), 
                              (dataframe[vector[2]].value_counts()/dataframe.shape[0]).tolist()))
        ic = 0
        for r in vector_r_dict.keys():
            ic += vector_r_dict[r]*math.log2(vector_r_dict[r]/
                                    (vector_0_r_dict[r[0]]
                                     *vector_1_r_dict[r[1]]
                                     *vector_2_r_dict[r[2]]))
    else:
        print('Vector length is not 2 or 3')
    return ic


def create_score_matrix(data_train, num):
    V_list = data_train.columns[1:].tolist()
    all_pairs_list = [('Y', V_list[i]) for i in range(len(V_list))]
    all_triplets_list = [('Y', V_list[i], V_list[j]) for i in range(len(V_list)-1) for j in range(i+1, len(V_list))]
    pair_ic_dict = {v:get_information_content(list(v), data_train) for v in all_pairs_list}
    triplet_ic_dict = {v:get_information_content(list(v), data_train) for v in all_triplets_list}
    score_matrix = np.zeros((1, len(data_train.columns)))
    for r in range(1,num+1):
        row = np.zeros((1, len(data_train.columns)))
        for c in range(1,num+1):
            if c != r:
                X_r = 'X' + str(r)
                X_c = 'X' + str(c)
                row[0,c] = get_information_content(list(('Y', X_r, X_c)), data_train) - pair_ic_dict[('Y',X_c)]
        score_matrix = np.concatenate((score_matrix, row), axis=0)
    
    max_triplet = max(triplet_ic_dict.items(), key=operator.itemgetter(1))[0]
    max_triplet_ic = max(triplet_ic_dict.items(), key=operator.itemgetter(1))[1]
    
    max_row = int(str(max_triplet[1])[1:])
    score_matrix[max_row, :] = 0
    score_matrix[max_row, 0] = pair_ic_dict[('Y', max_triplet[1])]
    
    max_row2 = int(str(max_triplet[2])[1:])
    score_matrix[max_row2, :] = 0
    score_matrix[max_row2, max_row] = get_information_content(list(('X'+str(max_row), 'X'+str(max_row2))), data_train)
    print('max triplet:', max_triplet, ' information content:', max_triplet_ic)
    return score_matrix, \
          ('Y', max_triplet[1]), score_matrix[max_row, 0], \
          ('X'+str(max_row), 'X'+str(max_row2)), score_matrix[max_row2, max_row], \
          max_triplet, triplet_ic_dict[max_triplet]

def get_triplet_list_from_chu_liu(heads, X_max_index, X_max2_index):
    triplet_list = []
    pair_list = []
    #X_max_index = heads.index(0)
    #X_max2_index = heads.index(X_max_index)
    print(X_max_index, X_max2_index)
    for i in range(1, len(heads)):
        if i != X_max_index:
            triplet_list.append(('Y', 'X'+str(heads[i]), 'X'+str(i)))
            if i != X_max2_index:
                pair_list.append(('Y', 'X'+str(heads[i])))
    return triplet_list, pair_list

def replace_triplet_prob(t0, label, dataframe, vector_realization, var_to_index_dict):
    t0p1_prob = dataframe.loc[(dataframe[t0[0]]==label)\
                     &(dataframe[t0[1]]==vector_realization[var_to_index_dict[t0[1]]])].shape[0]/dataframe.shape[0]
    if t0p1_prob==0:
        t0p1_prob = replace_pair_prob((t0[0],t0[1]), label, dataframe, vector_realization)
    t0p2_prob = dataframe.loc[(dataframe[t0[0]]==label)\
                 &(dataframe[t0[2]]==vector_realization[var_to_index_dict[t0[2]]])].shape[0]/dataframe.shape[0]
    if t0p2_prob==0:
        t0p2_prob = replace_pair_prob((t0[0],t0[2]), label, dataframe, vector_realization)
    t0l_prob = dataframe.loc[(dataframe[t0[0]]==label)].shape[0]/dataframe.shape[0]
    t0_prob2 = t0p1_prob*t0p2_prob/t0l_prob
    return t0_prob2
def replace_pair_prob(pair, label, dataframe, vector_realization, var_to_index_dict):
    p1_prob = dataframe.loc[(dataframe[pair[0]]==label)].shape[0]/dataframe.shape[0]
    p2_prob = dataframe.loc[(dataframe[pair[1]]==vector_realization[var_to_index_dict[pair[1]]])].shape[0]/dataframe.shape[0]
    p_prob = p1_prob*p2_prob
    return p_prob

def get_probability(vector_realization, dataframe, triplet_list, pair_list, label, var_to_index_dict, algorithm): 
    t_prob_list = []
    p_prob_list = []
    if algorithm == 'GNBA':
      t0 = triplet_list[0]
      t0_prob = dataframe.loc[(dataframe[t0[0]]==label)\
                       &(dataframe[t0[1]]==vector_realization[var_to_index_dict[t0[1]]])\
                       &(dataframe[t0[2]]==vector_realization[var_to_index_dict[t0[2]]])].shape[0]/dataframe.shape[0]
      if t0_prob>0:
          t_prob_list.append(t0_prob)
      else:
          t0_prob2 = replace_triplet_prob(t0, label, dataframe, vector_realization)
          t_prob_list.append(t0_prob2)
      for triplet in triplet_list[1:]:
          p_prob = dataframe.loc[(dataframe[triplet[0]]==label)\
                       &(dataframe[triplet[1]]==vector_realization[var_to_index_dict[triplet[1]]])].shape[0]/dataframe.shape[0]
          if p_prob>0:
              p_prob_list.append(p_prob)
              t_prob = dataframe.loc[(dataframe[triplet[0]]==label)\
                           &(dataframe[triplet[1]]==vector_realization[var_to_index_dict[triplet[1]]])\
                           &(dataframe[triplet[2]]==vector_realization[var_to_index_dict[triplet[2]]])].shape[0]/dataframe.shape[0]
              if t_prob>0:
                  t_prob_list.append(t_prob)
              else:
                  t_prob2 = replace_triplet_prob(triplet, label, dataframe, vector_realization)
                  t_prob_list.append(t_prob2)
          else:
              p_prob2 = replace_pair_prob((triplet[0], triplet[1]), label, dataframe, vector_realization)
              t_prob2 = replace_triplet_prob(triplet, label, dataframe, vector_realization)
              p_prob_list.append(p_prob2)
              t_prob_list.append(t_prob2)
      prob = np.prod(t_prob_list)/np.prod(p_prob_list)
      
    elif algorithm == 'GNBO':
      for triplet in triplet_list:
          t_prob = dataframe.loc[(dataframe[triplet[0]]==label)\
                           &(dataframe[triplet[1]]==vector_realization[var_to_index_dict[triplet[1]]])\
                           &(dataframe[triplet[2]]==vector_realization[var_to_index_dict[triplet[2]]])].shape[0]/dataframe.shape[0]
          if t_prob>0:
              t_prob_list.append(t_prob)
          else:
              t_prob2 = replace_triplet_prob(triplet, label, dataframe, vector_realization)
              t_prob_list.append(t_prob2)
      #pair_list.remove(('Y', 'X0'))
      for pair in pair_list:
          p_prob = dataframe.loc[(dataframe[pair[0]]==label)\
                       &(dataframe[pair[1]]==vector_realization[var_to_index_dict[pair[1]]])].shape[0]/dataframe.shape[0]
          if p_prob>0:
              p_prob_list.append(p_prob)
          else:
              p_prob2 = replace_pair_prob((pair[0], pair[1]), label, dataframe, vector_realization)
              p_prob_list.append(p_prob2)
      prob = np.prod(t_prob_list)/np.prod(p_prob_list)
    return prob

def get_probabilities_and_prediction(vector_realization, y, dataframe, triplet_list, pair_list, algorithm):
    # p_1
    p_1 = get_probability(vector_realization, dataframe, triplet_list, pair_list, label1, algorithm)
    # p_2
    p_2 = get_probability(vector_realization, dataframe, triplet_list, pair_list, label2, algorithm)
    
    if p_1 >= p_2:
        pred = label1
    elif p_1 < p_2:
        pred = label2
    else:
        pred = np.nan
    return pd.DataFrame({'label': y, 'prob_'+str(label1): p_1, 'prob_'+str(label2): p_2, 'prediction': pred})

def get_sorted_triple_list(data_train):
    V_list = data_train.columns[1:].tolist()
    all_triplets_list = [('Y', V_list[i], V_list[j]) for i in range(len(V_list)-1) for j in range(i+1, len(V_list))]
    all_pairs_list = [('Y', V_list[i]) for i in range(len(V_list))]
    triplet_ic_dict = {v:get_information_content(list(v), data_train) for v in all_triplets_list}
    max_ic_triplet = max(triplet_ic_dict.items(), key=operator.itemgetter(1))[0]

    V2_list = [('Y', max_ic_triplet[1]), ('Y', max_ic_triplet[2])]
    V3_list = [max_ic_triplet]

    V_1, V_2 = max_ic_triplet[1], max_ic_triplet[2]
    V_list.remove(V_1)
    V_list.remove(V_2)
    
    num_steps = len(V_list)
    max_ic_diff_dict = {}
    for step in range(num_steps):
        triplet_list = [('Y', V2[1], V_new) for V2 in V2_list for V_new in V_list]
        ic_diff_dict = {triple: 
                        get_information_content(list(triple), data_train) 
                        - get_information_content([triple[0], triple[1]], data_train) 
                        for triple in triplet_list}
        max_ic_diff_triplet, max_ic_diff_dict[max_ic_diff_triplet[2]] = max(ic_diff_dict.items(), key=operator.itemgetter(1))
        V_list.remove(max_ic_diff_triplet[2])
        V2_list.append(('Y', max_ic_diff_triplet[2]))
        V3_list.append(max_ic_diff_triplet)
    max_ic = triplet_ic_dict[max_ic_triplet]
    return max_ic, max_ic_diff_dict, V3_list

def get_sorted_triplet_list_from_chu_liu(heads, score_matrix, X_max_index, X_max2_index):
    d_parent = {}
    d_ic = {}
    for i in range(1, len(heads)):
        d_parent[i] = heads[i]
        d_ic[i] = score_matrix[i, heads[i]]
    
    sorted_triplet_list = []
    sorted_pair_list = []
    ic_list = []
    for i in range(1, len(heads)):
        non_parents = list(set(d_parent.keys()) - set(d_parent.values()))
        d_temp = {k:d_ic[k] for k in non_parents}
        min_ic_vertex = min(d_temp, key=d_temp.get)
        if (min_ic_vertex != X_max_index) & (min_ic_vertex != X_max2_index):
            sorted_triplet_list.append(('Y', 'X'+str(d_parent[min_ic_vertex]), 'X'+str(min_ic_vertex)))
            sorted_pair_list.append(('Y', 'X'+str(d_parent[min_ic_vertex])))
            ic_list.append(d_ic[min_ic_vertex])
        del d_parent[min_ic_vertex]
    sorted_triplet_list.append(('Y', 'X'+str(heads[X_max2_index]), 'X'+str(X_max2_index)))
    ic_list.append(d_ic[X_max2_index])
    #sorted_pair_list = sorted_pair_list[:-1]
    print(d_parent)
    return sorted_triplet_list[::-1], sorted_pair_list[::-1], ic_list[::-1]

def test_result(data_test, data_train, V3_list, V2_list, label_pos, algorithm):
    data_test_results = pd.DataFrame(columns = ['label', 'prob_'+str(label1), 'prob_'+str(label2), 'prediction'])
    for index, row in data_test.iterrows():
        y = np.array([row['Y']])
        vector = np.array(row[1:])
        pred_row = get_probabilities_and_prediction(vector, y, data_train, V3_list, V2_list, algorithm)
        data_test_results = pd.concat([data_test_results, pred_row], 
                                      ignore_index=True)

    accuracy = data_test_results.loc[data_test_results.label == data_test_results.prediction].shape[0]\
                                            /data_test_results.shape[0]
    precision = data_test_results.loc[(data_test_results.label == label_pos)\
                                      &(data_test_results.prediction == label_pos)].shape[0]\
                                        /data_test_results.loc[data_test_results.prediction == label_pos].shape[0]
    recall = data_test_results.loc[(data_test_results.label == label_pos)\
                                   &(data_test_results.prediction == label_pos)].shape[0]\
                                        /data_test_results.loc[data_test_results.label == label_pos].shape[0]
    auc = roc_auc_score(data_test_results['label'], data_test_results['prediction'])
    return data_test_results, accuracy, precision, recall, auc

def test_results(data_test, data_train, V3_list, V2_list, label_pos):
    accuracy_list = []
    precision_list = []
    recall_list = []
    auc_list = []
    for num_of_triplets in range(1,num_cols-1):
        data_test_results = pd.DataFrame(columns = ['label', 'prob_'+str(label1), 'prob_'+str(label2), 'prediction'])
        for index, row in data_test.iterrows():
            y = np.array([row['Y']])
            vector = np.array(row[1:])
            pred_row = get_probabilities_and_prediction(vector, y, data_train, V3_list[:num_of_triplets], V2_list[:(num_of_triplets-1)], label_pos, algorithm)
            data_test_results = pd.concat([data_test_results, pred_row], 
                                          ignore_index=True)

        accuracy = data_test_results.loc[data_test_results.label == data_test_results.prediction].shape[0]\
                                            /data_test_results.shape[0]
        precision = data_test_results.loc[(data_test_results.label == label_pos)\
                                          &(data_test_results.prediction == label_pos)].shape[0]\
                                            /data_test_results.loc[data_test_results.prediction == label_pos].shape[0]
        recall = data_test_results.loc[(data_test_results.label == label_pos)\
                                       &(data_test_results.prediction == label_pos)].shape[0]\
                                            /data_test_results.loc[data_test_results.label == label_pos].shape[0]
        auc = roc_auc_score(data_test_results['label'], data_test_results['prediction'])
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc)
    return data_test_results, accuracy_list, precision_list, recall_list, auc_list
