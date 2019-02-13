# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:43:36 2018

@author: Thomas Anderson
"""

import codecs
import numpy as np
import csv
import re

def export_all_labels(filepath, all_label_pred, label_class):
    nrow = len(all_label_pred[label_class[0]])
    ncol = len(label_class)
    all_labels_array = np.ndarray(shape=(nrow, ncol), dtype=int)
    for i in range(nrow):
        for j in range(ncol):
            all_labels_array[i][j] = all_label_pred[label_class[j]][i]
    
    np.savetxt(filepath, all_labels_array)

def export_community(filepath, community_list):
    community_list_length = len(community_list)
    time_series_length = len(community_list[0].timeline)
    community_list_for_export = []
    export_header_str = ('code,longitude,latitude,grade,'
                         'price_1,price_2,price_3,price_4,price_5,price_6,price_7,price_8,price_9,price_10,'
                         'price_11,price_12,price_13,price_14,price_15,price_16,price_17,price_18,price_19,price_20,'
                         'price_21,price_22,price_23,price_24,'
                         'district,region,name,address,archi_age,vol_rate,green_rate,pro_costs,'
                         'ts_eucli,ts_pearson,ts_dtw')
    export_header = []
    for one_field_name in re.split(",", export_header_str):
        export_header.append(one_field_name)
    
    for i in range(community_list_length):
        one_community = []
        one_community.append(community_list[i].code)
        one_community.append(community_list[i].longitude)
        one_community.append(community_list[i].latitude)
        one_community.append(community_list[i].grade)
        for j in range(time_series_length):
            one_community.append(community_list[i].price_time_series[j])
        one_community.append(community_list[i].district)
        one_community.append(community_list[i].region)
        one_community.append(community_list[i].name)
        one_community.append(community_list[i].address)
        one_community.append(community_list[i].architectural_age)
        one_community.append(community_list[i].volume_rate)
        one_community.append(community_list[i].greening_rate)
        one_community.append(community_list[i].property_costs)
        one_community.append(community_list[i].price_time_series_euclidean_cluster_label)
        one_community.append(community_list[i].price_time_series_pearson_cluster_label)
        one_community.append(community_list[i].price_time_series_dtw_cluster_label)
        
        community_list_for_export.append(one_community)
    with codecs.open(filepath, 'w', 'utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(export_header)
        writer.writerows(community_list_for_export)

def export_price_time_series_pearson_coefficient_matrix(filepath, community_list):
    community_number = len(community_list)
    price_time_series_length = len(community_list[0].timeline)
    dist_matrix = np.zeros((community_number, community_number), dtype=np.float64)
    row = dist_matrix.shape[0]
    col = dist_matrix.shape[1]
    for i in range(row):
        if i%500 == 0:
            print(i)
        for j in range(col):
            if i==j:
                continue;
            elif dist_matrix[i][j] == 0:
                if dist_matrix[j][i] > 0:
                    dist_matrix[i][j] = dist_matrix[j][i]
                elif dist_matrix[j][i] == 0:
                    sum_i = 0
                    sum_j = 0
                    for k in range(price_time_series_length):
                        sum_i += community_list[i].price_time_series[k]
                        sum_j += community_list[j].price_time_series[k]
                    avg_i = float(sum_i) / float(k)
                    avg_j = float(sum_j) / float(k)
                    sum_numerator = 0
                    sum_denominator_1 = 0
                    sum_denominator_2 = 0
                    for k in range(price_time_series_length):
                        sum_numerator += (community_list[i].price_time_series[k] - avg_i) * (community_list[j].price_time_series[k] - avg_j)
                        sum_denominator_1 += pow(community_list[i].price_time_series[k] - avg_i, 2)
                        sum_denominator_2 += pow(community_list[j].price_time_series[k] - avg_j, 2)
                    dist_matrix[i][j] = float(sum_numerator) / (np.sqrt(float(sum_denominator_1)) * np.sqrt(float(sum_denominator_2)))
                    dist_matrix[j][i] = dist_matrix[i][j]
            elif dist_matrix[i][j] > 0:
                if dist_matrix[j][i] > 0:
                    continue;
                elif dist_matrix[j][i] == 0:
                    dist_matrix[j][i] = dist_matrix[i][j]
                    
    for i in range(row):
        for j in range(col):
            dist_matrix[i][j] = 1 - dist_matrix[i][j]
    
    np.savetxt(filepath, dist_matrix)

def export_price_time_series_euclidean_distance_matrix(filepath, community_list):
    community_number = len(community_list)
    price_time_series_length = len(community_list[0].timeline)
    dist_matrix = np.zeros((community_number, community_number), dtype=np.float64)
    row = dist_matrix.shape[0]
    col = dist_matrix.shape[1]
    for i in range(row):
        if i%500 == 0:
            print(i)
        for j in range(col):
            if i==j:
                continue;
            elif dist_matrix[i][j] == 0:
                if dist_matrix[j][i] > 0:
                    dist_matrix[i][j] = dist_matrix[j][i]
                elif dist_matrix[j][i] == 0:
                    summary = 0
                    for k in range(price_time_series_length):
                        summary += (community_list[i].price_time_series[k] - community_list[j].price_time_series[k])
                    dist_matrix[i][j] = float(summary) / float(k)
                    dist_matrix[j][i] = dist_matrix[i][j]
            elif dist_matrix[i][j] > 0:
                if dist_matrix[j][i] > 0:
                    continue;
                elif dist_matrix[j][i] == 0:
                    dist_matrix[j][i] = dist_matrix[i][j]
    
    np.savetxt(filepath, dist_matrix)

# export location_euclidean_distance_matrix
def export_location_euclidean_distance_matrix(filepath, community_list):
    enlarge_ratio = 100
    community_number = len(community_list)
    dist_matrix = np.zeros((community_number, community_number), dtype=np.float64)
    row = dist_matrix.shape[0]
    col = dist_matrix.shape[1]
    for i in range(row):
        if i%500 == 0:
            print(i)
        for j in range(col):
            if i==j:
                continue;
            elif dist_matrix[i][j] == 0:
                if dist_matrix[j][i] > 0:
                    dist_matrix[i][j] = dist_matrix[j][i]
                elif dist_matrix[j][i] == 0:
                    dist_matrix[i][j] = np.sqrt( pow(enlarge_ratio*(community_list[i].latitude - community_list[j].latitude),2) 
                    + pow(enlarge_ratio*(community_list[i].longitude - community_list[j].longitude),2) )
                    dist_matrix[j][i] = dist_matrix[i][j]
            elif dist_matrix[i][j] > 0:
                if dist_matrix[j][i] > 0:
                    continue;
                elif dist_matrix[j][i] == 0:
                    dist_matrix[j][i] = dist_matrix[i][j]
                
    np.savetxt(filepath, dist_matrix)

# export community_location
def export_community_location(filepath, community_list):
    with codecs.open(filepath, 'w', 'utf-8') as f:
        for community in community_list:
            f.write('{}'.format(community.longitude))
            f.write('\t\t')
            f.write('{}'.format(community.latitude))
            f.write('\t\t')
            f.write('\n')

def export_inter_class_samples_distribution(filepath, label_number, label_class, label_level_num):
    with codecs.open(filepath, 'w', 'utf-8') as f:
        for one_label_class in label_class:
            for i in range(label_level_num):
                f.write('{}'.format(label_number[one_label_class][str(i+1)]))
                f.write('\t\t')
            f.write('\n')

# export community_price_time_series
def export_community_price_time_series(filepath, community_list):
    with codecs.open(filepath, 'w', 'utf-8') as f:
        '''
        for name in header:
            if name == 'price_time_series':
                for i in range(24):
                    f.write(name + '_' + str(i+1) + '\t\t')
            else:
                f.write(name + '\t\t')
        f.write('\n')
        '''
        
        for community in community_list:
            '''
            f.write(community.address)
            f.write('\t\t')
            f.write('{}'.format(community.architectural_age))
            f.write('\t\t')
            f.write(community.code)
            f.write('\t\t')
            f.write('{}'.format(community.construction_area))
            f.write('\t\t')
            f.write(community.district)
            f.write('\t\t')
            f.write('{}'.format(community.floor_area))
            f.write('\t\t')
            f.write('{}'.format(community.greening_rate))
            f.write('\t\t')
            f.write('{}'.format(community.latitude))
            f.write('\t\t')
            f.write('{}'.format(community.longitude))
            f.write('\t\t')
            f.write(community.name)
            f.write('\t\t')
            '''
            for i in range(24):
                f.write('{}'.format(community.price_time_series[i]))
                f.write('\t\t')
            '''
            f.write('{}'.format(community.property_costs))
            f.write('\t\t')
            f.write(community.region)
            f.write('\t\t')
            f.write('{}'.format(community.volume_rate))
            f.write('\t\t')
            '''
            f.write('\n')
            
# export community_greening_rate_and_volume_rate
def export_community_two_rates(filepath, community_list):
    with codecs.open(filepath, 'w', 'utf-8') as f:
        for community in community_list:
            f.write('{}'.format(community.greening_rate))
            f.write('\t\t')
            f.write('{}'.format(community.volume_rate))
            f.write('\t\t')
            f.write('\n')

# export community_price_time_series_avg_and_var
def export_community_price_time_series_avg_and_var(filepath, community_list):
    with codecs.open(filepath, 'w', 'utf-8') as f:
        for community in community_list:
            f.write('{}'.format(community.price_average))
            f.write('\t\t')
            f.write('{}'.format(community.price_variance))
            f.write('\t\t')
            f.write('\n')