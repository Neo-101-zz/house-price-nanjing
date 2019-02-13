# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 08:41:25 2018

@author: Thomas Anderson
"""
import sys
sys.path.append(r'D:\Program\Project\house_price_nanjing_soufang')
import numpy as np
import input_data
import export_data
import matplotlib_statistics
import skleran_methods
import rpy2_from_stackoverflow
import pandas_classify

invalid_value = 255
grade_class_num = 5
linkage_types_tuple = ('average', 'complete')
grade_set = set([1, 2, 3, 4, 5])
nbclust_best_num = {'location': 3, 'avg_and_var':2, 'two_rates': 3}

work_space_spider_str = r'D:\Program\Spider\house_price_nanjing_soufang' + '\\'
work_space_analysis_str = r'D:\Program\Project\house_price_nanjing_soufang\data' + '\\'

scraped_community_info_file_name = 'community_'
scraped_community_info_file_num = 4
scraped_community_info_filepath = []
for i in range(scraped_community_info_file_num):
    scraped_community_info_filepath.append(work_space_spider_str + scraped_community_info_file_name + str(i+1) + '.csv')

scraped_community_grade_file_name = 'community_grade_'
scraped_community_grade_file_num = 1
scraped_community_grade_filepath = []
for i in range(scraped_community_grade_file_num):
    scraped_community_grade_filepath.append(work_space_spider_str + scraped_community_grade_file_name + str(i+1) + '.csv')

community_export_filepath = work_space_analysis_str + 'community.csv' 

community_price_time_series_export_filepath = work_space_analysis_str + 'community_price_time_series.data'
community_location_export_filepath = work_space_analysis_str + 'community_location.data'
community_two_rates_export_filepath = work_space_analysis_str + 'community_two_rates.data'
community_price_time_series_avg_and_var_export_filepath = work_space_analysis_str + 'community_price_time_series_avg_and_var.data'
dtw_dist_matrix_filepath = work_space_analysis_str + 'dtw_dist_matrix.txt'
location_euclidean_dist_matrix_filepath = work_space_analysis_str + 'location_euclidean_dist_matrix.txt'
price_time_series_euclidean_dist_matrix_filepath = work_space_analysis_str + 'price_time_series_euclidean_dist_matrix.txt'
price_time_series_pearson_coefficient_matrix_filepath = work_space_analysis_str + 'price_time_series_pearson_coefficient_matrix.txt'
label_number_filepath = work_space_analysis_str + 'label_number.txt'
all_labels_filepath =  work_space_analysis_str + 'all_labels.txt'

label_class = (u'CTG', u'EU+GAL', u'EU+CL', 
               u'PE+GAL', u'PE+CL', 
               u'DTW+GAL', u'DTW+CL')
label_class_chinese = (u'小区总评分', u'欧几里得距离，组平均连锁', u'欧几里得距离，全连锁', 
                       u'Pearson相关系数，组平均连锁', u'Pearson相关系数，全连锁', 
                       u'DTW距离，组平均连锁', u'DTW距离，全连锁')
label_level_num = 5
label_number = {
        u'CTG': {'1': 191, '2': 757, '3': 1601, '4': 771, '5': 224}, 
        u'EU+GAL': {'1': 1394, '2': 2146, '3': 1, '4': 1, '5': 2}, 
        u'EU+CL': {'1': 293, '2': 1731, '3': 650, '4': 762, '5': 108}, 
        u'PE+GAL': {'1': 17, '2': 3471, '3': 1, '4': 3, '5': 52}, 
        u'PE+CL': {'1': 2612, '2': 387, '3': 19, '4': 508, '5': 18}, 
        u'DTW+GAL': {'1': 2953, '2': 521, '3': 2, '4': 3, '5': 65}, 
        u'DTW+CL': {'1': 955, '2': 13, '3': 2017, '4': 433, '5': 126}
        }
#show_label_number_distribution_2(label_number, label_class, label_level_num)


#export_inter_class_samples_distribution(label_number_filepath, label_number, label_class, label_level_num)
#delta = []
#for one_label_class in label_class:
#    diff = 0
#    for i in range(label_level_num):
#        diff += abs(label_number[one_label_class][str(i+1)] - label_number['小区总评分'][str(i+1)])
#    delta.append(diff)
#print(delta)


#------------------input community list------------------
print('------------------input community list------------------')
community_list = []
for i in range(scraped_community_info_file_num):
    header = import_community_info_from_csv(scraped_community_info_filepath[i], community_list)
    
#------------------get grade label------------------
print('------------------get grade label------------------')
community_grade_list = []
for i in range(scraped_community_grade_file_num):
    header = import_community_grade_from_csv(scraped_community_grade_filepath[i], community_grade_list)

for i in range(len(community_list)):
    for j in range(len(community_grade_list)):
        if community_list[i].code == community_grade_list[j].code:
            community_list[i].grade = community_grade_list[j].grade
     
for community in community_list:
    if int(community.grade) <= 0 or int(community.grade) > 5:
        community_list.remove(community)

for i in range(len(community_list)):
    if i > len(community_list) - 1:
        break
    if int(community_list[i].grade) <= 0 or int(community_list[i].grade) > 5:
        community_list.remove(community_list[i])

count_grade_0 = 0
for i in range(len(community_list)):
    if int(community_list[i].grade) <= 0 or int(community_list[i].grade) > 5:
        print(community_list[i].code)
        print(community_list[i].grade)
        count_grade_0 += 1
print('number of zero grade: ' + str(count_grade_0))

print('length of community_list: ' + str(len(community_list)))

all_price = []
for community in community_list:
    for price in community.price_time_series:
        all_price.append(price)

all_districts = set([])
for community in community_list:
    all_districts.add(community.district)

districts_price = dict()
for district in all_districts:
    districts_price[district] = []

for community in community_list:
    for price in community.price_time_series:
        districts_price[community.district].append(price)

community_location = dict()
for district in all_districts:
    community_location[district] = dict()
    community_location[district]['longitude'] = []
    community_location[district]['latitude'] = []
for community in community_list:
    community_location[community.district]['longitude'].append(community.longitude)
    community_location[community.district]['latitude'].append(community.latitude)

community_location_for_kmeans = []
for community in community_list:
    community_location_for_kmeans.append([community.longitude, community.latitude])
community_location_array_for_kmeans = np.array(community_location_for_kmeans)

for community in community_list:
    community.price_average = 0
    community.price_variance = 0
    for price in community.price_time_series:
        community.price_average += price
    community.price_average /= len(community.timeline)
    for price in community.price_time_series:
        community.price_variance += pow(price - community.price_average, 2)
    community.price_variance /= len(community.timeline)

community_time_series_statistics_for_kmeans = []
for community in community_list:
    community_time_series_statistics_for_kmeans.append([community.longitude, community.latitude, 
                                                        community.price_average, community.price_variance])
community_time_series_statistics_array_for_kmeans = np.array(community_time_series_statistics_for_kmeans)

community_two_rates_for_kmeans = []
for community in community_list:
    if community.greening_rate != None and community.volume_rate != None:
        community_two_rates_for_kmeans.append([community.longitude, community.latitude, 
                                               community.greening_rate, community.volume_rate])

community_two_rates_array_for_kmeans = np.array(community_two_rates_for_kmeans)

#max_diff = 0
#min_diff = 0
#
#for i in range(len(community_list)):
#    diff = community_list[i].price_time_series[23] - community_list[i].price_time_series[0]
#    if diff > max_diff:
#        max_diff = diff
#        max_diff_index = i
#    if diff < min_diff:
#        min_diff = diff
#        min_diff_index = i
#print('max_diff_index: ' + str(max_diff_index))
#print('max_diff: ' + str(max_diff))
#print('min_diff_index: ' + str(min_diff_index))
#print('min_diff: ' + str(min_diff))
#several_index = (0, 12, 19, 3256, 1359, 188)
#display_several_time_series(community_list, several_index)

#export_community_price_time_series(community_price_time_series_export_filepath, community_list)
#export_community_location(community_location_export_filepath, community_list)
#export_community_two_rates(community_two_rates_export_filepath, community_list)
#export_community_price_time_series_avg_and_var(community_price_time_series_avg_and_var_export_filepath, community_list)

#cal_probability_distributions(all_price, all_districts, districts_price)

#display_community_location(all_districts, community_location)

#display_community_grade_with_contour(community_list, grade_class_num)

#display_grade_price(community_list, grade_class_num, label_class[0])

#test()

#times = [0, 23]
#display_static_price_contour(community_list, times)
'''
#------------------location cluster------------------
#print('------------------location cluster------------------')
##for i in range(len(all_districts)):
##    title = u'南京市小区位置聚类图(n_clusters=' + str(i+1) + ')'
##    only_base_on_location = True
##    axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}]
##    cluster_kmeans(community_location_array_for_kmeans, i+1, only_base_on_location, title, axis_labels)
#title = u'小区位置聚类图'
#only_base_on_location = True
#axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}]
#cluster_kmeans(community_location_array_for_kmeans, nbclust_best_num['location'], only_base_on_location, title, axis_labels)

#------------------prcie time series statistics cluster------------------
#print('------------------prcie time series statistics cluster------------------')
#title = u'小区时间序列统计特征聚类图'
#only_base_on_location = False
#axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}, 
#               {'x': u'均价（元）', 'y': u'方差（元^2）'}]
#cluster_kmeans(community_time_series_statistics_array_for_kmeans, nbclust_best_num['avg_and_var'], only_base_on_location, title, axis_labels)

#------------------two rates cluster------------------
#print('------------------two rates cluster------------------')
#title = u'小区绿化率、容积率聚类图'
#only_base_on_location = False
#axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}, 
#               {'x': u'绿化率', 'y': u'容积率'}]
#cluster_kmeans(community_two_rates_array_for_kmeans, nbclust_best_num['two_rates'], only_base_on_location, title, axis_labels)

#------------------location Euclidean Cluster------------------
#print('------------------location Euclidean Cluster------------------')
##export_location_euclidean_distance_matrix(euclidean_dist_matrix_filepath, community_list)
#location_euclidean_distance_matrix = np.loadtxt(location_euclidean_dist_matrix_filepath)
#title = u'南京市小区位置欧氏距离层级聚类图'
#axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}]
#cluster_use_dist_matrix(5, location_euclidean_distance_matrix, community_location_for_kmeans, title, axis_labels)
'''

all_label_pred = dict()
for one_label in label_class:
    all_label_pred[one_label] = []

load_all_labels(all_labels_filepath, all_label_pred, label_class)

#display_all_label_individually(community_list, grade_class_num, all_label_pred, label_class, label_class_chinese, 'letter')
#display_all_label_individually(community_list, grade_class_num, all_label_pred, label_class, label_class_chinese, 'chinese')
display_all_price_time_series(community_list)

prove_my_guess(community_list)

'''
for community in community_list:
    all_label_pred[label_class[0]].append(int(community.grade) - 1)
#------------------price time series Euclidean Cluster------------------
print('------------------price time series Euclidean Cluster------------------')
#export_price_time_series_euclidean_distance_matrix(price_time_series_euclidean_dist_matrix_filepath, community_list)
price_time_series_euclidean_dist_matrix = np.loadtxt(price_time_series_euclidean_dist_matrix_filepath)
for linkage_type in linkage_types_tuple:
    label_pred_price_time_series_euclidean_cluster = cluster_use_dist_matrix_2(grade_class_num, price_time_series_euclidean_dist_matrix, 'euclidean', linkage_type, community_list)
    if linkage_type == 'average':
        all_label_pred[label_class[1]] = label_pred_price_time_series_euclidean_cluster
    elif linkage_type == 'complete':
        all_label_pred[label_class[2]] = label_pred_price_time_series_euclidean_cluster
    for i in range(len(community_list)):
        community_list[i].price_time_series_euclidean_cluster_label[linkage_type] = label_pred_price_time_series_euclidean_cluster[i] + 1
    cal_cluster_class_avg_prcie_time_series(community_list, grade_class_num, 'euclidean', linkage_type, label_class_chinese)

#------------------price time series Pearson Cluster------------------
print('------------------price time series Pearson Cluster------------------')
#export_price_time_series_pearson_coefficient_matrix(price_time_series_pearson_coefficient_matrix_filepath, community_list)
price_time_series_pearson_coefficient_matrix = np.loadtxt(price_time_series_pearson_coefficient_matrix_filepath)
for linkage_type in linkage_types_tuple:
    label_pred_price_time_series_pearson_cluster = cluster_use_dist_matrix_2(grade_class_num, price_time_series_pearson_coefficient_matrix, 'pearson', linkage_type, community_list)
    if linkage_type == 'average':
        all_label_pred[label_class[3]] = label_pred_price_time_series_pearson_cluster
    elif linkage_type == 'complete':
        all_label_pred[label_class[4]] = label_pred_price_time_series_pearson_cluster
    for i in range(len(community_list)):
        community_list[i].price_time_series_pearson_cluster_label[linkage_type] = label_pred_price_time_series_pearson_cluster[i] + 1
    cal_cluster_class_avg_prcie_time_series(community_list, grade_class_num, 'pearson', linkage_type, label_class_chinese)

#------------------price time series dtw Cluster------------------
print('------------------price time series dtw Cluster------------------')
dtw_dist_matrix = np.loadtxt(dtw_dist_matrix_filepath)
for linkage_type in linkage_types_tuple:
    label_pred_price_time_series_dtw_cluster = cluster_use_dist_matrix_2(grade_class_num, dtw_dist_matrix, 'dtw', linkage_type, community_list)
    if linkage_type == 'average':
        all_label_pred[label_class[5]] = label_pred_price_time_series_dtw_cluster
    elif linkage_type == 'complete':
        all_label_pred[label_class[6]] = label_pred_price_time_series_dtw_cluster
    for i in range(len(community_list)):
        community_list[i].price_time_series_dtw_cluster_label[linkage_type] = label_pred_price_time_series_dtw_cluster[i] + 1
    cal_cluster_class_avg_prcie_time_series(community_list, grade_class_num, 'dtw', linkage_type, label_class_chinese)

#export_all_labels(all_labels_filepath, all_label_pred, label_class)
'''

#------------------cal label_number------------------
#print('------------------cal label_number------------------')
#label_number = dict()
#label_class = ('grade', 'ts_euclidean_average', 'ts_euclidean_complete', 'ts_pearson_average', 'ts_pearson_complete', 'ts_dtw_average', 'ts_dtw_complete')
#label_level_num = 5
#
#for one_label_class in label_class:
#    label_number[one_label_class] = dict()
#    for i in range(grade_class_num):
#        label_number[one_label_class][str(i + 1)] = 0
#for community in community_list:
#    label_number['grade'][str(community.grade)] += 1
#    label_number['ts_euclidean_average'][str(community.price_time_series_euclidean_cluster_label['average'])] += 1
#    label_number['ts_euclidean_complete'][str(community.price_time_series_euclidean_cluster_label['complete'])] += 1
#    label_number['ts_pearson_average'][str(community.price_time_series_pearson_cluster_label['average'])] += 1
#    label_number['ts_pearson_complete'][str(community.price_time_series_pearson_cluster_label['complete'])] += 1
#    label_number['ts_dtw_average'][str(community.price_time_series_dtw_cluster_label['average'])] += 1
#    label_number['ts_dtw_complete'][str(community.price_time_series_dtw_cluster_label['complete'])] += 1
#
#print(label_number)



#price_class = classify(community_list, 24 - 1)
#for i in range(len(community_list)):
#    community_list[i].price_class_label = price_class['category'][i]

#------------------export community------------------
#print('------------------export community------------------')
#export_community(community_export_filepath, community_list)

'''
#------------------export numeric dataset------------------
print('------------------export numeric dataset------------------')
export_header = ('code,longitude,latitude,architectural_age,volume_rate,greening_rate,property_costs,'
#                 'price_1,price_2,price_3,price_4,price_5,price_6,price_7,price_8,price_9,price_10,'
#                 'price_11,price_12,price_13,price_14,price_15,price_16,price_17,price_18,price_19,price_20,'
#                 'price_21,price_22,price_23,price_24,'
                 'ts_euclidean_label_average,ts_euclidean_label_complete,'
                 'ts_pearson_label_average,ts_pearson_label_complete,'
                 'ts_dtw_label_average,ts_dtw_label_complete,'
                 'grade,')
obervation_number = len(community_list)
field_number = export_header.count(',') + 1
dataset = np.recarray(obervation_number, dtype=
                      [('code',np.uint32),
                       ('longitude',np.float64),
                       ('latitude',np.float64),
                       ('architectural_age',np.uint8),
                       ('volume_rate',np.float16),
                       ('greening_rate',np.float16),
                       ('property_costs',np.float16),
#                       ('price_time_series',np.uint32,24),
                       ('ts_euclidean_label_average',np.uint8),
                       ('ts_euclidean_label_complete',np.uint8),
                       ('ts_pearson_label_average',np.uint8),
                       ('ts_pearson_label_complete',np.uint8),
                       ('ts_dtw_label_average',np.uint8),
                       ('ts_dtw_label_complete',np.uint8),
                       ('grade',np.uint8),
                       ])

for i in range(obervation_number):
    dataset['code'][i] = community_list[i].code
    dataset['longitude'][i] = community_list[i].longitude
    dataset['latitude'][i] = community_list[i].latitude
    dataset['architectural_age'][i] = community_list[i].architectural_age if community_list[i].architectural_age != None else invalid_value
    dataset['volume_rate'][i] = community_list[i].volume_rate if community_list[i].volume_rate != None else invalid_value
    dataset['greening_rate'][i] = community_list[i].greening_rate if community_list[i].greening_rate != None else invalid_value
    dataset['property_costs'][i] = community_list[i].property_costs if community_list[i].property_costs != None else invalid_value
#    for j in range(len(community_list[0].timeline)):
#        dataset['price_time_series'][i][j] = community_list[i].price_time_series[j]
    dataset['ts_euclidean_label_average'][i] = community_list[i].price_time_series_euclidean_cluster_label['average']
    dataset['ts_euclidean_label_complete'][i] = community_list[i].price_time_series_euclidean_cluster_label['complete']
    dataset['ts_pearson_label_average'][i] = community_list[i].price_time_series_pearson_cluster_label['average']
    dataset['ts_pearson_label_complete'][i] = community_list[i].price_time_series_pearson_cluster_label['complete']
    dataset['ts_dtw_label_average'][i] = community_list[i].price_time_series_dtw_cluster_label['average']
    dataset['ts_dtw_label_complete'][i] = community_list[i].price_time_series_dtw_cluster_label['complete']
    dataset['grade'][i] = community_list[i].grade

np.savetxt(work_space_analysis_str + "numeric.csv", dataset, delimiter=",", header=export_header, comments='')
np.savetxt(work_space_analysis_str + "all_code.txt", dataset['code'])
np.savetxt(work_space_analysis_str + "all_longitude.txt", dataset['longitude'])
np.savetxt(work_space_analysis_str + "all_latitude.txt", dataset['latitude'])
np.savetxt(work_space_analysis_str + "all_architectural_age.txt", dataset['architectural_age'])
np.savetxt(work_space_analysis_str + "all_volume_rate.txt", dataset['volume_rate'])
np.savetxt(work_space_analysis_str + "all_greening_rate.txt", dataset['greening_rate'])
np.savetxt(work_space_analysis_str + "all_property_costs.txt", dataset['property_costs'])
np.savetxt(work_space_analysis_str + "ts_euclidean_label_average.txt", dataset['ts_euclidean_label_average'])
np.savetxt(work_space_analysis_str + "ts_euclidean_label_complete.txt", dataset['ts_euclidean_label_complete'])
np.savetxt(work_space_analysis_str + "ts_pearson_label_average.txt", dataset['ts_pearson_label_average'])
np.savetxt(work_space_analysis_str + "ts_pearson_label_complete.txt", dataset['ts_pearson_label_complete'])
np.savetxt(work_space_analysis_str + "ts_dtw_label_average.txt", dataset['ts_dtw_label_average'])
np.savetxt(work_space_analysis_str + "ts_dtw_label_complete.txt", dataset['ts_dtw_label_complete'])
np.savetxt(work_space_analysis_str + "all_grade.txt", dataset['grade'])
'''
