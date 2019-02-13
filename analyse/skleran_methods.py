# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:03:47 2018

@author: Thomas Anderson
"""
#import matplotlib
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import kde
import matplotlib.cm as cm

def cluster_use_dist_matrix_2(cluster_number, dtw_dist_matrix, distance_type, linkage_type, 
                                community_list):
    title = u'时间序列聚类分布图(' + distance_type + ','+ linkage_type + ')'
    all_class_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
    agg = AgglomerativeClustering(n_clusters=cluster_number, affinity='precomputed', linkage=linkage_type)
    agg.fit_predict(dtw_dist_matrix)  # Returns class labels.
    label_pred = agg.labels_
    '''
    Euclidean_average:12345
    Euclidean_complete:42315
    
    Pearson_average:51423
    Pearson_complete:12543
    
    DTW_average:13425
    DTW_complete:12543
    '''
    all_distance_types = ('euclidean', 'pearson', 'dtw')
    all_linkage_types = ('average', 'complete')
    all_rematch_strs = ('12345', '42315', '51423', '12543', '13425', '12543')
    rematch = dict()
    count = 0
    for i in all_distance_types:
        rematch[i] = dict()
        for j in all_linkage_types:
            rematch[i][j] = dict()
            for k in range(cluster_number):
                rematch[i][j][str(k)] = int(all_rematch_strs[count][k]) - 1
            count += 1
    
    for i in range(len(label_pred)):
        label_pred[i] = rematch[distance_type][linkage_type][str(label_pred[i])]
    
    clustered_community = []
    for i in range(cluster_number):
        temp = dict()
        temp['all_longitude'] = []
        temp['all_latitude'] = []
        clustered_community.append(temp)
    count = 0
    for label in label_pred:
        clustered_community[label]['all_longitude'].append(community_list[count].longitude)
        clustered_community[label]['all_latitude'].append(community_list[count].latitude)
        count += 1
    
    colors = cm.rainbow(np.linspace(0, 1, cluster_number))
    
    fig, axes = plt.subplots(ncols=cluster_number, nrows=2, figsize=(15, 12))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(cluster_number):
        x0 = clustered_community[i]['all_longitude']
        y0 = clustered_community[i]['all_latitude']
        
        data = np.ndarray(shape=(len(x0), 2), dtype=float)
        for j in range(len(x0)):
        	data[j][0] = x0[j]
        	data[j][1] = y0[j]
        
        x, y = data.T
        
        axes[0][i].set_title(all_class_labels[i] + u'小区(' + str(len(x)) + '个)')
    
        if len(x0) > 10:
            k = kde.gaussian_kde(data.T)
            xi, yi = np.mgrid[xlim_min:xlim_max:nbins*1j, ylim_min:ylim_max:nbins*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            axes[0][i].contourf(xi, yi, zi.reshape(xi.shape), 8, alpha=.75, cmap=plt.cm.rainbow)
            axes[0][i].contour(xi, yi, zi.reshape(xi.shape), colors='black', linewidths=.5)
        
        axes[0][i].set_xlim((xlim_min, xlim_max))
        axes[0][i].set_ylim((ylim_min, ylim_max))
        axes[0][i].set_xlabel(u'经度（°，E）')
        axes[0][i].set_ylabel(u'纬度（°，N）')
        
        axes[1][i].set_title(all_class_labels[i] + u'小区(' + str(len(x)) + '个)')
        axes[1][i].scatter(x, y, color=colors[i], s=0.75)
        axes[1][i].set_xlim((xlim_min, xlim_max))
        axes[1][i].set_ylim((ylim_min, ylim_max))
        axes[1][i].set_xlabel(u'经度（°，E）')
        axes[1][i].set_ylabel(u'纬度（°，N）')
    
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)
    
    return label_pred

def cluster_use_dist_matrix(cluster_number, dtw_dist_matrix, distance_type, linkage_type, 
                                all_community_location_array):
    title = u'时间序列聚类分布图(' + distance_type + ','+ linkage_type + ')'
    axis_labels = [{'x': u'经度（°，E）', 'y': u'纬度（°，N）'}]
    agg = AgglomerativeClustering(n_clusters=cluster_number, affinity='precomputed', linkage=linkage_type)
    agg.fit_predict(dtw_dist_matrix)  # Returns class labels.
    label_pred = agg.labels_
    
    lis = range(cluster_number)
    classes = ['{:d}'.format(x+1) for x in lis]
    clustered_community = dict()
    for one_class in classes:
        clustered_community[one_class] = []
    count = 0
    for label in label_pred:
        clustered_community[classes[label]].append(all_community_location_array[count])
        count += 1
    for one_class in classes:
        clustered_community[one_class] = np.array(clustered_community[one_class])
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 4.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    ax = plt.gca()
    ax.set_aspect(1)
    colors = cm.gist_rainbow(np.linspace(0, 1, cluster_number))
    color_index = 0
    handles = []
    labels = []
    for one_class in classes:
        one_label = plt.scatter(clustered_community[one_class][:, 0], 
                                clustered_community[one_class][:, 1], 
                                color=colors[color_index], s=0.1, marker='.')
        handles.append(one_label)
        labels.append(one_class)
        color_index += 1
    plt.xlim((118.25, 119.5))
    plt.xlabel(axis_labels[0]['x'])
    plt.ylabel(axis_labels[0]['y'])
    plt.legend(handles = handles, labels = labels, loc = 'best')
    
    plt.tight_layout()
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)
    
    return label_pred

def cluster_kmeans(all_community_location_array, cluster_number, only_base_on_location, title, axis_labels):
    if only_base_on_location:
        label_pred = KMeans(n_clusters=cluster_number).fit_predict(all_community_location_array)
    else:
        label_pred = KMeans(n_clusters=cluster_number).fit_predict(all_community_location_array[:, 2:4])
    
    lis = range(cluster_number)
    classes = ['{:d}'.format(x+1) for x in lis]
    clustered_community = dict()
    for one_class in classes:
        clustered_community[one_class] = []
    count = 0
    for label in label_pred:
        clustered_community[classes[label]].append(all_community_location_array[count])
        count += 1
    for one_class in classes:
        clustered_community[one_class] = np.array(clustered_community[one_class])

    ax = plt.gca()
    ax.set_aspect(1)
    if cluster_number > 2:
        colors = cm.gist_rainbow(np.linspace(0, 1, cluster_number))
    else:
        colors = cm.gist_rainbow(np.linspace(0, 1, 2 * cluster_number))
    if only_base_on_location == False:
        plt.subplot(121)
    color_index = 0
    handles = []
    labels = []
    for one_class in classes:
        one_label = plt.scatter(clustered_community[one_class][:, 0], 
                                clustered_community[one_class][:, 1], 
                                color=colors[color_index], s=0.3)
        handles.append(one_label)
        labels.append(one_class)
        color_index += 1
    
    plt.xlim((118.25, 119.5))
    plt.xlabel(axis_labels[0]['x'])
    plt.ylabel(axis_labels[0]['y'])
    plt.legend(handles = handles, labels = labels, loc = 'best')
    if only_base_on_location == False:
        plt.subplot(122)
        color_index = 0
        handles = []
        labels = []
        for one_class in classes:
            one_label = plt.scatter(clustered_community[one_class][:, 2], 
                                    clustered_community[one_class][:, 3], 
                                    color=colors[color_index], s=0.3)
            handles.append(one_label)
            labels.append(one_class)
            color_index += 1
        plt.xlabel(axis_labels[1]['x'])
        plt.ylabel(axis_labels[1]['y'])
        plt.legend(handles = handles, labels = labels, loc = 'best')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.885)
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)
    
    return label_pred