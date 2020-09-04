import csv
import re
import numpy as np

def load_all_labels(all_labels_filepath, all_label_pred, label_class):
    all_labels_array = np.loadtxt(all_labels_filepath, dtype=int)
    nrow, ncol = all_labels_array.shape
    for i in range(len(label_class)):
        for j in range(nrow):
            all_label_pred[label_class[i]].append(all_labels_array[j][i])

class community_grade(object):
    def __init__(self, csv_row):
        self.code = csv_row[0]
        self.grade = csv_row[1]

class community_info(object):
    timeline = {'2016-04', '2016-05', '2016-06', '2016-07', '2016-08', '2016-09', 
                '2016-10', '2016-11', '2016-12', '2017-01', '2017-02', '2017-03', 
                '2017-04', '2017-05', '2017-06', '2017-07', '2017-08', '2017-09', 
                '2017-10', '2017-11', '2017-12', '2018-01', '2018-02', '2018-03',}
    
    def __init__(self, csv_row):
        self.grade = 0
        self.price_time_series_euclidean_cluster_label = dict()
        self.price_time_series_pearson_cluster_label = dict()
        self.price_time_series_dtw_cluster_label = dict()
        for linkage_type in ('average', 'complete'):
            self.price_time_series_euclidean_cluster_label[linkage_type] = 0
            self.price_time_series_pearson_cluster_label[linkage_type] = 0
            self.price_time_series_dtw_cluster_label[linkage_type] = 0
        
        self.address = re.split('（', csv_row[0])[0]
        
        if len(csv_row[1]) != 0:
            self.architectural_age = 2018 - int(csv_row[1])
        else:
            self.architectural_age = None
        
        self.code = csv_row[2]
        
        if len(csv_row[3]) != 0:
            self.construction_area = int(csv_row[3])
        else:
            self.construction_area = None
            
        self.district = csv_row[4]
        
        if len(csv_row[5]) != 0:
            self.floor_area = int(csv_row[5])
        else:
            self.floor_area = None
            
        if len(csv_row[6]) != 0:
            self.greening_rate = float(re.match('\d+', csv_row[6]).group(0)) / 100
        else:
            self.greening_rate = None
        
        self.latitude = float(csv_row[7])
        self.longitude = float(csv_row[8])
        self.name = csv_row[9].replace(' ', '')
        self.price_time_series = []
        for one_month_price in re.split(",", csv_row[10]):
            self.price_time_series.append(int(one_month_price))
        
        if len(csv_row[11]) != 0:
            self.property_costs = float(csv_row[11])
        else:
            self.property_costs = None
            
        self.region = csv_row[12]
        
        if len(csv_row[13]) != 0:
            self.volume_rate = float(csv_row[13])
        elif self.construction_area != None and self.floor_area != None:
            self.volume_rate = float(self.construction_area) / float(self.floor_area)
        else:
            self.volume_rate = None
    
    pass

def whether_keep_community_info_csv_row(row):
    #---cleaning---
    #   1 address
    if(len(row[0]) == 0):
        return False
    #   2 architectural_age
    '''
    if(len(row[1]) < 4):
        return False
    '''
    #   3 code
    if(len(row[2]) == 0):
        return False
    #   4 construction_area
    '''
    if(len(row[3]) == 0):
        return False
    '''
    #   5 district
    if(len(row[4]) == 0):
        return False
    #   6 floor_area
    '''
    if(len(row[5]) == 0):
        return False
    '''
    #   7 greening_rate
    '''
    if(len(row[6]) == 0):
        return False
    '''
    #   8 latitude
    if(len(row[7]) < 4):
        return False
    #   9 longitude
    if(len(row[8]) < 4):
        return False
    #   10 name
    if(len(row[9]) == 0):
        return False
    #   11 price_time_series
    if(len(row[10]) == 0):
        return False
    #   12 property_costs
    '''
    if(len(row[11]) == 0):
        return False
    '''
    #   13 region
    if(len(row[12]) == 0):
        return False
    #   14 volume_rate
    '''
    if(len(row[13]) == 0):
        return False
    '''
    return True

def whether_csv_row_has_been_inserted(community_list, one_community):
    for record in community_list:
        if(record.code == one_community.code):
            return True
    return False

def whether_price_is_not_positive(one_community):
    for price in one_community.price_time_series:
        if price <= 0:
            return True
    return False

def import_community_info_from_csv(file_path_str, community_list):
    with open(file_path_str, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        row_num = 0
        for row in reader:
            if row_num == 0:
                header = row
                row_num += 1
            else:
                row_num += 1
                if len(row) > 0:
                    if(whether_keep_community_info_csv_row(row) == False):
                        continue
                    one_community = community_info(row)
                    if(len(one_community.price_time_series) != len(one_community.timeline)):
                        continue
                    if(whether_csv_row_has_been_inserted(community_list, one_community) == True):
                        continue
                    if(whether_price_is_not_positive(one_community)):
                        continue
                    community_list.append(one_community)
    return header

def whether_keep_community_grade_csv_row(row):
    #---cleaning---
    #   1 code
    if len(row[0]) == 0:
        return False
    #   2 grade
    if len(row[1]) == 0:
        return False
    if int(row[1]) <= 0 or int(row[1]) > 5:
        return False
    
    return True

def whether_grade_csv_row_has_been_inserted(community_grade_list, one_community_grade):
    for record in community_grade_list:
        if record.code == one_community_grade.code:
            return True
    return False

def import_community_grade_from_csv(file_path_str, community_grade_list):
    with open(file_path_str, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        row_num = 0
        for row in reader:
            if row_num == 0:
                header = row
                row_num += 1
            else:
                row_num += 1
                if len(row) > 0:
                    if whether_keep_community_grade_csv_row(row) == False:
                        continue
                    one_community_grade = community_grade(row)
                    if whether_grade_csv_row_has_been_inserted(community_grade_list, one_community_grade) == True:
                        continue
                    community_grade_list.append(one_community_grade)
    return header