#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
from scipy.stats import kde
from mpl_toolkits.mplot3d import axes3d

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

def prove_my_guess(community_list):
    total_diff = 0
    up_count = 0
    down_count = 0
    num = len(community_list)
    for i in range(num):
        diff = community_list[i].price_time_series[19] - community_list[i].price_time_series[18]
        total_diff += diff
        if diff > 0:
            up_count += 1
        else:
            down_count += 1
    
    avg_diff = total_diff / num
    print('total_diff: ' + str(total_diff))
    print('avg_diff: ' + str(avg_diff))
    print('up_count: ' + str(up_count))
    print('down_count: ' + str(down_count))

def display_all_price_time_series(community_list):
    num_community = len(community_list)
    len_time_series = len(community_list[0].timeline)
    
    all_price_time_series = []
    for i in range(num_community):
        all_price_time_series.append(community_list[i].price_time_series)

    dates = []
    
    for year in range(2016, 2019):
        for month in range(1, 13):
            dates.append(dt.datetime(year=year, month=month, day=1))
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    dates = dates[3:27]
    file_title = u'所有住宅小区时间序列图'

    x_label = u'日期'
    y_label = u'均价（元）'
    all_cluster_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(num_community):
        plt.plot(dates, all_price_time_series[i])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   #to get a tick every 4 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))     #optional formatting 
    ax.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(u'时间序列图\\'+file_title+'.png', dpi=500)
    plt.show()

def display_all_label_individually(community_list, cluster_num, all_label_pred, label_class, label_class_chinese, suptitle_type):
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
    colors = cm.rainbow(np.linspace(0, 1, cluster_num))
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    all_class_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    plot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
    
    plt.rcParams.update({'font.size': 12})
    
    clusters = dict()
    for one_label in label_class:
        clusters[one_label] = []
        for i in range(cluster_num):
            temp = dict()
            temp['all_longitude'] = []
            temp['all_latitude'] = []
            clusters[one_label].append(temp)
        for i in range(len(all_label_pred[one_label])):
            clusters[one_label][all_label_pred[one_label][i]]['all_longitude'].append(community_list[i].longitude)
            clusters[one_label][all_label_pred[one_label][i]]['all_latitude'].append(community_list[i].latitude)
    
    
    nbins = 50
    
    for i in range(len(label_class)):
        fig, axes = plt.subplots(ncols=cluster_num, nrows=1, figsize=(15, 6))
        plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.4)
        for j in range(cluster_num):
            x0 = clusters[label_class[i]][j]['all_longitude']
            y0 = clusters[label_class[i]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            axes[j].scatter(x, y, color=colors[j], s=1.5)
            axes[j].set_xlim((xlim_min, xlim_max))
            axes[j].set_ylim((ylim_min, ylim_max))
            axes[j].set_xlabel(u'经度（°，E）')
            axes[j].set_ylabel(u'纬度（°，N）')
            file_title = label_class_chinese[i]
            if suptitle_type == 'letter':
                plot_title = plot_labels[i]
                plt.suptitle(plot_title, fontsize=35, x=0.05, weight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            elif suptitle_type == 'chinese':
                plot_title = label_class_chinese[i]
                plt.suptitle(plot_title, fontsize=35, x=0.5, weight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            plt.savefig(u'聚类图\\空间分布散点图\\空间分布散点图('+file_title+').png', dpi=500)
    
    for i in range(len(label_class)):
        fig, axes = plt.subplots(ncols=cluster_num, nrows=1, figsize=(15, 6))
        plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.9, wspace=0.4)
        for j in range(cluster_num):
            x0 = clusters[label_class[i]][j]['all_longitude']
            y0 = clusters[label_class[i]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            if len(x0) > 10:
                kernel = kde.gaussian_kde(data.T)
                xi, yi = np.mgrid[xlim_min:xlim_max:nbins*1j, ylim_min:ylim_max:nbins*1j]
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
                axes[j].contourf(xi, yi, zi.reshape(xi.shape), 24, alpha=.75, cmap=plt.cm.rainbow)
#                axes[i][j].contour(xi, yi, zi.reshape(xi.shape), colors='black', linewidths=.5)
            axes[j].set_xlim((xlim_min, xlim_max))
            axes[j].set_ylim((ylim_min, ylim_max))
            axes[j].set_xlabel(u'经度（°，E）')
            axes[j].set_ylabel(u'纬度（°，N）')
            file_title = label_class_chinese[i]
            if suptitle_type == 'letter':
                plot_title = plot_labels[i]
                plt.suptitle(plot_title, fontsize=35, x=0.05, weight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            elif suptitle_type == 'chinese':
                plot_title = label_class_chinese[i]
                plt.suptitle(plot_title, fontsize=35, x=0.5, weight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.90])
            plt.savefig(u'聚类图\\空间分布密度图\\空间分布密度图('+file_title+').png', dpi=500)
    

def display_all_label_in_two_times(community_list, cluster_num, all_label_pred, label_class):
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
    colors = cm.rainbow(np.linspace(0, 1, cluster_num))
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    all_class_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    
    clusters = dict()
    for one_label in label_class:
        clusters[one_label] = []
        for i in range(cluster_num):
            temp = dict()
            temp['all_longitude'] = []
            temp['all_latitude'] = []
            clusters[one_label].append(temp)
        for i in range(len(all_label_pred[one_label])):
            clusters[one_label][all_label_pred[one_label][i]]['all_longitude'].append(community_list[i].longitude)
            clusters[one_label][all_label_pred[one_label][i]]['all_latitude'].append(community_list[i].latitude)
    
    ncols = cluster_num
    nrows = 1 + (len(label_class) - 1) / 2
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 32))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    #------------------average linkage------------------

    for i in range(nrows):
        for j in range(ncols):
            if i == 0:
                x0 = clusters[label_class[i]][j]['all_longitude']
                y0 = clusters[label_class[i]][j]['all_latitude']
            else:
                x0 = clusters[label_class[2 * i - 1]][j]['all_longitude']
                y0 = clusters[label_class[2 * i - 1]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[i][j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[i][j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            axes[i][j].scatter(x, y, color=colors[j], s=1.5)
            axes[i][j].set_xlim((xlim_min, xlim_max))
            axes[i][j].set_ylim((ylim_min, ylim_max))
            axes[i][j].set_xlabel(u'经度（°，E）')
            axes[i][j].set_ylabel(u'纬度（°，N）')
    
    title = u'空间分布散点图'
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)
    
    
    
    fig, axes = plt.subplots(ncols=cluster_num, nrows=len(label_class), figsize=(15, 55))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(len(label_class)):
        for j in range(cluster_num):
            x0 = clusters[label_class[i]][j]['all_longitude']
            y0 = clusters[label_class[i]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[i][j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[i][j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            if len(x0) > 10:
                kernel = kde.gaussian_kde(data.T)
                xi, yi = np.mgrid[xlim_min:xlim_max:nbins*1j, ylim_min:ylim_max:nbins*1j]
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
                axes[i][j].contourf(xi, yi, zi.reshape(xi.shape), 24, alpha=.75, cmap=plt.cm.rainbow)
#                axes[i][j].contour(xi, yi, zi.reshape(xi.shape), colors='black', linewidths=.5)
            axes[i][j].set_xlim((xlim_min, xlim_max))
            axes[i][j].set_ylim((ylim_min, ylim_max))
            axes[i][j].set_xlabel(u'经度（°，E）')
            axes[i][j].set_ylabel(u'纬度（°，N）')
    
    title = u'空间分布密度图'
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)

def display_all_label(community_list, cluster_num, all_label_pred, label_class):
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
    colors = cm.rainbow(np.linspace(0, 1, cluster_num))
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    all_class_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    
    clusters = dict()
    for one_label in label_class:
        clusters[one_label] = []
        for i in range(cluster_num):
            temp = dict()
            temp['all_longitude'] = []
            temp['all_latitude'] = []
            clusters[one_label].append(temp)
        for i in range(len(all_label_pred[one_label])):
            clusters[one_label][all_label_pred[one_label][i]]['all_longitude'].append(community_list[i].longitude)
            clusters[one_label][all_label_pred[one_label][i]]['all_latitude'].append(community_list[i].latitude)
    
    fig, axes = plt.subplots(ncols=cluster_num, nrows=len(label_class), figsize=(15, 55))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(len(label_class)):
        for j in range(cluster_num):
            x0 = clusters[label_class[i]][j]['all_longitude']
            y0 = clusters[label_class[i]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[i][j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[i][j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            axes[i][j].scatter(x, y, color=colors[j], s=1.5)
            axes[i][j].set_xlim((xlim_min, xlim_max))
            axes[i][j].set_ylim((ylim_min, ylim_max))
            axes[i][j].set_xlabel(u'经度（°，E）')
            axes[i][j].set_ylabel(u'纬度（°，N）')
    
    title = u'空间分布散点图(合并)'
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)
    
    
    
    fig, axes = plt.subplots(ncols=cluster_num, nrows=len(label_class), figsize=(15, 55))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(len(label_class)):
        for j in range(cluster_num):
            x0 = clusters[label_class[i]][j]['all_longitude']
            y0 = clusters[label_class[i]][j]['all_latitude']
            
            data = np.ndarray(shape=(len(x0), 2), dtype=float)
            for k in range(len(x0)):
            	data[k][0] = x0[k]
            	data[k][1] = y0[k]
            
            x, y = data.T
            
            if i == 0:
                axes[i][j].set_title(all_grade_labels[j] + u'小区(' + str(len(x)) + '个)')
            else:
                axes[i][j].set_title(all_class_labels[j] + u'小区(' + str(len(x)) + '个)')
            if len(x0) > 10:
                kernel = kde.gaussian_kde(data.T)
                xi, yi = np.mgrid[xlim_min:xlim_max:nbins*1j, ylim_min:ylim_max:nbins*1j]
                zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))
                axes[i][j].contourf(xi, yi, zi.reshape(xi.shape), 24, alpha=.75, cmap=plt.cm.rainbow)
#                axes[i][j].contour(xi, yi, zi.reshape(xi.shape), colors='black', linewidths=.5)
            axes[i][j].set_xlim((xlim_min, xlim_max))
            axes[i][j].set_ylim((ylim_min, ylim_max))
            axes[i][j].set_xlabel(u'经度（°，E）')
            axes[i][j].set_ylabel(u'纬度（°，N）')
    
    title = u'空间分布密度图(合并)'
    plt.savefig(u'聚类图\\'+title+'.png', dpi=500)

def test():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    print('X:')
    print(X)
    print('Y:')
    print(Y)
    print('Z:')
    print(Z)
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    
    ax.set_xlabel('X')
    ax.set_xlim(-40, 40)
    ax.set_ylabel('Y')
    ax.set_ylim(-40, 40)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)
    
    plt.show()

def display_static_price_contour(community_list, times):
    all_lon = []
    all_lat = []
    all_price = []
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
        
    for i in range(len(times)):
        one_price = []
        all_price.append(one_price)
    for community in community_list:
        all_lon.append(community.longitude)
        all_lat.append(community.latitude)
        for i in range(len(times)):
            all_price[i].append(community.price_time_series[times[i]])
    
    fig, axes = plt.subplots(ncols=len(times), nrows=1, figsize=(9, 5))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(len(times)):
        x0 = all_lon
        y0 = all_lat
        
        data = np.ndarray(shape=(len(x0), 2), dtype=float)
        for j in range(len(x0)):
        	data[j][0] = x0[j]
        	data[j][1] = y0[j]
        
        x, y = data.T
        
        xi, yi = np.meshgrid(x, y)
        zi = all_price[i]
        
#        axes[i].set_title(all_grade_labels[i] + u'小区(' + str(len(x)) + '个)')
        axes[i].contourf(xi, yi, zi, 8, alpha=.75, cmap=plt.cm.rainbow)
        axes[i].contour(xi, yi, zi, colors='black', linewidths=.5)
        
        axes[i].set_xlim((xlim_min, xlim_max))
        axes[i].set_ylim((ylim_min, ylim_max))
        axes[i].set_xlabel(u'经度（°，E）')
        axes[i].set_ylabel(u'纬度（°，N）')
        
    title = u'静态分析图'
    plt.savefig(u'其他分析图\\'+title+'.png', dpi=500)
    
def show_label_number_distribution_2(label_number, label_class, label_level_num):
#    label_class = ('grade', 'ts_euclidean_average', 'ts_euclidean_complete', 
#                  'ts_pearson_average', 'ts_pearson_complete', 'ts_dtw_average', 'ts_dtw_complete')
    name_list = (u'类1', u'类2', u'类3', u'类4', u'类5')
    color_tuple = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    all_num_list = []
        
    for i in range(label_level_num):
        num_list = []
        for one_label_class in label_class:
            num_list.append(label_number[one_label_class][str(i+1)])
        all_num_list.append(num_list)
    
    fig, ax = plt.subplots()
    x = np.arange(len(label_class))
    total_width=  0.75
    n = label_level_num
    width = total_width / n
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    for i in range(label_level_num):
        plt.barh(x + i * width, all_num_list[i], width, color=color_tuple[i], label=name_list[i])
        for j, v in enumerate(all_num_list[i]):
            ax.text(v + 30, j+ i * width, str(v), verticalalignment='center', fontsize=7, color=color_tuple[i], fontweight='bold')
    title = u'样本类间分布图'
    plt.xlim((0, 4000))
    ax.set_xlabel(u'样本数')
    ax.set_ylabel(u'聚类依据')
    ax.set_yticks(x + float(label_level_num) * width / 2)
    ax.set_yticklabels(label_class)
    ax.legend(loc='best')
    
    fig.tight_layout()
    plt.savefig(u'其他分析图\\'+title+'.png', dpi=500)
    plt.show()

def show_label_number_distribution(label_number, label_class, label_level_num):
#    label_class = ('grade', 'ts_euclidean_average', 'ts_euclidean_complete', 
#                  'ts_pearson_average', 'ts_pearson_complete', 'ts_dtw_average', 'ts_dtw_complete')
    name_list = [u'类一', u'类二', u'类三', u'类四', u'类五']
    color_tuple = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    all_num_list = []
    for one_label_class in label_class:
        num_list = []
        for i in range(label_level_num):
            num_list.append(label_number[one_label_class][str(i+1)])
        all_num_list.append(num_list)
    
    fig, ax = plt.subplots()
    x = np.arange(label_level_num)
    total_width=  0.8
    n = len(label_class)
    width = total_width / n
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 8.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    for i in range(len(label_class)):
        plt.barh(x + i * width, all_num_list[i], width, color=color_tuple[i], label=label_class[i])
        for j, v in enumerate(all_num_list[i]):
            ax.text(v + 30, j+ i * width, str(v), verticalalignment='center', fontsize=8, color=color_tuple[i], fontweight='bold')
    title = u'样本类间分布图'
    plt.xlim((0, 4000))
    ax.set_xlabel(u'样本数')
    ax.set_ylabel(u'分类')
    ax.set_yticks(x + float(len(label_class)) * width / 2)
    ax.set_yticklabels((u'类1', u'类2', u'类3', u'类4', u'类5'))
    ax.legend(loc='best')
    
    fig.tight_layout()
    plt.savefig(u'其他分析图\\'+title+'.png', dpi=500)
    plt.show()
    
def display_several_time_series(community_list, several_index):
    all_price_time_series = []
    all_community_name = []
    ts_len = len(community_list[0].timeline)
    for index in several_index:
        temp = []
        for i in range(ts_len):
            temp.append(community_list[index].price_time_series[i])
        all_price_time_series.append(temp)
        all_community_name.append(community_list[index].name)
    
    dates = []
    
    for year in range(2016, 2019):
        for month in range(1, 13):
            dates.append(dt.datetime(year=year, month=month, day=1))
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    dates = dates[3:27]
    file_title = u'小区均价时间序列示例'
    x_label = u'日期'
    y_label = u'均价（元）'
    handles = []
    labels = []
    colors = cm.rainbow(np.linspace(0, 1, len(several_index)))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(len(several_index)):
        one_label, = plt.plot(dates, all_price_time_series[i], color=colors[i])
        handles.append(one_label)
        labels.append(all_community_name[i])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   #to get a tick every 4 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))     #optional formatting 
    ax.grid(linestyle='--', linewidth=0.5)
    plt.legend(handles = handles, labels = labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(u'时间序列图\\'+file_title+'.png', dpi=500)
    plt.show()

def display_grade_price(community_list, grade_num, plot_title):
    community_num = len(community_list)
    time_series_length = len(community_list[0].timeline)
    all_grades_avg_price = []
    all_grade_count = []
    for i in range(grade_num):
        all_grade_count.append(0)
        one_grade_price = []
        for j in range(time_series_length):
            one_grade_price.append(0)
        all_grades_avg_price.append(one_grade_price)
    
    for i in range(community_num):
        all_grade_count[int(community_list[i].grade) - 1] += 1
        for j in range(time_series_length):
            all_grades_avg_price[int(community_list[i].grade) - 1][j] += community_list[i].price_time_series[j]
            
    for i in range(grade_num):
        for j in range(time_series_length):
            all_grades_avg_price[i][j] =  float(all_grades_avg_price[i][j]) / float(all_grade_count[i])
    
    dates = []
    for year in range(2016, 2019):
        for month in range(1, 13):
            dates.append(dt.datetime(year=year, month=month, day=1))
    
    dates = dates[3:27]
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size

    file_title = u'各星级小区均价时间序列均值图'
    x_label = u'日期'
    y_label = u'均价（元）'
    all_grade_labels = [u'一星小区', u'二星小区', u'三星小区', u'四星小区', u'五星小区']
    handles = []
    labels = []
    colors = cm.rainbow(np.linspace(0, 1, grade_num))
    color_index = 0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(grade_num):
        one_label, = plt.plot(dates, all_grades_avg_price[grade_num - 1 - i], color=colors[grade_num - 1 - color_index])
        handles.append(one_label)
        labels.append(all_grade_labels[grade_num - 1 - i])
        color_index += 1
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   #to get a tick every 4 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))     #optional formatting 
    ax.grid(linestyle='--', linewidth=0.5)
    plt.legend(handles = handles, labels = labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(u'时间序列图\\'+file_title+'.png', dpi=500)
    plt.show()

def cal_cluster_class_avg_prcie_time_series(community_list, n_cluster, type_str, linkage_type, label_class):
    all_classes_avg_price_time_series = []
    all_classes_records_count = []
    for cluster in range(n_cluster):
        one_class_avg_price_time_series = []
        all_classes_records_count.append(0)
        for i in range(len(community_list[0].timeline)):
            one_class_avg_price_time_series.append(0)
        all_classes_avg_price_time_series.append(one_class_avg_price_time_series)
    for community in community_list:
        if type_str == 'euclidean':
            index = community.price_time_series_euclidean_cluster_label[linkage_type] - 1
        elif type_str == 'pearson':
            index = community.price_time_series_pearson_cluster_label[linkage_type] - 1
        elif type_str == 'dtw':
            index = community.price_time_series_dtw_cluster_label[linkage_type] - 1
        for i in range(len(community_list[0].timeline)):
            all_classes_avg_price_time_series[index][i] += community.price_time_series[i]
        all_classes_records_count[index] += 1
    for i in range(n_cluster):
        for j in range(len(community_list[0].timeline)):
            all_classes_avg_price_time_series[i][j] = float(all_classes_avg_price_time_series[i][j]) / float(all_classes_records_count[i])
    dates = []
    
    for year in range(2016, 2019):
        for month in range(1, 13):
            dates.append(dt.datetime(year=year, month=month, day=1))
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 6.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    dates = dates[3:27]
    file_title = u'时间序列聚类均值图(' + type_str + ',' + linkage_type + ')'
    if type_str == 'euclidean':
        if linkage_type == 'average':
            plot_title = label_class[1]
        elif linkage_type == 'complete':
            plot_title = label_class[2]
    elif type_str == 'pearson':
        if linkage_type == 'average':
            plot_title = label_class[3]
        elif linkage_type == 'complete':
            plot_title = label_class[4]
    elif type_str == 'dtw':
        if linkage_type == 'average':
            plot_title = label_class[5]
        elif linkage_type == 'complete':
            plot_title = label_class[6]
    x_label = u'日期'
    y_label = u'均价（元）'
    all_cluster_labels = [u'类1', u'类2', u'类3', u'类4', u'类5']
    handles = []
    labels = []
    colors = cm.rainbow(np.linspace(0, 1, n_cluster))
    color_index = 0
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(n_cluster):
        one_label, = plt.plot(dates, all_classes_avg_price_time_series[n_cluster - 1 - i], color=colors[n_cluster - 1 - color_index])
        handles.append(one_label)
        labels.append(all_cluster_labels[n_cluster - 1 - i])
        color_index += 1
    
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))   #to get a tick every 4 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))     #optional formatting 
    ax.grid(linestyle='--', linewidth=0.5)
    plt.legend(handles = handles, labels = labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.title(plot_title)
    plt.savefig(u'时间序列图\\'+file_title+'.png', dpi=500)
    plt.show()
    
def display_community_grade_with_contour_sample(community_list, grade_num):
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    ax = plt.gca()
    ax.set_aspect(1)
    handles = []
    labels = []
    colors = cm.gist_rainbow(np.linspace(0, 1, grade_num))
    color_index = 0
    community_grade_list = []
    for i in range(grade_num):
        temp = dict()
        temp['all_longitude'] = []
        temp['all_latitude'] = []
        community_grade_list.append(temp)
    for community in community_list:
        community_grade_list[int(community.grade) - 1]['all_longitude'].append(community.longitude)
        community_grade_list[int(community.grade) - 1]['all_latitude'].append(community.latitude)
    
    x0 = community_grade_list[0]['all_longitude']
    y0 = community_grade_list[0]['all_latitude']
    
    data = np.ndarray(shape=(len(x0), 2), dtype=float)
    for i in range(len(x0)):
    	data[i][0] = x0[i]
    	data[i][1] = y0[i]
    
    x, y = data.T

    
    print(data.shape)
    print(data[0][0])
    print(data[1][1])
    '''
    # Create data: 200 points
    data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200)
    x, y = data.T
    print(data.shape)
    print(data[0][0])
    print(data[1][1])
    '''
    
    # Create a figure with 6 plot areas
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(21, 5))
     
    # Everything sarts with a Scatterplot
    axes[0].set_title('Scatterplot')
    axes[0].plot(x, y, 'ko')
    # As you can see there is a lot of overplottin here!
     
    # Thus we can cut the plotting window in several hexbins
    nbins = 20
    axes[1].set_title('Hexbin')
    axes[1].hexbin(x, y, gridsize=nbins, cmap=plt.cm.BuGn_r)
     
    # 2D Histogram
    axes[2].set_title('2D Histogram')
    axes[2].hist2d(x, y, bins=nbins, cmap=plt.cm.BuGn_r)
     
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
     
    # plot a density
    axes[3].set_title('Calculate Gaussian KDE')
    axes[3].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
     
    # add shading
    axes[4].set_title('2D Density with shading')
    axes[4].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
     
    # contour
    axes[5].set_title('Contour')
    axes[5].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
    axes[5].contour(xi, yi, zi.reshape(xi.shape))
    
    title = u'一星小区分布图'
    plt.savefig(u'分布图\\'+title+'.png', dpi=500)
    
def display_community_grade_with_contour(community_list, grade_num):
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    xlim_min = 118.4
    xlim_max = 119.1
    ylim_min = 31.2
    ylim_max = 32.6
    ax = plt.gca()
    ax.set_aspect(1)
    handles = []
    labels = []
    colors = cm.rainbow(np.linspace(0, 1, grade_num))
    community_grade_list = []
    for i in range(grade_num):
        temp = dict()
        temp['all_longitude'] = []
        temp['all_latitude'] = []
        community_grade_list.append(temp)
    for community in community_list:
        community_grade_list[int(community.grade) - 1]['all_longitude'].append(community.longitude)
        community_grade_list[int(community.grade) - 1]['all_latitude'].append(community.latitude)
    
    fig, axes = plt.subplots(ncols=grade_num, nrows=2, figsize=(15, 12))
    plt.subplots_adjust(left=0.075, right=0.95, bottom=0.05, top=0.95, wspace=0.4)
    nbins = 50
    
    for i in range(grade_num):
        x0 = community_grade_list[i]['all_longitude']
        y0 = community_grade_list[i]['all_latitude']
        
        data = np.ndarray(shape=(len(x0), 2), dtype=float)
        for j in range(len(x0)):
        	data[j][0] = x0[j]
        	data[j][1] = y0[j]
        
        x, y = data.T
        
        k = kde.gaussian_kde(data.T)
        xi, yi = np.mgrid[xlim_min:xlim_max:nbins*1j, ylim_min:ylim_max:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        axes[0][i].set_title(all_grade_labels[i] + u'小区(' + str(len(x)) + '个)')
        
#        axes[0][i].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.rainbow)
        axes[0][i].contourf(xi, yi, zi.reshape(xi.shape), 8, alpha=.75, cmap=plt.cm.rainbow)
        axes[0][i].contour(xi, yi, zi.reshape(xi.shape), colors='black', linewidths=.5)
        
        axes[0][i].set_xlim((xlim_min, xlim_max))
        axes[0][i].set_ylim((ylim_min, ylim_max))
        axes[0][i].set_xlabel(u'经度（°，E）')
        axes[0][i].set_ylabel(u'纬度（°，N）')
        
        axes[1][i].set_title(all_grade_labels[i] + u'小区(' + str(len(x)) + '个)')
        axes[1][i].scatter(x, y, color=colors[i], s=0.75)
        axes[1][i].set_xlim((xlim_min, xlim_max))
        axes[1][i].set_ylim((ylim_min, ylim_max))
        axes[1][i].set_xlabel(u'经度（°，E）')
        axes[1][i].set_ylabel(u'纬度（°，N）')
        
    title = u'各星级小区分布图'
    plt.savefig(u'分布图\\'+title+'.png', dpi=500)
    
def display_community_grade(community_list, grade_num):
    '''
    community_location = dict()
for district in all_districts:
    community_location[district] = dict()
    community_location[district]['longitude'] = []
    community_location[district]['latitude'] = []
for community in community_list:
    community_location[community.district]['longitude'].append(community.longitude)
    community_location[community.district]['latitude'].append(community.latitude)
    '''
    
    all_grade_labels = [u'一星', u'二星', u'三星', u'四星', u'五星']
    ax = plt.gca()
    ax.set_aspect(1)
    handles = []
    labels = []
    colors = cm.gist_rainbow(np.linspace(0, 1, grade_num))
    color_index = 0
    community_grade_list = []
    for i in range(grade_num):
        temp = dict()
        temp['all_longitude'] = []
        temp['all_latitude'] = []
        community_grade_list.append(temp)
    for community in community_list:
        community_grade_list[int(community.grade) - 1]['all_longitude'].append(community.longitude)
        community_grade_list[int(community.grade) - 1]['all_latitude'].append(community.latitude)
    for i in range(grade_num):
        one_label = plt.scatter(community_grade_list[i]['all_longitude'], 
                                community_grade_list[i]['all_latitude'], 
                                color=colors[color_index], s=1, alpha=0.03)
        handles.append(one_label)
        labels.append(all_grade_labels[i])
        color_index += 1
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 4.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    title = u'小区评级分布图'
    plt.xlim((118.25, 119.5))
    plt.xlabel(u'经度（°，E）')
    plt.ylabel(u'纬度（°，N）')
    leg = plt.legend(handles = handles, labels = labels, loc = 'best')
    plt.tight_layout()
    plt.savefig(u'分布图\\'+title+'.png', dpi=500)

def cal_probability_distributions(all_price, all_districts, districts_price):
    title = u'南京市小区均价概率分布直方图'
    plt.hist(all_price, bins=500, color='steelblue', density=True)
    
    plt.xlabel(u'均价（元）')
    plt.ylabel(u'概率')
    plt.tight_layout()
    plt.savefig(u'概率分布直方图\\'+title+'.png', dpi=500)
    plt.show()
    
    for district in all_districts:
        title = u'' + district + u'小区均价概率分布直方图'
        plt.hist(districts_price[district], bins=500, color='steelblue', density=True)
        plt.xlabel(u'均价（元）')
        plt.ylabel(u'概率')
        plt.tight_layout()
        plt.savefig(u'概率分布直方图\\'+title+'.png', dpi=500)
        
def display_community_location(all_districts, community_location):
    title = u'南京市小区分布图'
    ax = plt.gca()
    ax.set_aspect(1)
    handles = []
    labels = []
    colors = cm.gist_rainbow(np.linspace(0, 1, len(all_districts)))
    color_index = 0
    for district in all_districts:
        one_label = plt.scatter(community_location[district]['longitude'], 
                                community_location[district]['latitude'], 
                                color=colors[color_index], s=0.5)
        handles.append(one_label)
        labels.append(district)
        color_index += 1
    
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 4.0
    fig_size[1] = 4.0
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.xlim((118.25, 119.5))
    plt.xlabel(u'经度（°，E）')
    plt.ylabel(u'纬度（°，N）')
    leg = plt.legend(handles = handles, labels = labels, loc = 'best')
    plt.tight_layout()
    plt.savefig(u'分布图\\'+title+'.png', dpi=500)