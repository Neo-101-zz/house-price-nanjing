library(factoextra)
library(NbClust)

setwd("D:\\Program\\Project\\house_price_nanjing_soufang")

####################################################################################################

community_location <- read.table("community_location.data")
location_matrix <- data.matrix(community_location)

# # Elbow method
# pic <- fviz_nbclust(location_matrix, kmeans, method = "wss") +
#     geom_vline(xintercept = 4, linetype = 2)+
#   labs(subtitle = "Elbow method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\location_elbow_method.png")
# plot(pic)
# dev.off()
# # Silhouette method
# pic <- fviz_nbclust(location_matrix, kmeans, method = "silhouette")+
#   labs(subtitle = "Silhouette method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\location_silhouette_method.png")
# plot(pic)
# dev.off()
# # Gap statistic
# # nboot = 50 to keep the function speedy. 
# # recommended value: nboot= 500 for your analysis.
# # Use verbose = FALSE to hide computing progression.
# set.seed(123)
# pic <- fviz_nbclust(location_matrix, kmeans, nstart = 25,  method = "gap_stat", nboot = 50, verbose = FALSE)+
#   labs(subtitle = "Gap statistic method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\location_gap_statistic_method.png")
# plot(pic)
# dev.off()

nb <- NbClust(location_matrix, distance = "euclidean", min.nc = 2,
        max.nc = 12, method = "kmeans")
pic <- fviz_nbclust(nb)
png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\location_NbClust.png")
plot(pic)
dev.off()

####################################################################################################

community_two_rates <- read.table("community_two_rates.data")
community_two_rates_matrix <- data.matrix(community_two_rates)

# # Elbow method
# pic <- fviz_nbclust(community_two_rates_matrix, kmeans, method = "wss") +
#     geom_vline(xintercept = 4, linetype = 2)+
#   labs(subtitle = "Elbow method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\two_rates_elbow_method.png")
# plot(pic)
# dev.off()
# # Silhouette method
# pic <- fviz_nbclust(community_two_rates_matrix, kmeans, method = "silhouette")+
#   labs(subtitle = "Silhouette method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\two_rates_silhouette_method.png")
# plot(pic)
# dev.off()
# # Gap statistic
# # nboot = 50 to keep the function speedy. 
# # recommended value: nboot= 500 for your analysis.
# # Use verbose = FALSE to hide computing progression.
# set.seed(123)
# pic <- fviz_nbclust(community_two_rates_matrix, kmeans, nstart = 25,  method = "gap_stat", nboot = 50, verbose = FALSE)+
#   labs(subtitle = "Gap statistic method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\two_rates_gap_statistic_method.png")
# plot(pic)
# dev.off()

# nb <- NbClust(community_two_rates_matrix, distance = "euclidean", min.nc = 2,
#         max.nc = 12, method = "kmeans")
# pic <- fviz_nbclust(nb)
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\two_rates_NbClust.png")
# plot(pic)
# dev.off()

####################################################################################################

community_price_time_series_avg_and_var <- read.table("community_price_time_series_avg_and_var.data")
community_price_time_series_avg_and_var_matrix <- data.matrix(community_price_time_series_avg_and_var)

# # Elbow method
# pic <- fviz_nbclust(community_price_time_series_avg_and_var_matrix, kmeans, method = "wss") +
#     geom_vline(xintercept = 4, linetype = 2)+
#   labs(subtitle = "Elbow method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\price_time_series_avg_and_var_elbow_method.png")
# plot(pic)
# dev.off()
# # Silhouette method
# pic <- fviz_nbclust(community_price_time_series_avg_and_var_matrix, kmeans, method = "silhouette")+
#   labs(subtitle = "Silhouette method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\price_time_series_avg_and_var_silhouette_method.png")
# plot(pic)
# dev.off()
# # Gap statistic
# # nboot = 50 to keep the function speedy. 
# # recommended value: nboot= 500 for your analysis.
# # Use verbose = FALSE to hide computing progression.
# set.seed(123)
# pic <- fviz_nbclust(community_price_time_series_avg_and_var_matrix, kmeans, nstart = 25,  method = "gap_stat", nboot = 50, verbose = FALSE)+
#   labs(subtitle = "Gap statistic method")
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\price_time_series_avg_and_var_gap_statistic_method.png")
# plot(pic)
# dev.off()

# nb <- NbClust(community_price_time_series_avg_and_var_matrix, distance = "euclidean", min.nc = 2,
#         max.nc = 12, method = "kmeans")
# pic <- fviz_nbclust(nb)
# png(filename="D:\\Program\\Project\\house_price_nanjing_soufang\\cluster_number\\community_price_time_series_avg_and_var_NbClust.png")
# plot(pic)
# dev.off()