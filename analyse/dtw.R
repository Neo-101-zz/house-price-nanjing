setwd("D:\\Program\\Project\\house_price_nanjing_soufang\\data")
community_price_time_series <- read.table("community_price_time_series.data")
price_time_series_matrix <- data.matrix(community_price_time_series)

# first time create dist_matrix
# dist_vector <- vector(mode='double', length=(nrow(community_price_time_series) * nrow(community_price_time_series)))
# dist_matrix <- matrix(data=dist_vector, nrow=nrow(community_price_time_series), ncol=nrow(community_price_time_series))

# then read dist_matrix
dist_data_frame <- read.table("dtw_dist_matrix.txt")
dist_matrix <- data.matrix(dist_data_frame)

library(dtw)

for (i in 1:nrow(price_time_series_matrix)){
	# if(i > 2000){
	# 	write.table(dist_matrix, file="dist_matrix.txt", row.names=FALSE, col.names=FALSE)
	# 	q()
	# }
	for(j in 1:nrow(price_time_series_matrix)){
		if(i == j){
			dist_matrix[i, j] <- 0
			next
		}
	    if(dist_matrix[j, i] == 0){
	    	if(dist_matrix[i, j] == 0){
				vi <- as.numeric(price_time_series_matrix[i,])
				vj <- as.numeric(price_time_series_matrix[j,])
				alignment <- dtw(vi, vj, keep=TRUE, window.type="itakura")
				alignment$distance
				dist_matrix[i, j] <- alignment$distance
				dist_matrix[j, i] <- dist_matrix[i, j]
				next
	    	}
	    	if(dist_matrix[i, j] > 0){
	    		dist_matrix[j, i] <- dist_matrix[i, j]
	    		next
	    	}
		  
	    }
		if(dist_matrix[j, i] > 0){
			if(dist_matrix[i, j] == 0){
				dist_matrix[i, j] <- dist_matrix[j, i]
				next
			}
			if(dist_matrix[i, j] > 0){
				next
			}
		}
  	}
}

write.table(dist_matrix, file="dtw_dist_matrix.txt", row.names=FALSE, col.names=FALSE)
