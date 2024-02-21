library(dplyr)
library(scales)
library(ggplot2)
library(cluster)
library(rgl)
library(ggplot2)
library(GGally)
library(gridExtra)
library(tidyr)
library(factoextra)
#SEP_diag<-read.csv("SEPSIS_RE_W.csv")  
SEP_lab<-read.csv("lab_drg_870_872_filtered_Feb_iv.csv") ##input EHR data
SEP_diag_new<-read.csv("sep_re_new_001_adjtime.csv") ##soft cluster results
upone<-max(SEP_diag_new$X0)
uptwo<-max(SEP_diag_new$X1)
upthree<-max(SEP_diag_new$X2)
D_value<-max(upone,uptwo,upthree)
SEP_diag_new <- SEP_diag_new %>%
  mutate(abm = 1 - apply(select(., 2:4), 1, function(x) (min(x) / D_value)^(1/3)) )
soft_re<-SEP_diag_new[,c(8,9,10,11)]
rownames(soft_re)<-SEP_diag_new$SUBJECT_ID

set.seed(66)
kmeans_result <-pam(soft_re, 6)
centroids<-cbind(kmeans_result[["medoids"]])
colnames(centroids)<-c( "P1","P2","P3", "abm")
original_order <- 1:nrow(centroids)
new_order <- order(centroids[,4])
cluster_mapping <- setNames(original_order, new_order)
centroids <- centroids[order(centroids[,4]), ]
rownames(centroids) <- paste("cluster", 1:nrow(centroids), sep="")


write.csv(centroids, file = "result_centroids_MIMIC.csv")
kmeans_result[["centers"]]
colnames <- names(soft_re)
cluster_colors <- rainbow(6)
KCLU <- cluster_mapping[as.character(kmeans_result$clustering)]

soft_re$cluster <- factor(KCLU)
subset <- soft_re[, c(1,2,3,4)]
pdf("/Users/gaixin/Desktop/sep0627/MMIC_FIG/pair_fig0925.pdf", width = 8) 

panel.scatter <- function(x, y, ...){
  points(x, y, pch = 16, col = cluster_colors[KCLU])
}


par(mfrow = c(1, 1), mar = c(1, 1, 1, 4))
par(fig = c(0, 0.8, 0, 1))
pairs(subset[, !names(subset) %in% "empty"], 
      lower.panel = panel.scatter,
      upper.panel = NULL,
      diag.panel = NULL)


par(fig = c(0.5, 1, 0.5, 1), new = TRUE)
plot(0, 0, type = "n", axes = FALSE, xlab = "", ylab = "")
legend("center", legend = paste("Hybrid Sub-phenotype", 1:6), fill = cluster_colors, cex = 1.2, bty = "n") 
dev.off()

##################Silhouette Analysis###########################

k_range <- 2:20
avg_sil_values <- numeric(length(k_range))
for(i in seq_along(k_range)) {
  set.seed(i)
  kmeans_result <- kmeans(soft_re, centers = k_range[i])
  silhouette_result <- silhouette(kmeans_result$cluster, dist(soft_re))
  avg_sil_values[i] <- mean(silhouette_result[, 3])
}

silhouette_df <- data.frame(
  k = k_range,
  Avg_Silhouette = avg_sil_values
)

ggplot(silhouette_df, aes(x = k, y = Avg_Silhouette)) +
  geom_line() +
  geom_point() +
  xlab("Number of clusters (k)") +
  ylab("Average silhouette width") +
  ggtitle("Silhouette Analysis for Determining Optimal k") +
  ylim(0, max(silhouette_df$Avg_Silhouette)) 




##########################################################

SEP_lab$cluster <- KCLU[match(SEP_lab$SUBJECT_ID, rownames(soft_re))]
SEP_lab <- SEP_lab[!is.na(SEP_lab$cluster), ]

create_plot <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120*60, by = time_interval), labels = FALSE, include.lowest = TRUE))
  
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  
   SEP_lab_SC_summary <- SEP_lab_SC %>% 
     group_by(cluster, Time_Bin) %>%
     summarise(Median_Value = median(VALUE, na.rm = TRUE),
               Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
               Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
   
  
  
  
  
  #colors <- rainbow(6)  # 生成6种颜色
  
  ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_line(size = 1.2) +
    geom_ribbon(aes(ymin = Lower_CI, ymax = Upper_CI), alpha = 0.1) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = paste0("Mean ", feature, " Level"), color = "Cluster") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    ) +
    ggtitle(paste0("Comparative ", feature, " Levels Over Time for Each Cluster"))
}

pdf("/Users/gaixin/Desktop/sep0627/newdata0703/ComparativePlots_jldb001_c6_0824.pdf", width = 15)  # increase width to 1.5 times the default

features <- list(c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))

for (feature in features) {
  print(create_plot(feature[1], as.integer(feature[2])))
}

dev.off()





create_plot <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  

   SEP_lab_SC_summary <- SEP_lab_SC %>% 
     group_by(cluster, Time_Bin) %>%
     summarise(Median_Value = median(VALUE, na.rm = TRUE),
               Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
               Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  #
  
  ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = paste0("Median ", feature, " Level"), color = "Cluster") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    ) +
    ggtitle(paste0("Comparative ", feature, " Levels Over Time for Each Cluster"))
}

pdf("/Users/gaixin/Desktop/sep0627/newdata0703/ComparativePlots_jldb001_c6_smo_q_0824.pdf", width = 15)
features <- list(c("Arterial Blood Pressure systolic",4),c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))

for (feature in features) {
  print(create_plot(feature[1], as.integer(feature[2])))
}

dev.off()


library(jpeg)
create_plot <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = paste0("Median ", feature, " Level"), color = "Hybrid Sub-phenotype") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    ) +
    ggtitle(paste0("Comparative ", feature, " Levels Over Time for Each Hybrid Sub-phenotype"))
  
  ggsave(filename = paste0("/Users/gaixin/Desktop/sep0627/MMIC_FIG/mmic_0925/ComparativePlots_0922_", feature, ".jpg"), plot = p, dpi = 300)
}

#features <- list(c("systemicsystolic",4),c("Base Excess", 4), c("creatinine",4), c("heartrate",2), c("PT - INR",4), c("lactate", 12), c("respiration",8))
features <- list(c("Arterial Blood Pressure systolic",4),c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))

for (feature in features) {
  create_plot(feature[1], as.integer(feature[2]))
}





############################同类同幅对比图##############################
library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)
library(patchwork)
library(tidyverse)

library(cowplot)
create_plot <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  
  time_interval <- time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0, 120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = feature, color = "Hybrid Sub-phenotype") +
    guides(color = guide_legend(nrow = length(unique(SEP_lab_SC_summary$cluster)), byrow = TRUE)) +  # Set legend rows to the number of clusters
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      legend.background = element_rect(colour = "black", fill = NA, size = 0.5),
      legend.key = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 14),
      legend.position = "bottom",
      legend.box = "vertical",   # Set legend to vertical
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}


# Original plot function
create_plot_no_legend <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = feature, color = "Hybrid Sub-phenotype") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 20),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "none",  # No legend here
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    )
  return(p)
}

features <- list(c("Arterial Blood Pressure systolic",4),c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))
plot_list_no_legend <- list()

for (feature in features) {
  plot <- create_plot_no_legend(feature[1], as.integer(feature[2]))
  plot_list_no_legend[[feature[1]]] <- plot
}

one_plot_with_legend <- create_plot(features[[1]][1], as.integer(features[[1]][2]))
legend_grob <- get_legend(one_plot_with_legend)
empty_plot <- ggplot() + theme_void()
legend_plot <- empty_plot + annotation_custom(grob = legend_grob)

for (i in 1:7) {
  plot_list_no_legend[[i]] <- plot_list_no_legend[[i]] + labs(tag = LETTERS[i])
}

all_plots <- (plot_list_no_legend[[1]] | plot_list_no_legend[[2]] | plot_list_no_legend[[3]] | plot_list_no_legend[[4]]) / 
  (plot_list_no_legend[[5]] | plot_list_no_legend[[6]] | plot_list_no_legend[[7]] | legend_plot)

print(all_plots)



library(ggplot2)
library(grid)
library(gridExtra)
library(gtable)
library(patchwork)
library(tidyverse)
library(cowplot)
create_plot <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature)
  
  time_interval <- time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0, 120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = feature, color = "Hybrid Sub-phenotype") +
    guides(color = guide_legend(nrow = length(unique(SEP_lab_SC_summary$cluster)), byrow = TRUE)) +  # Set legend rows to the number of clusters
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      legend.background = element_rect(colour = "black", fill = NA, size = 0.5),
      legend.key = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 14),
      legend.position = "bottom",
      legend.box = "vertical",   # Set legend to vertical
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}


# Original plot function
create_plot_no_legend <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature,cluster == 1)
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = feature, color = "Hybrid Sub-phenotype") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 20),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "none",  # No legend here
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    )
  return(p)
}

features <- list(c("Arterial Blood Pressure systolic",4),c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))
plot_list_no_legend <- list()

for (feature in features) {
  plot <- create_plot_no_legend(feature[1], as.integer(feature[2]))
  plot_list_no_legend[[feature[1]]] <- plot
}

one_plot_with_legend <- create_plot(features[[1]][1], as.integer(features[[1]][2]))
legend_grob <- get_legend(one_plot_with_legend)


empty_plot <- ggplot() + theme_void()
legend_plot <- empty_plot + annotation_custom(grob = legend_grob)

for (i in 1:7) {
  plot_list_no_legend[[i]] <- plot_list_no_legend[[i]] + labs(tag = LETTERS[i])
}


all_plots <- (plot_list_no_legend[[1]] | plot_list_no_legend[[2]] | plot_list_no_legend[[3]] | plot_list_no_legend[[4]]) / 
  (plot_list_no_legend[[5]] | plot_list_no_legend[[6]] | plot_list_no_legend[[7]] | legend_plot)

print(all_plots)




create_plot_no_legend <- function(feature, time_interval_hours) {
  SEP_lab_SC <- SEP_lab %>% filter(FEATURE_NAME == feature,cluster == 6)
  time_interval <-  time_interval_hours  # convert to minutes
  SEP_lab_SC$Time_Bin <- with(SEP_lab_SC, cut(RECORD_MIN, breaks = seq(0,120, by = time_interval), labels = FALSE, include.lowest = TRUE))
  SEP_lab_SC <- na.omit(SEP_lab_SC, "Time_Bin")
  SEP_lab_SC_summary <- SEP_lab_SC %>% 
    group_by(cluster, Time_Bin) %>%
    summarise(Median_Value = median(VALUE, na.rm = TRUE),
              Lower_CI = quantile(VALUE, probs = 0.25, na.rm = TRUE),
              Upper_CI = quantile(VALUE, probs = 0.75, na.rm = TRUE))
  p <- ggplot(SEP_lab_SC_summary, aes(x = Time_Bin, y = Median_Value, color = as.factor(cluster))) +
    geom_smooth(aes(ymin = Lower_CI, ymax = Upper_CI), method = "loess", se = TRUE, alpha = 0.2) +
    scale_color_manual(values = cluster_colors[6]) +
    labs(x = paste0("Time Interval (", time_interval_hours, "-hour bins)"), y = feature, color = "Hybrid Sub-phenotype") +
    theme_bw() +
    theme(
      text = element_text(family = "serif", color = "black"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 20),
      axis.text = element_text(size = 10),
      legend.title = element_text(face = "bold", size = 12),
      legend.text = element_text(size = 10),
      legend.position = "none",  # No legend here
      panel.grid.major = element_line(colour = "grey", linetype = "dashed"),
      panel.grid.minor = element_blank()
    )
  return(p)
}

features <- list(c("Arterial Blood Pressure systolic",4),c("Base Excess", 4), c("Creatinine",4), c("Heart Rate",2), c("INR(PT)",4), c("Lactate", 12), c("Respiratory Rate",8))
plot_list_no_legend <- list()

for (feature in features) {
  plot <- create_plot_no_legend(feature[1], as.integer(feature[2]))
  plot_list_no_legend[[feature[1]]] <- plot
}

one_plot_with_legend <- create_plot(features[[1]][1], as.integer(features[[1]][2]))
legend_grob <- get_legend(one_plot_with_legend)
empty_plot <- ggplot() + theme_void()
legend_plot <- empty_plot + annotation_custom(grob = legend_grob)
for (i in 1:7) {
  plot_list_no_legend[[i]] <- plot_list_no_legend[[i]] + labs(tag = LETTERS[i])
}

all_plots <- (plot_list_no_legend[[1]] | plot_list_no_legend[[2]] | plot_list_no_legend[[3]] | plot_list_no_legend[[4]]) / 
  (plot_list_no_legend[[5]] | plot_list_no_legend[[6]] | plot_list_no_legend[[7]] )

print(all_plots)



