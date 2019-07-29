# Import Libraries
if(!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if(!require(tidyverse)) install.packages("dplyr")
library(dplyr)
if (!require(kknn))  install.packages("kknn")
library(kknn)
if (!require(Metrics))  install.packages("metrics")
library(Metrics)
if(!require(cluster)) install.packages("cluster")
library(cluster)

# Read in the source data file
df <- readRDS(file="clean_data.RDS")

##################
# Supervised learning - predict is likely to be in work after six months
##################
# Select the relevant columns and morph into factors where needed
df <- select(df, -c(X, survey_date_month, survey_num, working, job_start_date, job_leave_date, financial_situation_now, financial_situation_5years, age, fin_situ_now, fin_situ_future, com_score, num_score, company_size, monthly_pay, length_of_employment, peoplelive_15plus, province, dob))
df <- df %>% filter(!is.na(volunteer), !is.na(leadershiprole), !is.na(peoplelive), !is.na(anygrant), !is.na(anyhhincome), !is.na(givemoney_yes))
df <- mutate(df,
             loe_morethan6months = factor(loe_morethan6months),
             anygrant = factor(anygrant),
             anyhhincome = factor(anygrant),
             givemoney_yes = factor(anygrant),
             numchildren = as.numeric(numchildren),
             numearnincome = as.numeric(numearnincome))

# Impute numeric variables (e.g. age and test scores)
preProcess_missingdata_model <- preProcess(df, method="knnImpute")
df <- predict(preProcess_missingdata_model, newdata = df)

# Set the seed for consistency
set.seed(69420)

# Split into training and test data (70-30)
df_train_index <- df %>%
  select(unid) %>% 
  sample_frac(0.7)
df_train <- left_join(df_train_index, df)
df_test <- anti_join(df, df_train_index)

# Perform KNN algorithm to train the model
fit <- kknn(loe_morethan6months ~ age_at_survey + grit_score + opt_score + grit_score + numearnincome, df_train, df_test, k = 5, kernel = "rectangular", distance = 2)
predicted <- predict(fit)
results <- table(predicted, df_test$loe_morethan6months)
confusionMatrix(results)

######################
# Unsupervised exploration of the data to gain insights
######################
df_2 <- read.csv("data/processed/dataframe2.csv")

# Percentage of individuals employed more than 6 months is 0.067
table(df_2$loe_morethan6months)

options(scipen=999)

df_cluster_n <- df_2 %>%
  filter(!is.na(age_at_survey) & !is.na(fin_situ_change) & !is.na(grit_score) &!is.na(opt_score))

df_unid <- df_cluster_n %>%
  select(unid)

df_cluster <- df_cluster_n %>%
  select(age_at_survey, fin_situ_change, grit_score, opt_score) %>%
  scale()

set.seed(1234)

# Hierarchical Clustering
d <- dist(df_cluster, method = "euclidean")
hc1 <- hclust(d, method = "complete")
plot(hc1, cex=0.6, hang=-1)
clusterCut <- cutree(hc1,5)
table(clusterCut)
df_cluster_h5 <- bind_cols(as.data.frame(df_cluster), as.data.frame(clusterCut))

df_cluster_h5_sum <- df_cluster_h5 %>%
  group_by(clusterCut) %>%
  summarise(age_at_survey = mean(age_at_survey),
            fin_situ_change = mean(fin_situ_change),
            grit_score = mean(grit_score),
            opt_score = mean(opt_score))

h5 <- df_cluster_h5 %>%
  select(clusterCut) %>%
  rename(h5 = clusterCut)

df_clusters_unid <- bind_cols(df_unid, h5)

df_cluster_n <- left_join(df_cluster_n, df_clusters_unid, by="unid")

# Regressions
reg1 <- lm(loe_morethan6months ~ gender, data=df_cluster_n)
summary(reg1)
reg2 <- lm(loe_morethan6months ~ anyhhincome, data=df_cluster_n)
summary(reg2)
reg3 <- lm(loe_morethan6months ~ gender*as.factor(h5), data=df_cluster_n)
summary(reg3)