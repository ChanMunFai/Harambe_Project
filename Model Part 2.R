library(tidyverse)
library(dplyr)
df_2 <- readRDS(file="clean_data.RDS")

# Percentage of individuals employed more than 6 months is 0.067
table(df_2$loe_morethan6months)


### Section A: Unsupervised ML
library(cluster)
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

# Chose 5 clusters based on dendogram and regression trials 
# Hierarchical clustering to identify statistically significant clusters of individuals 

### Section B: Supervised ML
library(caret)
library(skimr)
library(RANN)

# Removed variables with many missing values
df_3 <- select(df_2, -c(X.1, X, unid, survey_date_month, survey_num, working, job_start_date, job_leave_date, financial_situation_now, financial_situation_5years, age, fin_situ_now, fin_situ_future, com_score, num_score, company_size, monthly_pay, length_of_employment, peoplelive_15plus, province, dob))
df_3 <- df_3 %>% filter(!is.na(volunteer), !is.na(leadershiprole), !is.na(peoplelive), !is.na(anygrant), !is.na(anyhhincome), !is.na(givemoney_yes))

# Convert non-numeric variables to factors
df_3 <- mutate(df_3,
               loe_morethan6months = factor(loe_morethan6months),
               anygrant = factor(anygrant),
               anyhhincome = factor(anyhhincome),
               givemoney_yes = factor(givemoney_yes),
               numchildren = as.numeric(numchildren),
               numearnincome = as.numeric(numearnincome))

# Impute numeric variables (e.g. age and test scores)
preProcess_missingdata_model <- preProcess(df_3, method="knnImpute")
df_3 <- predict(preProcess_missingdata_model, newdata = df_3)

# Data Pre-processing
dummies_model <- dummyVars(loe_morethan6months ~., data=df_3)
df_3_loe_morethan6months <- df_3$loe_morethan6months
df_3_mat <- predict(dummies_model, newdata=df_3)
df_3 <- data.frame(df_3_mat)

df_3$loe_morethan6months <- df_3_loe_morethan6months

# Splitting Data into training and testing data
set.seed(100)
trainRowNumbers <- createDataPartition(df_3$loe_morethan6months, p=0.8, list=FALSE)
trainData <- df_3[trainRowNumbers,]
testData <- df_3[-trainRowNumbers,]

# Model rpart (recursive partitioning) and Cross Validation
install.packages("e1071")
trControl <- trainControl(method = "cv", number=10, verboseIter = TRUE)
model_rpart <- train(loe_morethan6months ~., data=trainData, method="rpart", trControl=trControl, tuneGrid=expand.grid(cp=seq(0.000,0.02,0.0025)))
predicted <- predict(model_rpart, testData[,-length(testData)])
model_rpart$results

# Confusion Matrix
results_rpart <- table(predicted, testData$loe_morethan6months)
confusionMatrix(results_rpart)
# Accuracy of 0.9166, but model did not predict correctly predict any positive cases.

### Section C1: Other Models (less useful models)
df_3i <- select(df_2, -c(X.1, X, survey_date_month, survey_num, working, job_start_date, job_leave_date, financial_situation_now, financial_situation_5years, age, fin_situ_now, fin_situ_future, com_score, num_score, company_size, monthly_pay, length_of_employment, peoplelive_15plus, province, dob))
df_3i <- df_3i %>% filter(!is.na(volunteer), !is.na(leadershiprole), !is.na(peoplelive), !is.na(anygrant), !is.na(anyhhincome), !is.na(givemoney_yes))
df_3i <- mutate(df_3i,
                loe_morethan6months = factor(loe_morethan6months),
                anygrant = factor(anygrant),
                anyhhincome = factor(anyhhincome),
                givemoney_yes = factor(givemoney_yes),
                numchildren = as.numeric(numchildren),
                numearnincome = as.numeric(numearnincome))

# Impute numeric variables (e.g. age and test scores)
preProcess_missingdata_model <- preProcess(df_3i, method="knnImpute")
df_3i <- predict(preProcess_missingdata_model, newdata = df_3i)

# Split into training and testing data
df_train_index <- df_3i %>%
  select(unid) %>% 
  sample_frac(0.7)
df_train <- left_join(df_train_index, df_3i)
df_test <- anti_join(df_3i, df_train_index)

# Logistic Regression
logistic <- glm(loe_morethan6months ~ gender + age_at_survey + opt_score + grit_score, data=df_train, family="binomial")
predicted_log=predict(logistic,df_test, type="response")
predicted_log <- ifelse(predicted_log>0.5,1,0)
table(predicted_log, df_test$loe_morethan6months)
# Results: Accuracy = 0.92, Sensitivity = 0.00

# SVM 
library(e1071)
fit_svm <- svm(loe_morethan6months ~ gender + age_at_survey + fin_situ_change, data=df_train)
predicted_svm <- predict(fit, df_test)
results_SVM <- table(predicted_svm, df_test$loe_morethan6months)
confusionMatrix(results_SVM)
# Results: Accuracy = 0.92, Sensitivity = 0.00

# Naive Bayes 
fit_nb <- naiveBayes(loe_morethan6months ~ gender + age_at_survey + opt_score + grit_score, data=df_train)
predicted_nb <- predict(fit_nb, df_test)
results_nb <- table(predicted_nb, df_test$loe_morethan6months)
confusionMatrix(results_nb)
# Results: Accuracy = 0.92, Sensitivity = 0.00

# GBM 
install.packages("gbm")
library(caret)
fitControl <- trainControl(method = "repeatedcv", number=4, repeats=4)
fit_gbm <- train(loe_morethan6months ~ gender + age_at_survey, data=df_train, method="gbm", trControl=fitControl, verbose=FALSE)
predicted_gbm = predict(fit_gbm, df_test, type="prob")[,2]
predicted_gbm <- ifelse(predicted_gbm>0.5,1,0)
table(predicted_gbm, df_test$loe_morethan6months)
# Results: Accuracy = 0.92, Sensitivity = 0.00

### Section C2: Other Models (more useful models)
# KNN - classification based on distance from nearest neighbours
library(kknn)
library(Metrics)
fit_knn <- kknn(loe_morethan6months ~ age_at_survey + grit_score + opt_score + grit_score + numearnincome, df_train, df_test, k = 5, kernel = "rectangular", distance = 2)
predicted_knn <- predict(fit_knn)
results_knn <- table(predicted_knn, df_test$loe_morethan6months)
confusionMatrix(results_knn)
# Results: Accuracy = 0.91, Sensitivity = 0.018

# Random Forest - decision trees formed by random sampling of training data points and random subsets of features
install.packages("randomForest")
library(randomForest)
fit_rf <- randomForest(loe_morethan6months ~ gender + age_at_survey + fin_situ_change + opt_score, df_train, ntree=500)
predicted_rf <- predict(fit_rf,df_test)
results_rf <- table(predicted_rf, df_test$loe_morethan6months)
confusionMatrix(results_rf)
# Results: Accuracy = 0.92, Sensitivity = 0.001


### Section D: XGBoost Model
library(tidyverse)
library(RANN)
library(caret)
library(skimr)
library(xgboost)

# Data cleaning : removal of variables + imputation
df_4 <- select(df_2, -c(X.1, X, survey_date_month, survey_num, working, job_start_date, job_leave_date, financial_situation_now, financial_situation_5years, age, fin_situ_now, fin_situ_future, com_score, num_score, company_size, monthly_pay, length_of_employment, peoplelive_15plus, province, dob))
df_4 <- df_4 %>% filter(!is.na(volunteer), !is.na(leadershiprole), !is.na(peoplelive), !is.na(anygrant), !is.na(anyhhincome), !is.na(givemoney_yes))
df_4 <- mutate(df_4,
               loe_morethan6months = factor(loe_morethan6months),
               anygrant = factor(anygrant),
               anyhhincome = factor(anygrant),
               givemoney_yes = factor(anygrant),
               numchildren = as.numeric(numchildren),
               numearnincome = as.numeric(numearnincome))
preProcess_missingdata_model <- preProcess(df_4, method="knnImpute")
df_4 <- predict(preProcess_missingdata_model, newdata = df_4)

# Sourced from https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/

# Dummy Variables
dummies_model2 <- dummyVars(loe_morethan6months ~., data=df_4)
df_4_loe_morethan6months <- df_4$loe_morethan6months
df_4_mat <- predict(dummies_model2, newdata=df_4)
df_4 <- data.frame(df_4_mat)
df_4$loe_morethan6months <- df_4_loe_morethan6months

# Splitting Data
set.seed(100)
trainRowNumbers2 <- createDataPartition(df_4$loe_morethan6months, p=0.8, list=FALSE)
trainData2 <- df_4[trainRowNumbers2,]
testData2 <- df_4[-trainRowNumbers2,]

# Loading labels of train data
labels = trainData2 ['loe_morethan6months']
trainData2i = trainData2 [-grep('loe_morethan6months', colnames(trainData2))]

# Combine train and test data
df_all <- df_4[-grep('loe_morethan6months',colnames(df_4))]
X = df_all[df_all$unid %in% trainData2i$unid,]
X_test = df_all[df_all$unid %in% testData2$unid,]

# Converting labels into a matrix (paramters are already best tuned)
mat_y <- as.matrix(labels)
xgb <- xgboost(data = data.matrix(X[,-1]),
               label = mat_y,
               eta = 0.025,
               max_depth = 10,
               nround=200,
               objective = "binary:logistic",
               booster = "gbtree",
               subsample = 0.8,
               scale_pos_weight = 0.5,
               colsample_bytree = 1,
               min_child_weight = 1, 
               gamma = 0
)

#Testing the results of our model
z_pred <- predict(xgb, data.matrix(X_test[,-1]))
z_pred <- ifelse(z_pred>0.5,1,0)
table(z_pred, testData$loe_morethan6months)

nrounds = 1000
# Tuning the parameters
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 0.8
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds
  #index = createFolds(tr_treated$Id_clean), # fix the folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results
)

xgb_tune <- caret::train(loe_morethan6months ~ ., data=trainData2,
                         method='xgbTree',
                         verbose = TRUE,
                         trControl = tune_control,
                         tuneGrid = tune_grid,
                         nthread = 4)
xgb_tune$bestTune

# Finding most important variables
names <- dimnames(data.matrix(X[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model=xgb)
xgb.plot.importance(importance_matrix[1:10,])

# Not very useful results from XGBoost; need further tuning 