# Required libraries
if(!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)
if(!require(tidyverse)) install.packages("dplyr")
library(dplyr)
if(!require(caret)) install.packages("caret")
library(caret)
if(!require(skimr)) install.packages("skimr")
library(skimr)
if(!require(RANN)) install.packages("RANN")
library(RANN)
if (!require(e1071)) install.packages("e1071")
library(e1071)

# Read in the processed dataframe
df <- readRDS(file="clean_data.RDS")

# Select only relevant data from the columns
df <- select(df, -c(X, survey_date_month, survey_num, gender, job_start_date, age, job_leave_date, financial_situation_5years, financial_situation_now, fin_situ_future,  peoplelive_15plus, province, dob, company_size, monthly_pay)) %>%
    filter(!is.na(volunteer), !is.na(leadershiprole), !is.na(peoplelive), !is.na(anygrant), !is.na(anyhhincome), !is.na(givemoney_yes)) %>%
    mutate(anygrant = factor(anygrant),
                   anyhhincome = factor(anyhhincome),
                   givemoney_yes = factor(givemoney_yes),
                   working = factor(working),
                   numchildren = as.numeric(numchildren),
                   numearnincome = as.numeric(numearnincome),
                   fin_situ_now = as.numeric(fin_situ_now))

# Impute numeric variables (e.g. age and test scores)
preProcess_missingdata_model <- preProcess(df, method="knnImpute")
df <- predict(preProcess_missingdata_model, newdata = df)

# Set the seed for consistency
set.seed(69420)

# Split into training and test data
df_train_index <- df %>%
  select(unid) %>% 
  sample_frac(0.7)
df_train <- left_join(df_train_index, df)
df_test <- anti_join(df, df_train_index)

######################
# Random Forest
######################
#if (!require(randomForest)) install.packages("randomForest")
#library(randomForest)
#fit_rf <- randomForest(working ~ grit_score + anygrant + age_at_survey + fin_situ_now + cft_score + com_score + num_score + numearnincome, data = df_train, ntrees=1000)
#predicted_rf <- predict(fit_rf, df_test)
#results_rf <- table(predicted_rf, df_test$working)
#confusionMatrix(results_rf)

######################
# SVM
#####################
#fit <- svm(working ~ grit_score + anygrant + age_at_survey + fin_situ_now + cft_score + com_score + num_score + numearnincome, data = df_train)
#predicted_svm <- predict(fit, df_test)
#results_SVM <- table(predicted_svm, df_test$working)
#confusionMatrix(results_SVM)

#####################
# Naive Bayes
# Performs the best - this model was chosen
#####################
fit_nb <- naiveBayes(working ~ grit_score + anygrant + age_at_survey + fin_situ_now + cft_score + com_score + num_score + numearnincome, data = df_train)
predicted_nb <- predict(fit_nb, df_test)
results_nb <- table(predicted_nb, df_test$working)
confusionMatrix(results_nb)
