if (!require(tidyverse)) install.packages("tidyverse")
if (!require(lubridate)) install.packages("lubridate")
library(tidyverse)
library(lubridate)

# Read in the main Harambee data
df <- read.csv("data/raw/teaching_training_data.csv")

# Calculate age
df <- df %>%
  mutate(age_at_survey = interval(dob,survey_date_month)/years(1)) %>%
  mutate(age = floor(age_at_survey))

# Add Financial Situation Change to the main dataframe
df <- df %>% 
  mutate(fin_situ_now = parse_number(as.character(financial_situation_now))) %>% 
  mutate(fin_situ_future = parse_number(as.character(financial_situation_5years))) %>% 
  mutate(fin_situ_change = fin_situ_future - fin_situ_now)

# Added in scores
df_cft <- read.csv("data/raw/teaching_training_data_cft.csv")
df_com <- read.csv("data/raw/teaching_training_data_com.csv")
df_grit <- read.csv("data/raw/teaching_training_data_grit.csv")
df_num <- read.csv("data/raw/teaching_training_data_num.csv")
df_opt <- read.csv("data/raw/teaching_training_data_opt.csv")

# Function to remove duplicate scores from the data
duplicate_purge <- function(data) {
  data %>% 
    select(2:3) %>% 
    distinct(unid, .keep_all = TRUE)
}

# Apply to score dataframes
df_cft <- duplicate_purge(df_cft)
df_com <- duplicate_purge(df_com)
df_grit <- duplicate_purge(df_grit)
df_num <- duplicate_purge(df_num)
df_opt <- duplicate_purge(df_opt)

# Merge score and main dataframes
df <- df %>%
  left_join(df_grit, by="unid")%>%
  left_join(df_opt, by="unid")%>%
  left_join(df_cft, by ="unid") %>% 
  left_join(df_com, by ="unid") %>% 
  left_join(df_num, by ="unid")

# Purge the score dataframes from the environment
rm(df_cft,df_com,df_grit,df_num,df_opt)

# Convert TRUE/FALSE to 1/0 in "Working" column
df <- mutate(df, working = factor(ifelse(working==TRUE,1,0)))

# Convert to 1/0 in "company_size" column
df <- mutate(df, company_size = factor(ifelse(company_size=="More than 200 employees",1,0)))

# Write all data a csv
write_csv(df, "data/processed/raw_data.csv")

# Use data from first survey only and use distinct individuals only [Model 1 runs on df]
df <- df %>% filter (survey_num == 1)
df <- df %>% distinct(unid,.keep_all=TRUE)

# Length of Employment
df_2 <- df
df_2 <- df_2 %>% mutate (job_start_date=ymd(job_start_date))
df_2 <- df_2 %>% mutate (job_leave_date=ymd(job_leave_date))
df_3 <- df_2 %>% filter (!is.na(job_start_date), !is.na(job_leave_date))
df_3 <- mutate(df_3, length_of_employment= job_leave_date - job_start_date)
df_3 <- df_3 %>% select(unid, length_of_employment)

df_2 <- left_join(df_2, df_3, by="unid")
rm(df_3)

# Remove individuals with missing job leave date values
df_2 <- df_2 %>% filter(working==0|(!is.na(job_start_date)&!is.na(job_leave_date)))

# Create Binary for 6 months of employment [Model 2 runs on df_2]
df_2 <- mutate (df_2, loe_morethan6months = factor(ifelse(length_of_employment>180,1,0)))
df_2$loe_morethan6months[is.na(df_2$loe_morethan6months)] <- 0
table(df_2$loe_morethan6months)

write.csv(df, 'dataframe1.csv')
write.csv(df_2, 'dataframe2.csv')

# Save to RDS for project purposes
saveRDS(df_2, file="clean_data.rds")
