#OPIM 407 ADVANCED DATA ANALYTICS

#PROFESSOR: MEHMET ERDAL BALABAN

#STUDENT NAME AND ID: AHMET EREN KESER - 23572 / ASLI GÜRLER - / HASAN - / MUSA BAGHIROV - / ZEYNEP ŞENER -

#CASE STUDY: BANK CHURN DATA ATTRITION RATE PREDICTION

#USED ALGORITHM: NAIVE BAYES WITH STRATIFIED SAMPLING, DECISION TREE ALGORITHM, RANDOM FOREST ALGORITHM

getwd()
# Reading the data
bank <- read.csv("BankChurners.csv")

# Get data summary and types by the following functions
summary(bank)
str(bank)

library(dplyr)
glimpse(bank)
# There are no missing values 


# Data visualisation

# Client num and last two columns are useless so is better to delete them

bank <- select(bank, -c("CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"))
library(ggplot2)

# For continous variables it is possible to look for distributions, check outliers and correlation

ggplot(data = bank,aes(Credit_Limit,color = Attrition_Flag))+geom_freqpoly(binwidth = 250, size = 1)
ggplot(data = bank,aes(Total_Revolving_Bal,color = Attrition_Flag))+geom_freqpoly(binwidth = 200, size = 1)
ggplot(data = bank,aes(Months_Inactive_12_mon,color = Attrition_Flag))+geom_freqpoly(binwidth = 5, size = 1)

# Describe the trends and add more variables

boxplot(bank$Credit_Limit)$out
boxplot(bank$Total_Revolving_Bal)$out
boxplot(bank$Months_Inactive_12_mon)$out
bank_cor <- round(cor(bank[,c("Credit_Limit", "Total_Revolving_Bal", "Months_Inactive_12_mon")]), 1)
library(corrplot)
corrplot(bank_cor,  title = "Correlation")+theme(plot.title = element_text(hjust = 0.5))
# No correlation observed 

#For factor variables
ggplot(bank, aes(x=Gender,fill=Attrition_Flag))+ geom_bar(position = 'fill')
ggplot(bank, aes(x=Education_Level,fill=Attrition_Flag))+ geom_bar(position = 'fill')
ggplot(bank, aes(x=Income_Category,fill=Attrition_Flag))+ geom_bar(position = 'fill')

# You can add more attributes and describe them the historgram can be viewed more clearly by clicking zoom button

# Naive Bayes Algorithm

# Data Preprocessing
# Transforming Categorical Variables into Factor Variables 

numeric_columns <- c(2,4,9:20)
for(i in 1:ncol(bank)){
     if(i %in% numeric_columns)
        bank[,i] <- as.numeric(bank[,i])
     else
         bank[,i] <- as.factor(bank[,i])
 }

# SMOTE Technique to Balance the Data
library(caret)
set.seed(1)
#install.packages("DMwR")
library(DMwR)
bank_s <- SMOTE(Attrition_Flag ~ . , bank, perc.over = 100, k=3)
table(bank_s$Attrition_Flag)

# Modelling

# Creating Training and Testing Datasets
# Stratified Sampling (Hold Out)
library(caret)
set.seed(1)
my_indexes <- createDataPartition(y = bank_s$Attrition_Flag, p = .70, list = FALSE)
training <- as.data.frame(bank_s[my_indexes,])
test <- as.data.frame(bank_s[-my_indexes,])
table(bank_s$Attrition_Flag)
table(training$Attrition_Flag)
table(test$Attrition_Flag)

# Applying Naive Bayes Algorithm
library(e1071)
naiveB_model <- naiveBayes(training[,2:20], training[[1]])
naiveB_model

# Finding Predictions of The Model
(nb_predictions <- predict(naiveB_model, test[,2:20]))
(nb_probs <- predict(naiveB_model, test[,2:20], "raw"))
myConfTable <- confusionMatrix(data = nb_predictions, reference = test[[1]], dnn = c("Predictions", "Actual/Reference"), mode = "everything")
print(myConfTable)                                        
(accuracy <- myConfTable$overall["Accuracy"])
 # Accuracy 
# 0.8002049 

probmax <- NULL
for(i in 1:nrow(test)){
     probmax[i] <- max(nb_probs[i,1],nb_probs[i,2])      
 }
# Average Probability of Predicted Outcomes
 
results <- data.frame(test[[1]], nb_predictions, nb_probs,probmax)
(ave_prob <- mean(probmax))
# [1] 0.8867182

(acc <- (my_table[1]+my_table[4])/nrow(test))
plot(x=1:length(acc), y = acc, type = "o", col = "blue", xlab = "k", ylab = "Accuracy",
      main = "Performance Evaluation")
plot(x=1:length(ave_prob), y = ave_prob, type = "o", col = "blue", xlab = "k", ylab = "Average Probabilty",
      main = "Performance Evaluation")

# Accuracy value is 0.80, average probability of predicted outcomes value is 0.89.

# Applying Decision Tree Algorithm

treedata<-bank
treedata %>% mutate_if(is.character, as.factor) -> treedata
set.seed(1)
treeslice <- sample(0:1, size= nrow(treedata), prob = c(0.75,0.25), replace = TRUE)
trainingtree <- treedata[treeslice == 0, ]
testtree <- treedata[treeslice == 1, ]
library(rpart)

# Applying the rpart algorithm on all attributes
DTree1 <- rpart(formula = Attrition_Flag ~., data = trainingtree, 
                method = "class", parms = list(split = "gini"))
#predicting data based on class and prob (inorder to produce roc curve)
predict(DTree1, data = trainingtree, type = "class") -> training_pre1
predict(DTree1, data = trainingtree, type = "prob") -> training_pro1
predict(DTree1, newdata= testtree, type = "class") -> test_pre1
predict(DTree1, newdata = testtree, type = "prob") -> test_pro1

# Cross validation for confusion Matrix and sketching roc curves (AUC values)

library(pROC)
library(caret)
confusionMatrix(data = training_pre1, reference = trainingtree$Attrition_Flag)
#accuracy 0.9391
traintree_actual <- ifelse(trainingtree$Attrition_Flag == "Existing Customer", 1,0)
#auc 0.7930
roc <- roc(traintree_actual, training_pro1[,2], plot= TRUE, print.auc=TRUE)
#accuracy 0.9336
confusionMatrix(data = test_pre1, reference = testtree$Attrition_Flag)
testtree_actual <- ifelse(testtree$Attrition_Flag == "Attrited Customer", 1,0)
#auc 0.914
roc <- roc(testtree_actual, test_pro1[,2], plot = TRUE, print.auc = TRUE)

# Accuracy values are 0.93 for both test and training data sets, AUC values are approximately 0.79 and 0.91.

# Applying Random Forest Algorithm

set.seed(1)
#install.packages("randomForest")
library(randomForest)
Rforest <- randomForest(formula = Attrition_Flag ~., data = trainingtree)
print(Rforest)

# Cross validation for confusion Matrix and sketching roc curves (AUC values)
predict(Rforest, data = trainingtree, type = "class") -> trainingf_pre1
predict(Rforest, data = trainingtree, type = "prob") -> trainingf_pro1
predict(Rforest, newdata= testtree, type = "class") -> testf_pre1
predict(Rforest, newdata = testtree, type = "prob") -> testf_pro1

confusionMatrix(data = trainingf_pre1, reference = trainingtree$Attrition_Flag)
#accuracy 0.9612
traintreef_actual <- ifelse(trainingtree$Attrition_Flag == "Attrited Customer", 1,0)
#auc 0.988
roc <- roc(traintreef_actual, trainingf_pro1[,2], plot= TRUE, print.auc=TRUE)
#accuracy 0.9674
confusionMatrix(data = testf_pre1, reference = testtree$Attrition_Flag)
testtreef_actual <- ifelse(testtree$Attrition_Flag == "Attrited Customer", 1,0)
#auc 0.991
roc <- roc(testtreef_actual, testf_pro1[,2], plot = TRUE, print.auc = TRUE)

# Accuracy values are 0.96 for both test and training data sets, AUC values are approximately 0.99

