install.packages("ggplot2")
install.packages("readr")
install.packages("gmodels")
install.packages("car")
install.packages("ResourceSelection")
install.packages("caret")
install.packages("ROCR")
install.packages("rpart")       
install.packages("rpart.plot")  
install.packages("randomForest")
install.packages("pROC")
install.packages("xgboost")
install.packages("Matrix")



library(xgboost)
library(caret)
library(ResourceSelection)
library(car)
library(readr)
library(ggplot2)
library(gmodels)
library(ROCR)
library(Matrix)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)



# Reading the CSV file
bank <- read_delim("bank-additional.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)

# Describing the Dataset
str(bank)

# Data Cleaning
# Converting the Categorical Variables to factors

bank$job = as.factor(bank$job)
bank$education = as.factor(bank$education)
bank$marital = as.factor(bank$marital)
bank$default = as.factor(bank$default)
bank$housing = as.factor(bank$housing)
bank$loan = as.factor(bank$loan)
bank$contact = as.factor(bank$contact)
bank$month = as.factor(bank$month)
bank$day_of_week = as.factor(bank$day_of_week)
bank$poutcome = as.factor(bank$poutcome)
bank$y = as.factor(bank$y)

# Producing the summary of Bank Dataset
summary(bank)

# Finding Duplicate
sum(duplicated(bank))

# Finding NA Values
sum(is.na(bank$age))
sum(is.na(bank$campaign))
sum(is.na(bank$pdays))
sum(is.na(bank$previous))
sum(is.na(bank$emp.var.rate))
sum(is.na(bank$cons.price.idx))
sum(is.na(bank$cons.conf.idx))
sum(is.na(bank$nr.employed))

# Performing Exploratory Data Analysis

# Job Variable
ggplot(bank,aes(job))+geom_bar(aes(fill= y),position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Marital Variable
ggplot(bank,aes(marital))+geom_bar(aes(fill= y),position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Education
ggplot(bank,aes(education))+geom_bar(aes(fill= y), position = position_dodge()) + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Age boxplot
ggplot(bank,aes(x = y,y = age))+geom_boxplot(aes(fill= y))+xlab("Subscribed")

# Day_of_week Barplot
ggplot(bank,aes(day_of_week))+geom_bar(aes(fill= y), position = position_dodge())+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Month Barplot
ggplot(bank,aes(month))+geom_bar(aes(fill= y), position = position_dodge())+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Housing Barplot
ggplot(bank,aes(housing))+geom_bar(aes(fill= y), position = position_dodge())+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Finding Cross Tables for Unkown Categories

# Cross Table for job 
CrossTable(bank$job, bank$y)

# Cross Table for Marital Status
CrossTable(bank$marital, bank$y)

# Cross Table for Education
CrossTable(bank$education, bank$y)

# Chi Square Test for Education
chisq.test(bank$education, bank$y)

# Cross Table for Housing
CrossTable(bank$housing, bank$y)

# Chi-Square Test for Housing
chisq.test(bank$housing, bank$y)

# Cross Table for Personal Loan
CrossTable(bank$loan, bank$y)

# Chi-Square Test for Personal Loan
chisq.test(bank$loan, bank$y)

# Modifying P_days
table(bank$pdays)

bank$pdays_1<-ifelse(bank$pdays==999,0,1)

table(bank$pdays_1)

# Defining if pdays_1 is significant
CrossTable(bank$pdays_1, bank$y)

# Chi-Square Test for Pdays_1
chisq.test(bank$pdays_1, bank$y)

# Overview of data
plot(bank$y)

# Remove from the report
# Scatter Plots for Numeric Variables
pairs1 = bank[, c("y","age","duration","campaign","previous","previous",
                  "emp.var.rate","cons.price.idx","cons.conf.idx")]

pairs(pairs1)

# Scatter Plot for Categorical Variables
pairs2 = bank[, c("y","job","marital","education","default","housing","loan",
                  "contact","month","day_of_week","poutcome","pdays_1")]

pairs(pairs2)

# Data Cleaning 
# Removing the Unknown Variables from factor variables
bank_clean <- bank
bank_clean <- subset(bank_clean, bank_clean$job != "unknown")
bank_clean <- subset(bank_clean, bank_clean$education != "unknown")
bank_clean <- subset(bank_clean, bank_clean$marital != "unknown")
bank_clean = subset(bank_clean, select = -c(age, duration, default, day_of_week, housing, loan, pdays))

table(bank$y)
table(bank_clean$y)

# Logistic Regression Model with all variables
Model_1 = glm(data = bank, formula = y ~ .-pdays, family = binomial)

summary(Model_1)

# Creating Training and Testing Samples
set.seed(1)
row.number = sample(1:nrow(bank_clean), 0.8*nrow(bank_clean))
train_bank = bank_clean[row.number,]
test_bank = bank_clean[-row.number,]

summary(train_bank)
summary(test_bank)

Model_2 = glm(formula = y ~ ., data = train_bank, family = binomial)

summary(Model_2)

# Checking Co linearity
vif(Model_2)

# Stepwise selection of the variables
glm.null.train_bank = glm(y ~ 1, data = train_bank, family = "binomial")
glm.full.train_bank = glm(y ~ ., 
                          data = train_bank, family = "binomial")

step.AIC1 = step(glm.null.train_bank, scope = list(upper=glm.full.train_bank),
                 direction ="both",test ="Chisq", trace = F)


step.AIC1

summary(step.AIC1)

selected_features <- all.vars(step.AIC1$formula)[-1]

selected_features

hoslem.test(step.AIC1$y, fitted(step.AIC1), g=10)

test_bank$PredProb = predict.glm(step.AIC1, newdata = test_bank, type = "response")

test_bank$PredSub = ifelse(test_bank$PredProb >= 0.5, "yes", "no")

table(test_bank$PredSub)

caret::confusionMatrix(test_bank$y,as.factor(test_bank$PredSub))


PredProb1 = prediction(predict.glm(step.AIC1, newdata = test_bank, type = "response"), test_bank$y)

# Computing threshold for cutoff to best trade off sensitivity and specificity
plot(unlist(performance(PredProb1,'sens')@x.values),unlist(performance(PredProb1,'sens')@y.values), type='l', lwd=2, ylab = "", xlab = 'Cutoff')
mtext('Sensitivity',side=2)
mtext('Sensitivity vs. Specificity Plot for AIC Model', side=3)

# Second specificity in same plot
par(new=TRUE)
plot(unlist(performance(PredProb1,'spec')@x.values),unlist(performance(PredProb1,'spec')@y.values), type='l', lwd=2,col='red', ylab = "", xlab = 'Cutoff')
axis(4,at=seq(0,1,0.2)) 
mtext('Specificity',side=4, col='red')

par(new=TRUE)

min.diff <-which.min(abs(unlist(performance(PredProb1, "sens")@y.values) - unlist(performance(PredProb1, "spec")@y.values)))
min.x<-unlist(performance(PredProb1, "sens")@x.values)[min.diff]
min.y<-unlist(performance(PredProb1, "spec")@y.values)[min.diff]
optimal <-min.x

abline(h = min.y, lty = 3)
abline(v = min.x, lty = 3)
text(min.x,0,paste("optimal threshold=",round(optimal,5)), pos = 4)

test_bank$PredSubOptimal = ifelse(test_bank$PredProb >= 0.08, "yes", "no")


table(test_bank$PredSubOptimal)

caret::confusionMatrix(test_bank$y,as.factor(test_bank$PredSubOptimal))

# XG Boost
set.seed(1)
row.number = sample(1:nrow(bank_clean), 0.8*nrow(bank_clean))
train_bank = bank_clean[row.number,]
test_bank = bank_clean[-row.number,]

# Create Train & Test Matrices with Selected Features
train_matrix_selected <- model.matrix(as.formula(paste("y ~", paste(selected_features, collapse = " + "))), data = train_bank)[, -1]
test_matrix_selected <- model.matrix(as.formula(paste("y ~", paste(selected_features, collapse = " + "))), data = test_bank)[, -1]

# Ensure Column Names Match
colnames(test_matrix_selected) <- colnames(train_matrix_selected)

# Convert Target Variable to Binary (0 = no, 1 = yes)
train_labels <- ifelse(train_bank$y == "yes", 1, 0)
test_labels <- ifelse(test_bank$y == "yes", 1, 0)

# Convert Data into XGBoost DMatrix Format
dtrain_selected <- xgb.DMatrix(data = train_matrix_selected, label = train_labels)
dtest_selected <- xgb.DMatrix(data = test_matrix_selected, label = test_labels)


ncol(train_matrix_selected)
ncol(test_matrix_selected)

table(train_bank$y)

pos_weight <- sum(train_labels == 0) / sum(train_labels == 1)
pos_weight

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = pos_weight  # Adjusts for class imbalance
)

# Train the model with 100 boosting rounds
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  verbose = 1
)

xgb_probs <- predict(xgb_model, dtest)

roc_curve <- roc(test_labels, xgb_probs)  # test_labels = binary {0,1} for "no" and "yes"
plot(roc_curve, col = "blue", main = "ROC Curve for XGBoost")
roc_data <- coords(roc_curve, "best", ret = c("threshold", "sensitivity", "specificity"))
optimal_cutoff <- roc_data["threshold"]
optimal_cutoff


xgb_pred <- ifelse(xgb_probs >= 0.444809, "yes", "no")
xgb_pred <- factor(xgb_pred, levels = levels(test_bank$y))

# Compute Confusion Matrix
conf_matrix_xgb <- confusionMatrix(xgb_pred, test_bank$y)
print(conf_matrix_xgb)

# Random Forest
# Set seed for reproducibility
set.seed(1)

selected_formula <- as.formula(paste("y ~", paste(selected_features, collapse = " + ")))

# Train Random Forest Model with Selected Features
rf_model <- randomForest(selected_formula, data = train_bank, 
                         ntree = 200,       # Number of trees
                         mtry = 4,          # Number of features per split
                         importance = TRUE, # Compute feature importance
                         sampsize = c(300, 300)) # Balanced sampling

# Print Model Summary
print(rf_model)
rf_probs <- predict(rf_model, newdata = test_bank, type = "prob")
roc_curve_rf <- roc(test_bank$y, rf_probs[, "yes"])  # Compute ROC Curve

# Plot ROC Curve
plot(roc_curve_rf, col = "red", main = "ROC Curve for Random Forest")

roc_data_rf <- coords(roc_curve_rf, "best", ret = c("threshold", "sensitivity", "specificity"))

# Extract Optimal Probability Cutoff
optimal_cutoff_rf <- roc_data_rf["threshold"]
optimal_cutoff_rf

rf_pred_adjusted <- ifelse(rf_probs[,"yes"]>= 0.495,"yes","no")
rf_pred_adjusted <- factor(rf_pred_adjusted, levels = levels(test_bank$y))
conf_matrix_rf_adjusted <- confusionMatrix(rf_pred_adjusted, test_bank$y)
print(conf_matrix_rf_adjusted)

varImpPlot(rf_model, main = "Feature Importance in Random Forest")
