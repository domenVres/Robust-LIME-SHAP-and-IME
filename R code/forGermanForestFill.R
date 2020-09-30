library(semiArtificial)

# Load the training set
train <- read.csv(file="../Data/german_forest_train.csv")

# List of categorical features to be encoded as factor
categorical <- c('Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'MissedPayments', 'NoCurrentLoan',
                 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
                 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed', 'JobClassIsSkilled')
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}
train$response <- as.factor(train$response)

# Train generator on training set
forestGen <- treeEnsemble(response~., data = train)

#######
# SHAP
#######
# Load test set
test <- read.csv(file="../Data/german_forest_test.csv")

# Encode categorical features as factor
for (feature in categorical) {
  test[, feature] <- as.factor(test[, feature])
}
test$response <- as.factor(test$response)

# Repeat every test set instance 100 times
test <- test[rep(seq_len(nrow(test)), each = 100), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/german_forest_shap.csv", row.names = FALSE)

# Generate data for training of the adversarial model (this is not locally generated)
forestGenerated <- newdata(forestGen, size = 100)
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/german_shap_adversarial_train_forest.csv", row.names = FALSE)

#######
# LIME
#######
# Load test set
test <- read.csv(file="../Data/german_forest_test.csv")

# Encode categorical features as factor
for (feature in categorical) {
  test[, feature] <- as.factor(test[, feature])
}
test$response <- as.factor(test$response)

# Repeat every test set instance 5000 times
test <- test[rep(seq_len(nrow(test)), each = 5000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/german_forest_lime.csv", row.names = FALSE)

# Generate data for training of the adversarial model (this is not locally generated)
forestGenerated <- newdata(forestGen, size = nrow(train))
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/german_lime_adversarial_train_forest.csv", row.names = FALSE)

#######
# IME
#######
test <- read.csv(file="../Data/german_forest_test.csv")

# Encode categorical features as factor
for (feature in categorical) {
  test[, feature] <- as.factor(test[, feature])
}
test$response <- as.factor(test$response)

# Repeat every test set instance 1000 times
test <- test[rep(seq_len(nrow(test)), each = 1000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/german_forest_ime.csv", row.names = FALSE)

# Generate data for training of the adversarial model (locally generated)
for (col in 1:ncol(train))
  train[sample.int(nrow(train),size=0.5*nrow(train)),col] <- NA

forestGenerated <- newdata(forestGen, fillData = train)
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/german_ime_adversarial_train_forest.csv", row.names = FALSE)