library(semiArtificial)

# Load train set and mark categorical features
train <- read.csv(file="../Data/compas_forest_train.csv")
train$response <- as.factor(train$response)
train$two_year_recid <- as.factor(train$two_year_recid)

# Training of generator on training set
forestGen <- treeEnsemble(response~., data = train)

#######
# SHAP
#######
# Same for test set
test <- read.csv(file="../Data/compas_forest_test.csv")
test$response <- as.factor(test$response)
test$two_year_recid <- as.factor(test$two_year_recid)

# Repeat every test set instance 100 times
test <- test[rep(seq_len(nrow(test)), each = 100), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/compas_forest_shap.csv", row.names = FALSE)

# Generate data for training of the adversarial model (this is not locally generated)
forestGenerated <- newdata(forestGen, size = 100)
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/compas_shap_adversarial_train_forest.csv", row.names = FALSE)

#######
# LIME
#######
test <- read.csv(file="../Data/compas_forest_test.csv")
test$response <- as.factor(test$response)
test$two_year_recid <- as.factor(test$two_year_recid)

# Repeat every test set instance 100 times
test <- test[rep(seq_len(nrow(test)), each = 5000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/compas_forest_lime.csv", row.names = FALSE)

# Generate data for training of the adversarial model (this is not locally generated)
forestGenerated <- newdata(forestGen, size = nrow(train))
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/compas_lime_adversarial_train_forest.csv", row.names = FALSE)

#######
# IME
#######
test <- read.csv(file="../Data/compas_forest_test.csv")
test$response <- as.factor(test$response)
test$two_year_recid <- as.factor(test$two_year_recid)

# Repeat every test set instance 100 times
test <- test[rep(seq_len(nrow(test)), each = 1000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

# Generate data with filling NA values
generatedData <- newdata(forestGen, fillData = test)

# Remove response column
generatedData <- generatedData[, -ncol(generatedData)]

# Save data
write.csv(generatedData, file = "../Data/compas_forest_ime.csv", row.names = FALSE)

# # Generate data for training of the adversarial model (locally generated)
for (col in 1:ncol(train))
  train[sample.int(nrow(train),size=0.5*nrow(train)),col] <- NA

forestGenerated <- newdata(forestGen, fillData = train)
# Remove response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/compas_ime_adversarial_train_forest.csv", row.names = FALSE)