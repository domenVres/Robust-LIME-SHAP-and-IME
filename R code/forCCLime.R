library(semiArtificial)

# Load the training data
train <- read.csv(file="../Data/cc_RBF_train.csv")
# List of categorical features to be encoded as factor
categorical <- c("unrelated_column_one", "unrelated_column_two", "response")
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}

#############
# rbfDataGen
#############
generator <- rbfDataGen(response~., data = train)

# Generate data for explanation method
generated <- newdata(generator, size = 5000)
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/cc_RBF.csv", row.names = FALSE)

# Generate data for training of the adversarial model
generated <- newdata(generator, size = nrow(train))
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/cc_adversarial_train_RBF.csv", row.names = FALSE)

###############
# treeEnsemble
###############
forestGenerator <- treeEnsemble(response~., data = train)

# Generate data for explanation method
forestGenerated <- newdata(forestGenerator, size = 5000)
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/cc_forest.csv", row.names = FALSE)

# Generate data for training of the adversarial model
forestGenerated <- newdata(forestGenerator, size = nrow(train))
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/cc_adversarial_train_forest.csv", row.names = FALSE)