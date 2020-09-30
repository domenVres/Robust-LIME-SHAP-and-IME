library(semiArtificial)

# Load the training set
train <- read.csv(file="../Data/german_RBF_train.csv")
# List of categorical features to be encoded as factor
categorical <- c('Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'MissedPayments', 'NoCurrentLoan',
                 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
                 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed', 'JobClassIsSkilled')
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}
train$response <- as.factor(train$response)

#############
# rbfDataGen
#############
generator <- rbfDataGen(response~., data = train)

# Generate data for explanation method
generated <- newdata(generator, size = 100)
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/german_RBF.csv", row.names = FALSE)

# Generate data for training of the adversarial model
generated <- newdata(generator, size = 100)
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/german_adversarial_train_RBF.csv", row.names = FALSE)

#
# RANDOM FOREST
#
forestGenerator <- treeEnsemble(response~., data = train)

# Generate data for explanation method
forestGenerated <- newdata(forestGenerator, size = 100)
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/german_forest.csv", row.names = FALSE)

# Generate data for training of the adversarial model
forestGenerated <- newdata(forestGenerator, size = 100)
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/german_adversarial_train_forest.csv", row.names = FALSE)

#################################################
# For RBF adversarial model at forest experiment
#################################################
train <- read.csv(file="../Data/german_forest_train.csv")

categorical <- c('Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'MissedPayments', 'NoCurrentLoan',
                 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
                 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed', 'JobClassIsSkilled')
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}
train$response <- as.factor(train$response)
generator <- rbfDataGen(response~., data = train)

generated <- newdata(generator, size = 100)
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/german_adversarial_train_RBF.csv", row.names = FALSE)