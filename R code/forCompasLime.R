library(semiArtificial)

# Load training set
train <- read.csv(file="../Data/compas_RBF_train.csv")
# Encode categorical features as factor
train$response <- as.factor(train$response)
train$two_year_recid <- as.factor(train$two_year_recid)

#############
# rbfDataGen
#############
generator <- rbfDataGen(response~., data = train)

# Generate data for explanation method
generated <- newdata(generator, size = 5000)
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/compas_RBF.csv", row.names = FALSE)

# Generate data for training of the adversarial model
generated <- newdata(generator, size = nrow(train))
# Remove the response column
generated <- generated[, -ncol(generated)]
write.csv(generated, file = "../Data/compas_adversarial_train_RBF.csv", row.names = FALSE)

################
# treEnsemble
################
forestGenerator <- treeEnsemble(response~., data = train)

# Generate data for explanation method
forestGenerated <- newdata(forestGenerator, size = 5000)
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/compas_forest.csv", row.names = FALSE)

# Generate data for training of the adversarial model
forestGenerated <- newdata(forestGenerator, size = nrow(train))
# Remove the response column
forestGenerated <- forestGenerated[, -ncol(forestGenerated)]
write.csv(forestGenerated, file = "../Data/compas_adversarial_train_forest.csv", row.names = FALSE)