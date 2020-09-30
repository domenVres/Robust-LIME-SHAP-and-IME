library(semiArtificial)

#########
# Compas
#########
# Load train data and transform one hot encoded variables back to their original values
train <- read.csv(file="../Podatki/IME/compas_train.csv")
new <- integer(nrow(train))
for (i in 1:nrow(train)) {
  if(train[i, "c_charge_degree_F"] == 1) {
    new[i] = "F"
  }
  else {
    new[i] = "M"
  }
}
train <- train[, -which(names(train) %in% c("c_charge_degree_F", "c_charge_degree_M"))]
train$c_charge_degree <- as.factor(new)

sex <- integer(nrow(train))
for (i in 1:nrow(train)) {
  if(train[i, "sex_Female"] == 1) {
    sex[i] = "Female"
  }
  else {
    sex[i] = "Male"
  }
}
train <- train[, -which(names(train) %in% c("sex_Female", "sex_Male"))]
train$sex <- as.factor(sex)
train$y <- as.factor(train$y)
train$two_year_recid <- as.factor(train$two_year_recid)

# Do the same for test data
test <- read.csv(file="../Podatki/IME/compas_test.csv")
new <- integer(nrow(test))
for (i in 1:nrow(test)) {
  if(test[i, "c_charge_degree_F"] == 1) {
    new[i] = "F"
  }
  else {
    new[i] = "M"
  }
}
test <- test[, -which(names(test) %in% c("c_charge_degree_F", "c_charge_degree_M"))]
test$c_charge_degree <- as.factor(new)

sex <- integer(nrow(test))
for (i in 1:nrow(test)) {
  if(test[i, "sex_Female"] == 1) {
    sex[i] = "Female"
  }
  else {
    sex[i] = "Male"
  }
}
test <- test[, -which(names(test) %in% c("sex_Female", "sex_Male"))]
test$sex <- as.factor(sex)

# Encode rest of the categorical features as factor
test$y <- as.factor(test$y)
test$two_year_recid <- as.factor(test$two_year_recid)

# Generator training
forestGen <- treeEnsemble(y~., data = train)

# Generate new data
test <- test[rep(seq_len(nrow(test)), each = 1000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

generatedData <- newdata(forestGen, fillData = test)
# Drop y column
test <- test[, -which(names(test) %in% c("y"))]
# Save generated data
write.csv(generatedData, file = "../Podatki/IME/compas_forest_generated.csv", row.names = FALSE)

###########
# German
###########
# Load train data and transform one hot encoded variables back to their original values
train <- read.csv(file="../Podatki/IME/german_train.csv")
balance <- integer(nrow(train))
for (i in 1:nrow(train)) {
  if(train[i, "CheckingAccountBalance_lt_0"] == 1) {
    balance[i] = "lt_0"
  }
  else if(train[i, "CheckingAccountBalance_geq_0_lt_200"] == 1) {
    balance[i] = "geq_0_lt_200"
  }
  else {
    balance[i] = "geq_200"
  }
}
train <- train[, -which(names(train) %in% c("CheckingAccountBalance_lt_0", "CheckingAccountBalance_geq_0_lt_200", "CheckingAccountBalance_geq_200"))]
train$CheckingAccountBalance <- as.factor(balance)

balance <- integer(nrow(train))
for (i in 1:nrow(train)) {
  if(train[i, "SavingsAccountBalance_lt_100"] == 1) {
    balance[i] = "lt_100"
  }
  else if(train[i, "SavingsAccountBalance_geq_100_lt_500"] == 1) {
    balance[i] = "geq_100_lt_500"
  }
  else {
    balance[i] = "geq_500"
  }
}
train <- train[, -which(names(train) %in% c("SavingsAccountBalance_lt_100", "SavingsAccountBalance_geq_100_lt_500", "SavingsAccountBalance_geq_500"))]
train$SavingsAccountBalance <- as.factor(balance)

job <- integer(nrow(train))
for (i in 1:nrow(train)) {
  if(train[i, "YearsAtCurrentJob_lt_1"] == 1) {
    job[i] = "lt_1"
  }
  else if(train[i, "YearsAtCurrentJob_geq_1_lt_4"] == 1) {
    job[i] = "geq_1_lt_4"
  }
  else {
    job[i] = "geq_4"
  }
}
train <- train[, -which(names(train) %in% c("YearsAtCurrentJob_lt_1", "YearsAtCurrentJob_geq_1_lt_4", "YearsAtCurrentJob_geq_4"))]
train$YearsAtCurrentJob <- as.factor(job)

# Encode rest of the categorical features as factor
train$y <- as.factor(train$y)
# List of categorical features to be encoded as factor
categorical <- c('Gender', 'ForeignWorker', 'Single', 'HasTelephone', 'MissedPayments', 'NoCurrentLoan',
                 'CriticalAccountOrLoansElsewhere', 'OtherLoansAtBank', 'OtherLoansAtStore', 'HasCoapplicant',
                 'HasGuarantor', 'OwnsHouse', 'RentsHouse', 'Unemployed', 'JobClassIsSkilled')
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}

# Do the same for test data
test <- read.csv(file="../Podatki/IME/german_test.csv")
balance <- integer(nrow(test))
for (i in 1:nrow(test)) {
  if(test[i, "CheckingAccountBalance_lt_0"] == 1) {
    balance[i] = "lt_0"
  }
  else if(test[i, "CheckingAccountBalance_geq_0_lt_200"] == 1) {
    balance[i] = "geq_0_lt_200"
  }
  else {
    balance[i] = "geq_200"
  }
}
test <- test[, -which(names(test) %in% c("CheckingAccountBalance_lt_0", "CheckingAccountBalance_geq_0_lt_200", "CheckingAccountBalance_geq_200"))]
test$CheckingAccountBalance <- as.factor(balance)

balance <- integer(nrow(test))
for (i in 1:nrow(test)) {
  if(test[i, "SavingsAccountBalance_lt_100"] == 1) {
    balance[i] = "lt_100"
  }
  else if(test[i, "SavingsAccountBalance_geq_100_lt_500"] == 1) {
    balance[i] = "geq_100_lt_500"
  }
  else {
    balance[i] = "geq_500"
  }
}
test <- test[, -which(names(test) %in% c("SavingsAccountBalance_lt_100", "SavingsAccountBalance_geq_100_lt_500", "SavingsAccountBalance_geq_500"))]
test$SavingsAccountBalance <- as.factor(balance)

job <- integer(nrow(test))
for (i in 1:nrow(test)) {
  if(test[i, "YearsAtCurrentJob_lt_1"] == 1) {
    job[i] = "lt_1"
  }
  else if(test[i, "YearsAtCurrentJob_geq_1_lt_4"] == 1) {
    job[i] = "geq_1_lt_4"
  }
  else {
    job[i] = "geq_4"
  }
}
test <- test[, -which(names(test) %in% c("YearsAtCurrentJob_lt_1", "YearsAtCurrentJob_geq_1_lt_4", "YearsAtCurrentJob_geq_4"))]
test$YearsAtCurrentJob <- as.factor(job)

test$y <- as.factor(test$y)
for (feature in categorical) {
  test[, feature] <- as.factor(test[, feature])
}

# Generator training
forestGen <- treeEnsemble(y~., data = train)

# Generate new data
test <- test[rep(seq_len(nrow(test)), each = 1000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

generatedData <- newdata(forestGen, fillData = test)
# Drop y row
generatedData <- generatedData[, -which(names(generatedData) %in% c("y"))]
# Save generated data
write.csv(generatedData, file = "../Podatki/IME/german_forest_generated.csv", row.names = FALSE)

#####
# CC
#####
# Load training set and encode categorical features as factor
train <- read.csv(file="../Podatki/IME/cc_train.csv")

categorical <- c("unrelated_column_one", "unrelated_column_two", "y")
for (feature in categorical) {
  train[, feature] <- as.factor(train[, feature])
}

# Same for test set
test <- read.csv(file="../Podatki/IME/cc_test.csv")

for (feature in categorical) {
  test[, feature] <- as.factor(test[, feature])
}

# Generator training
forestGen <- treeEnsemble(y~., data = train)

# Generate new data
test <- test[rep(seq_len(nrow(test)), each = 1000), ]
for (col in 1:ncol(test))
  test[sample.int(nrow(test),size=0.5*nrow(test)),col] <- NA

generatedData <- newdata(forestGen, fillData = test)
# Drop y row
generatedData <- generatedData[, -which(names(generatedData) %in% c("y"))]
# Save generated data
write.csv(generatedData, file = "../Podatki/IME/cc_forest_generated.csv", row.names = FALSE)