# load in packages we'll use
library(readr)
library(tidyverse) # utility functions
library(rpart) # for regression trees
library(randomForest) # for random forests

# read the data and store data in DataFrame titled melbourne_data
melbourne_data <- read_csv("input/melb_data.csv")

summary(melbourne_data)

# The Melbourne data has some missing values (some houses for which some variables 
# weren't recorded.) We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data <- na.omit(melbourne_data)

# train a decision tree based on our dataset 
fit <- rpart(Price ~ Rooms + Bathroom + Landsize + BuildingArea +
               YearBuilt + Lattitude + Longtitude, data = melbourne_data)

# plot our regression tree 
plot(fit, uniform=TRUE)
# add text labels & make them 60% as big as they are by default
text(fit, cex=.6)

print("Making predictions for the following 5 houses:")
print(head(melbourne_data))

print("The predictions are")
print(predict(fit, head(melbourne_data)))

print("Actual price")
print(head(melbourne_data$Price))

# package with the mae function
library(modelr)

# get the mean average error for our model
mae(model = fit, data = melbourne_data)

# split our data so that 30% is in the test set and 70% is in the training set
splitData <- resample_partition(melbourne_data, c(test = 0.3, train = 0.7))

# how many cases are in test & training set? 
lapply(splitData, dim)

# fit a new model to our training set
fit2 <- rpart(Price ~ Rooms + Bathroom + Landsize + BuildingArea +
                YearBuilt + Lattitude + Longtitude, data = splitData$train)

# get the mean average error for our new model, based on our test data
mae(model = fit2, data = splitData$test)

#using random forests
library(randomForest)

fitRandomForest <- randomForest(Price ~ Rooms + Bathroom + Landsize + BuildingArea + YearBuilt + Lattitude + Longtitude
                                , data = splitData$train)

mae(model = fitRandomForest, data = splitData$test)
