# Method 2
setwd("D:\\Machine Learning\\R\\")

Stats = read.csv("DemographicData.csv")

head(Stats, n=7)

colnames(Stats)
rownames(Stats)

str(Stats)
summary(Stats)



# Using the $ sign
head(Stats)
Stats

Stats[3, 3]
Stats[3, "Birth.rate"]
Stats$Internet.users[2]
Stats[, "Internet.users"]
levels(Stats$Income.Group)

# Basic Operations with DF
Stats[1:10,] # subsetting

Stats[7,]

# Remember how the [] work:
Stats[7,]
is.data.frame(Stats[, 1, drop = F])

Stats[, 1, drop = F]

#multiply columns

Stats$NewColumn = Stats$Birth.rate * Stats$Internet.users
Stats[1:10,]

Stats$xyz = 1 : 5
Stats[1:10,]

# Remove column
Stats$NewColumn = NULL
Stats$xyz = NULL

Stats[1:10,]
