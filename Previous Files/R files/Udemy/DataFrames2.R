# Filtering DF

Stats[1:10,]
Filter = Stats$Internet.users < 2

Stats[Filter,]
Stats[Stats$Birth.rate > 40 & Stats$Internet.users < 2,]
Stats[Stats$Income.Group == "High income" & Stats$Internet.users > 90,]

Stats[Stats$Country.Name == "Malta",]

summary(Stats)
# Introduction to qplot()
library(ggplot2)


qplot(data = Stats, x = Internet.users, y = Birth.rate, size = I(4)
      , color = Income.Group)

?qplot

NewDataFrame = data.frame(Countries_2012_Dataset,
                          Codes_2012_Dataset,
                          Regions_2012_Dataset)
Stats[1:10,]
NewDataFrame[1:10,]

colnames(NewDataFrame) = c("Country.Name", "Country.Code", "Country.Region")

qplot(data = Stats, x = Internet.users, y = Birth.rate, color = NewDataFrame$Region)

NewDataFrame = data.frame(Country = Countries_2012_Dataset,
                          Code = Codes_2012_Dataset,
                          Region = Regions_2012_Dataset)


FullDataFrame = merge(Stats, NewDataFrame, by.x = "Country.Code", by.y = "Code")
FullDataFrame$Country = NULL

str(FullDataFrame)
colnames(FullDataFrame) = c("Country.Code", "Country.Name",
                            "Birth.Rate", "Internet.Users",
                            "Income.Group", "Country.Region")

qplot(data = FullDataFrame, x = Internet.Users,
      y = Birth.Rate, color = Country.Region,
      size = I(5), shape = I(19),
      alpha = I(0.7), main = "Birth Rate vs Internet Users")
