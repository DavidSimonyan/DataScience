DataCSV = read.csv("Section5-Homework-Data.csv")
DataCSV[1:10,]

str(DataCSV)
rm(DataCSV)

DataFertilityRate_1960 = DataCSV[DataCSV$Year == 1960,]
DataFertilityRate_2013 = DataCSV[DataCSV$Year == 2013,]
DataLifeExpectancy_1960 = data.frame(Country.Code = Country_Code, Life.Extectancy = Life_Expectancy_At_Birth_1960)
DataLifeExpectancy_2013 = data.frame(Country.Code = Country_Code, Life.Extectancy = Life_Expectancy_At_Birth_2013)

Data1960 = merge(DataFertilityRate_1960, DataLifeExpectancy_1960,
                 by.x = "Country.Code", by.y = "Country.Code")

Data1960[1:10,]

Data2013 = merge(DataFertilityRate_2013, DataLifeExpectancy_2013,
                 by.x = "Country.Code", by.y = "Country.Code")

Data2013[1:10,]

rm(DataFertilityRate_1960)
rm(DataFertilityRate_2013)
rm(DataLifeExpectancy_1960)
rm(DataLifeExpectancy_2013)

qplot(data = Data1960, x = Fertility.Rate, y = Life.Expectancy,
      size = I(4), alpha = I(0.4), color = Region, main = "Graph")
