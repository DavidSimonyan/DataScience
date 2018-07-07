setwd("D:\\Data Science\\R files")
Movies = read.csv("Section6-Homework-Data.csv")
library("ggplot2")
Movies[1:2,]
str(Movies)

colnames(Movies) = c("DayOfWeek", "Director", "Genre", "Title",
                     "ReleaseDate", "Studio", "AdjustedGross", "Budget",
                     "Gross", "IMDBRating", "MovieLensRating", "Overseas",
                     "OverseasPercent", "Profit", "ProfitPercent", "Runtime",
                     "USProfit", "GrossPercentUS")

FilteredMovies = Movies[Movies$Genre %in% c("action", "adventure", "animation", "comedy", "drama") &
                        Movies$Studio %in% c("Buena Vista Studios", "Fox", "Paramount Pictures",
                                             "Sony", "Universal", "WB"),]
str(FilteredMovies)

MoviesPlot = ggplot(data = FilteredMovies, aes(x = Genre, y = GrossPercentUS))

MoviesPlot +
  geom_jitter(aes(size = Budget, color = Studio)) +
  geom_boxplot(alpha = 0.4, outlier.colour = NA) +
  xlab("Genre") +
  ylab("Gross % US") +
  ggtitle("Domestic Gross % by Genre") +
  theme(axis.title.x = element_text(color = "Blue", size = 31),
        axis.title.y = element_text(color = "Blue", size = 31),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 20),
        plot.title = element_text(size = 40),
        legend.title = element_text(size = 20),
        legend.text = element_text(size = 20))


summary(Movies$Genre)
