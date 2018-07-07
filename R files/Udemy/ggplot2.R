MovieRatings = read.csv("Movie-Ratings.csv")
colnames(MovieRatings) = c("Film", "Genre",
                           "CriticRating", "AudienceRating",
                           "BudgetInMillions", "Year")
PlotObject = ggplot(data = MovieRatings, aes(x = AudienceRating))
PlotObject + geom_histogram(binwidth = 10, fill = "White", color = "Blue")
PlotObject = ggplot(data = MovieRatings)
PlotObject + geom_histogram(binwidth = 10,
                            aes(x = CriticRating),
                            fill = "White", color = "Blue")

PlotObject = ggplot(data = MovieRatings,
                    aes(x = CriticRating, y = AudienceRating,
                        color = Genre))
PlotObject + geom_point() + geom_smooth(fill = NA)
PlotObject = ggplot(data = MovieRatings,
                    aes(x = Genre, y = AudienceRating,
                        color = Genre))
PlotObject + geom_jitter() + geom_boxplot(size = 1.2, alpha = 0.4)

PlotObject = ggplot(data = MovieRatings,
                    aes(x = Genre, y = CriticRating,
                        color = Genre))
PlotObject + geom_jitter() + geom_boxplot(size = 1.4, alpha = 0.4)

PlotObject = ggplot(data = MovieRatings,
                    aes(x = BudgetInMillions))
PlotObject + geom_histogram(binwidth = 10, aes(fill = Genre),
                            color = "Black") + facet_grid(Genre~., scales = "free")

PlotObject = ggplot(data = MovieRatings,
                    aes(x = CriticRating, y = AudienceRating, color = Genre))
PlotObject + geom_point(size = 3) + facet_grid(Genre~.)
PlotObject + geom_point(size = 3) + facet_grid(.~Year)
PlotObject + geom_point(size = 3) + facet_grid(Genre~Year)
PlotObject + geom_point(aes(size = BudgetInMillions)) + geom_smooth() + facet_grid(Genre~Year)

PlotObject = ggplot(data = MovieRatings,
                    aes(x = CriticRating, y = AudienceRating,
                        color = Genre, size = BudgetInMillions))
PlotObject + geom_point() + xlim(50, 100) + ylim(50, 100)
PlotObject + geom_point() + coord_cartesian(xlim = c(50, 100), ylim = c(50, 100))


PlotObject + geom_point(aes(size = BudgetInMillions)) +
  geom_smooth() + facet_grid(Genre~Year) + coord_cartesian(ylim = c(0, 100))
