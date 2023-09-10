MovieRaitings = read.csv("Movie-Ratings.csv")

MovieRaitings[1:10,]

str(MovieRaitings)

colnames(MovieRaitings) = c("Film", "Genre",
                            "CriticRaiting", "AudienceRaiting",
                            "BudgetInMillions", "Year")

summary(MovieRaitings)

MovieRaitings[MovieRaitings$BudgetInMillions >= 200, c("Film", "BudgetInMillions")]

MovieRaitings$Year = factor(MovieRaitings$Year)

summary(MovieRaitings)

str(MovieRaitings)

?ggplot()

ggplot(data = MovieRaitings, aes(x = CriticRaiting, y = AudienceRaiting,
                                 color = Genre, size = BudgetInMillions)) +
  geom_point()


p = ggplot(data = MovieRaitings, aes(x = CriticRaiting, y = AudienceRaiting,
                                     color = Genre, size = BudgetInMillions))

p + geom_point()

p + geom_point(mapping = aes(size = CriticRaiting))
p + geom_point(mapping = aes(color = BudgetInMillions))
p + geom_point(aes(x = BudgetInMillions)) + xlab("Budget in Million $")


p = ggplot(data = MovieRaitings, aes(x = CriticRaiting, y = AudienceRaiting))
p + geom_point()

p + geom_point(aes(color = Genre))
p + geom_point(color = "DarkGreen")

p + geom_point(aes(size = BudgetInMillions, color = Genre))
p + geom_point(size = 7)

s = ggplot(data = MovieRaitings, aes(x = BudgetInMillions))
s + geom_histogram(binwidth = 10, aes(fill = Genre), color = "black")
s + geom_density(aes(fill = Genre), color = "black", position = "stack")
