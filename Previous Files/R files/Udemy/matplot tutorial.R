t(FieldGoals)

matplot(t(FieldGoals/FieldGoalAttempts), type="b", pch=14:17, col=c(1:10))
legend("bottomleft", inset=0.01, legend=Players, col=c(1:10), pch=14:17, horiz=F)
