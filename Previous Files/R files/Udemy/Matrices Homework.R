
DrawPlot = function(Data, rows=1:nrow(Data), columns=1:ncol(Data))
{
  Subset=Data[rows,columns,drop=F]
  matplot(t(Subset), type="b", pch=14:17, col=c(1:10))
  legend("bottomleft", inset=0.01, legend=Players[rows], col=c(1:10), pch=14:17, horiz=F)
}

DrawPlot(FreeThrowAttempts / Games)
DrawPlot(FreeThrows / FreeThrowAttempts)
DrawPlot((Points - FreeThrows) / FieldGoals)
