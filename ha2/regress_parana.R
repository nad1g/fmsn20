# open the SweObs data
parana <- data.frame(read.csv("parana.csv",header=FALSE))
colnames(parana) <- c("y", "lon","lat","dist", "elevation")

# consider various models.
# A: y ~ lat + lon + dist + elev
fitA <- lm(log(y) ~ lat + lon + dist + elevation, data = parana)
sA <- summary(fitA)
print(sA)

# B: y ~ lon + dist + elev
fitB <- lm(log(y) ~ lon + dist + elevation, data = parana)
sB <- summary(fitB)
print(sB)

# C: y ~ dist + elev
fitC <- lm(log(y) ~ dist + elevation, data = parana)
sC <- summary(fitC)
print(sC)

# D: y ~  lon + dist 
fitD <- lm(log(y) ~ lon + dist , data = parana)
sD <- summary(fitD)
print(sD)

# E: y ~  lon +  elev
fitE <- lm(log(y) ~ lon + elevation, data = parana)
sE <- summary(fitE)
print(sE)
plot.new()
plot(c(sA$sigma, sB$sigma, sC$sigma, sD$sigma, sE$sigma), ylab="SE", xaxt="n", xlab="Models", main="SE of residuals for the models")
axis(1, at=1:5, labels=c("A","B","C","D","E"))

plot.new()
aics <- c(AIC(fitA),AIC(fitB),AIC(fitC),AIC(fitD),AIC(fitE))
plot(aics, ylab="AIC", xaxt="n", xlab="Models", main="AIC of the models")
axis(1, at=1:5, labels=c("A","B","C","D","E"))