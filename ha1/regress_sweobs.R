# open the SweObs data
swe_obs <- data.frame(read.csv("SweObs.csv",header=FALSE))
colnames(swe_obs) <- c("lon","lat","elevation","dcoast", "dcoastswe","y")

# consider various models.
# A: y ~ lat + elevation + dcoast + dcoastswe
fitA <- lm(y ~ lat + elevation + dcoast + dcoastswe, data = swe_obs)
sA <- summary(fitA)
print(sA)
# B: y ~ lat + elevation + dcoastswe
fitB <- lm(y ~ lat + elevation + dcoastswe, data=swe_obs)
sB <- summary(fitB)
print(sB)
# C: y ~ lat + elevation + docast
fitC <- lm(y ~ lat + elevation + dcoast, data=swe_obs)
sC <- summary(fitC)
print(sC)
# D: y ~ lat + dcoast + dcoastswe
fitD <- lm(y ~ lat + dcoast + dcoastswe, data=swe_obs)
sD <- summary(fitD)
print(sD)
# E: y ~ lat + elevation
fitE <- lm(y ~ lat + elevation, data=swe_obs)
sE <- summary(fitE)
print(sE)
plot.new()
plot(c(sA$sigma, sB$sigma, sC$sigma, sD$sigma, sE$sigma), ylab="SE", xaxt="n", xlab="Models", main="SE of residuals for the models")
axis(1, at=1:5, labels=c("A","B","C","D","E"))

plot.new()
aics <- c(AIC(fitA),AIC(fitB),AIC(fitC),AIC(fitD),AIC(fitE))
plot(aics, ylab="AIC", xaxt="n", xlab="Models", main="AIC of the models")
axis(1, at=1:5, labels=c("A","B","C","D","E"))