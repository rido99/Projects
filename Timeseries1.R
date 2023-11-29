library(forecast)

#usunac stacjonarnosc i dopasowac proces ARMA albo ARIMA
ap <- scan(file = 'AirPass.txt')
ap <- ts(ap, start = c(2002, 1), end = c(2014, 2), frequency = 12)

tsdisplay(ap)

ap_sezon_el <- diff(ap, 12) #x_y - x_(t - 12)
tsdisplay(ap_sezon_el)

ap_st <- diff(ap_sezon_el)
tsdisplay(ap_st)

#Identyfikacja na podstawie autokorelacji za pomoca ruchomej średniej [na podst ACF]

acf(ap_st)
#h > 12 , wartosci miedzy linaimi czyli mozemy dopasowac ruchoma srednia rzedu 12
#pierwsza kreska to zero

#model na podst czesciowej autokorecji [model AR]

pacf(ap_st)

#H jest pawie 0 dla wszystkich h > 13 czyli mozemy dopasowac model AR(13)

#funkcja ar dopasowuje model AR(p) - tylko modele stacjonarne
ar.yw <- ar(ap_st, order.max = 13, aic = F) #metoda YW
ar.mle <- ar(ap_st, order.max = 13, method = 'mle') #metoda najwiekszej wiarygodnosci
print(ar.mle)

#oba modele daly takie same rzedy, inne wspolczynniki (wartosci nie takie same ale zblizone)

ar.optym.aic <- ar(ap_st, aic = T) #Metoda AIC
print(ar.optym.aic)

ar.aic <- ar.optym.aic$aic #róznice AIC(model optymalny) - AIC(model_AR(p))
print(ar.aic)
#najmniejsza wartosc dla p = 13 [liczba mniejsza od 1]
#min aic = aic1
#aic 13 - aic 1 = 0.63
#aic 21 - aic 1 = 8.19
#aic 21 > aic 13 czyli aic 21 chyba better

plot(as.numeric(names(ar.aic)), ar.aic,
     xlab = 'Rząd modelu autoregresji (p)',
     ylab = 'Porównanie kryterium AIC',
     type = "b")

#Funkcja ARIMA - bardziej uniwersalna

#Model MA(12) dla ap_st <-> ARIMA(0,1,12)(0,1,0)_12
#ARIMA(p, d, q)(P, D, Q)_S
#czesc sezonowa róznicujemy po S

m1 <- Arima(ap, order = c(0,1,12), seasonal = list(order = c(0,1,0), period = 12))

summary(m1)

#Model AR(13) dla ap_st <-> ARIMA(13,1,0)(0,1,0)_12

m2 <- Arima(ap, order = c(13,1,0), seasonal = list(order = c(0,1,0), period = 12))

summary(m2)

#AIC - m1
#AICC - m1
#BIC - m1

#ME - m2
#RMSE - m1
#MAE - m1
#MPE - m2
#MAPE - m1
#MAS - m1

#czyli wybieramy pierwszy

#Diagnostyka reszt m1 i m2

par(mar = c(1,1,1,1))
tsdiag(m1)
tsdiag(m2)

# w obydwu przypadkach reszty wygladaja ok
#hipoteza zerowa <- reszty sa losowe
#reszty powinny byc losowe
m1.resid <- residuals(m1)
m2.resid <- residuals(m2) #nie mozna odrzucic hipotezy o losowosci reszt

Box.test(m1.resid, type = 'Ljung-Box')
# p value 0.26 <- nie odrzucamy [bo p value duze]
Box.test(m2.resid, type = 'Ljung-Box')
# p value 0.99 <- nie odrzucamy

#dobrze zeby reezty mialy rozklad normalny
hist(m1.resid)
qqnorm(m1.resid)
qqline(m1.resid)

hist(m2.resid)
qqnorm(m2.resid)
qqline(m2.resid)

#wykresy kwantyli 3/10 wiec robimy test

shapiro.test(m1.resid)
# odrzucamy hipoteze
ks.test(m1.resid, 'pnorm')
# odrzucamy hipoteze ze reszty sa normalne

shapiro.test(m2.resid)
# odrzucamy hipoteze dalej mniejsze niz 0.05
ks.test(m2.resid, 'pnorm')
# odrzucamy

#wyciagamy wspolczynniki
coef1 <- m1$coef
coef2 <- m2$coef

coef.std.1 <- sqrt(diag(m1$var.coef))
coef.std.2 <- sqrt(diag(m2$var.coef))

ratio1 <- coef1 / coef.std.1/ 1.96
ratio2 <- coef2 / coef.std.2/ 1.96

ratio1

which(abs(ratio1) >= 1)

istotne <- which(abs(ratio1) >= 1)

m1.fixed <- numeric(12)
m1.fixed[istotne] <- NA 
#dopasowanie tam gdzie sa NA
m1.is <- Arima(ap, order = c(0,1,12), seasonal = list(order = c(0,1,0), period = 12),
               fixed = m1.fixed)
print(m1.is)

m1$aic - m1.is$aic
m1$aicc - m1.is$aicc
m1$bic - m1.is$bic

#ten model lepszy

#wybor automatyczny rzedu rownicowania d

d.opt <- ndiffs(ap) #wyzancza optymalne d
d.opt

D.opt <- nsdiffs(ap)
D.opt

#Automatyczny wybor modelu

ap.opt.aic <- auto.arima(ap, ic = 'aic')
ap.opt.aic

ap.opt.aicc <- auto.arima(ap, ic = 'aicc')
ap.opt.aicc

ap.opt.bic <- auto.arima(ap, ic = 'bic')
ap.opt.bic

