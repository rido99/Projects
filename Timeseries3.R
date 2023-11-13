#funkcja window - wybpr podzbiory danych

library(expsmooth) # dane usgdp

data("usgdp")

tsdisplay(usgdp)

# Różnicowanie z opóźnieniem 1
usgdp.diff<-diff(usgdp)
tsdisplay(usgdp.diff)

# Różnicowanie z opóźnieniem 3
usgdp.diff3<-diff(usgdp,3)
tsdisplay(usgdp.diff3)

# Różnicowanie z opóźnieniem sezonowym
ap<-scan(file = "AirPass.txt")

ap<-ts(ap,start=c(2002,1),end=c(2014,2),frequency = 12)

cycle(ap)

tsp(ap)
ap.2002.2010 <- window(ap, start=c(2002,1),end = c(2010,12))
deltat(ap)

ap.train <- window(ap, end = 2010,12)
ap.test <- window(ap, start=c(2011,1))

library(quantmod)

#http://finance.yahoo.com
getSymbols(Symbols = "ETH-USD", src = "yahoo", from = '2020-01-01', to = '2022-01-01')
getSymbols(Symbols = "BTC-USD", src = "yahoo", from = '2020-01-01', to = '2022-01-01')

#pobrać IBM
`ETH-USD`[1:3,]
ETH_USD.close <- `ETH-USD`$`ETH-USD.Close`
`ETH_USD.close`[1:3,]

BTC_USD.close <- `BTC-USD`$`BTC-USD.Close`
par(mfrow = c(2,1))
par(mar = c(1,1,1,1))
plot(ETH_USD.close)
plot(BTC_USD.close)

library(tseries)

BTC <- get.hist.quote(instrument = "BTC-USD",
                      quote = c("Open", "High", "Low", "Close"), 
                      provider = 'yahoo',
                      start = '2020-01-01', end = '2020-12-01')

ETH <- get.hist.quote(instrument = "ETH-USD",
                      quote = c("Open", "High", "Low", "Close"), 
                      provider = 'yahoo',
                      start = '2020-01-01', end = '2020-12-01')

BTC[1:3, ]
ETH[1:3, ]

plot(BTC)
plot(ETH)

#wykresy szeregow czasowych
library(lattice)
xyplot(ap)

#aspekt - stosunek szerokosci do dlugosci
xyplot(ap, aspect = 1/4)

#Dla dlugich szeregow - wykresy panelowe
xyplot(ap, strip = TRUE, cut = list(number = 3, overlap = 0.5))

xyplot(ap, strip = FALSE, cut = list(number = 3, overlap = 0.5))

xyplot(ap, strip = TRUE, cut = list(number = 4, overlap = 0.1))

monthplot(ap)

par(mfrow = c(2,1))
par(mar = c(1,1,1,1)) #zmniejszamy marginesy
boxplot(ap - cycle(ap), names = month.abb)
monthplot(ap)

#wasy to najwieksza i najmniejsza wartosc
#dolny i gorny bok pudelka to 1 i 3 kwartyl
#czarna kropka to mediana
#kropki to wartosci odstajace? 

library(forecast)

dev.off()
seasonplot(ap, col = rainbow(12))

#wykresy rozrzutu dla opoznionych wartosci

lag.plot(ap, lags = 12)
lag.plot(ap, lags = 12, do.lines = F)

ap.reszty <- decompose(ap)$random
plot(ap.reszty)

ap.reszty <- na.omit(ap.reszty)

lag.plot(ap.reszty, lag = 12, do.lines = F)
#usunelismy trend i sezonowosc 

#Korekty kalendarzowe
srednia.liczba.dni <- 365.25/12
liczba.dni.w.miesiacach <- monthdays(ap)
liczba.dni.w.miesiacach

#korekta danych ap
ap.korekta <- ap/liczba.dni.w.miesiacach * srednia.liczba.dni

par(mar = c(1,1,1,1))
ts.plot(ap, ap.korekta, main = 'Dane oryginalne vs skorygowane',
        col = c('blue', 'red'), lty = c(1,2))
legend('bottomright', legend = c("dane oryginalne", 'dane skorygowane'),
       col = c('blue', 'red'), lty = c(1,2))

sezonowosc.dane.oryginalne <- decompose(ap, type = 'additive')$figure
sezonowosc.dane.skorygowane <- decompose(ap.korekta, type = 'additive')$figure

par(las = 2)
barplot(rbind(sezonowosc.dane.oryginalne, sezonowosc.dane.skorygowane),
        beside = T, col = c('blue', 'green'), 
        legend.text = c('dane oryginalne', 'dane skorygowane'),
        names.arg = month.abb )
grid()


#Agregacja danych

ap.suma.kwart <- aggregate(ap, nfrequency = 4, FUN = sum)
ap.suma.kwart

ap.max.kwart <- aggregate(ap, 4, max)
ap.max.kwart

ap.srednie.roczne <- aggregate(ap, 1, mean) #srednie dane roczne
ap.srednie.roczne

par(mfrow = c(2,2))
plot(ap.suma.kwart)
plot(ap.max.kwart)
plot(ap.srednie.roczne)
#czwarty wykres <- wyeliminowano sezonowosc 

#dezagregacja szeregow

library(tempdisagg)

pkb <- scan(file = "pkb.txt")
pkb <- ts(pkb, start = c(1995, 1), end = c(2014, 2), frequency = 4)
plot(pkb)

#Budujemy model
da.mod <- td(pkb ~ 1, to = 'monthly', conversion = 'sum')
# w miejsce 1[szereg stale rowny 1] wsatwimy model ktory bedziemy chcieli porownycwac

#stosujemy model

pkb.da <- predict(da.mod)

print(window(pkb.da, start = 1995, end = 1996), digits = 2)
plot(da.mod)

summary(da.mod)

par(mfrow = c(2,1))
par(mar = c(1,1,1,1))
plot(pkb)
plot(pkb.da)



