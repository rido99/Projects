#dekompozycja

library(expsmooth)

data(dji)
head(dji)
dji.Close <- dji[, "Close"]

plot(dji.Close)
#stacjonarne nie jest bo ma trend rosnący, brak seznowości

#ruchoma średnia rzedów 1,5, i 10 
m3 <- filter(dji.Close, sides = 2, filter = rep (1/3,3))
#Ruchoma średnia rzędu q = ruchoma średnia obustronna z krokiem 2q + 1
#m(t) =  1/2q+1 suma od j = -q do q z X{t - j}
m11 <- filter(dji.Close, sides = 2, filter = rep (1/11,11))
m21 <- filter(dji.Close, sides = 2, filter = rep (1/21,21))

head(m3)
head(m11) #brak poczatkowych 5 obserwacji
head(m21) #brak poczatkowych 10 obserwacji

tail(m3)
tail(m11) 
tail(m21)

sum(is.na(m3)) #powstaje 2q braków
sum(is.na(m11))
sum(is.na(m21))

plot(m3, col = "blue", lty = 2)
lines(m11, col = "red", lty = 2) #dobre przyblizenie trendu, wpasowuje sie w środek
lines(m21, col = "green", lty = 2) #za bardzo wygladzono dane
# wybieramy czerwony, bo zielony nie zauwaza spadkow cen(końcówka)
lines(dji.Close)
#wybor odpo rzedu gra duzą role w wygładzeniu 

#Filtr Spencera 
#wagi średniej, jest ich osiem, w_j = 0 dla |j| > 7, w_j = w_-j dla |j|<= 7
#ruchoma srednia rzedu 7
m.spencer <- filter(dji.Close, 1/321 * c(-3, -6, -5, 3, 21, 46, 67,74, 67, 46, 21, 3, -5, -6, -3), sides = 2)

plot(m3, col = "blue", lty = 2)
lines(m11, col = "red", lty = 2) 
lines(m21, col = "green", lty = 2)
lines(m.spencer, col = "magenta", lty = 2) #najbardziej zbliżony do niebieskiego
lines(dji.Close, lty = 1)

ap <- scan(file = "AirPass.txt")
ap<-ts(ap,start=c(2002,1),end=c(2014,2),frequency = 12)
ap.d.a <- decompose(ap, type = "additive")
plot(ap.d.a)

ap.trend <- ap.d.a$trend
ap.sezon <- ap.d.a$seazonal
ap.ind.sezon <- ap.d.a$figure
ap.reszty <- ap.d.a$random

barplot(ap.ind.sezon) #jest sezonowosc, wzrosty w letnich, spadki w zimowych
barplot(ap.ind.sezon, names.arg = month.abb) 

pkb <- scan(file = "pkb.txt")
pkb <- ts(pkb, start = c(1995, 1), end = c(2014, 2), frequency = 4)

pkb.d.a <- decompose(pkb, type = "additive")
plot(pkb.d.a)

pkb.reszty <- pkb.d.a$random
tsdisplay(pkb.reszty) #odrzucamy model dekompozycji bo nie przypomina białego szumu
#duzo wartosci poza przedzialem ufnosci

pkb.d.m <- decompose(pkb, type = "multiplicative")
plot(pkb.d.m)
#nie ma takich duzych wahan
#algorytm dekompozycji nieparametrycznej
#1. mt - skladnik trendu, symetryczna ruchoma srednia rzedu q = okres czestotliwosci danych [frequency]
#gdy f parzyste
#wspolczynniki filter = c(0.5, rep(1, frequency - 1), 0.5)/frequency
#gdy f nieparzyste
#rep(1, freq)/freq

#2. addytywna: Yt = Xt - mt
#multiplikatywna: Yt = Xt/mt

#3. wynzaczamy indeksy sezonowe
#s1,s2,...,sfreq
#si - uśrednione wartosci Yt dla kazdej jednostki czasu i wszytskich lat

#4. standaryzacja indeksów sezonowych, aby nie wplywaly na trend całościowy
#add: odejmujemy srednia [s1 - mu,...]
#mult: dzielimy przez srednia [s1/mu, ...]
#mu - srednia s1,...,sf

#5. Wyznaczamy reszty 
#add: Ut = Xt - mt - st
#mult: Ut = Xt/(mt - st)

#mozna zrobic tylko jak mamy okresowe i sezonowe dane

#DEKOMPOZYCJA NA PODSTAWIE MODELU REGRESJI
#1. trend liniowy
#Xt = a + bt + Ut
#Xt = a + bt + s(t) + Ut

head(seasonaldummy(pkb))
#kody okresow 1,0,0 pierwszy kwartał, 0,0,0 czwarty kwartał

pkb.trend <- tslm(pkb ~ trend) #pomijamy sezonowosc w tym kroku, jakby jej nie bylo
summary(pkb.trend)

plot(pkb)
lines(fitted(pkb.trend), col = "red", lty = 2)
#nie uwzglednia zmian sezonowych

tsdisplay(residuals(pkb.trend))
# wykres reszt nie wyglada jak bialy szum

pkb.trend.sezon <- tslm(pkb ~ trend + season)
summary(pkb.trend.sezon)
#3 skladniki sezonowe - wszystkie istotne
# gdzie wieksze r^2? mamy lepszy model bo wieksze 

plot(pkb)
lines(fitted(pkb.trend.sezon), col = 'red', lty = 2)
#uwzglednia zmiany sezzonowe wiec better

tsdisplay(residuals(pkb.trend.sezon))
#nie wyszlo [bialy szum] bo zmienna wariancja 

#transformacja logarytmiczna
pkb.log <- tslm(pkb ~ trend + season, lambda = 0)
summary(pkb.log)
# tu nie porownujemy R^2 bo mamy dane po transformacji, zlogarytmiznowane

plot(pkb)
lines(fitted(pkb.log), col = "red", lty = 2)
tsdisplay(residuals(pkb.log))
# model liniowy nie jest dobrym przypuszczeniem dla tych danych







