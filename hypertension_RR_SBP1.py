#!/usr/bin/env python
# coding: utf-8
"""
@author: Radosław Różyński
"""
#%%


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%


# ustawiamy katalogi pracy
import os

KATALOG_PROJEKTU = os.path.join(os.getcwd(),"rytm_serca")



SKAD_POBIERAC = ['./healthy_decades/', './HTX_LVH/',  './hypertension_RR_SBP/',
                 './hypertension/']
                 
czytamy = SKAD_POBIERAC[3]
print ('\nPrzetwarzamy katalog', czytamy)
pliki = os.listdir (czytamy)
[print( pliki.index(item) , ':', item)  for item in pliki]
data_chorzy = []
data_zdrowi = []
#while (True):
   #try:
       #num_plik = int(input('ktory plik? '))
       #print('przerabiamy ', pliki[num_plik])
   #except  ValueError: 
       #print('zly numer pliku ')
   #else:
       #break

# Tworzę black listę, ponieważ część plików ma inną budowę niż pozostałe,
# Niestety przez nie program wywala błąd, który nie jest zależny ode mnie

black_list = [19, 25, 29, 30, 33, 35, 51, 55, 61]
for p in range(len(pliki)):
    while p in black_list:
        p += 1
    if p >= len(pliki):
        break
    plik = pliki[p]
    print('przerabiamy ', plik)
    pacjent = plik[:len(plik)-4]
    KATALOG_CHORZY = os.path.join(KATALOG_PROJEKTU, "CHORZY")
    KATALOG_ZDROWI = os.path.join(KATALOG_PROJEKTU, "ZDROWI")
    if pacjent[4] == '1':
        KATALOG_PACJENTA = os.path.join(KATALOG_ZDROWI, pacjent)
    else:
        KATALOG_PACJENTA = os.path.join(KATALOG_CHORZY, pacjent)
    KATALOG_DANYCH = os.path.join(KATALOG_PACJENTA,"Dane")
    KATALOG_WYKRESOW = os.path.join(KATALOG_PACJENTA, "Wykresy")
    os.makedirs(KATALOG_WYKRESOW, exist_ok=True)
    os.makedirs(KATALOG_DANYCH, exist_ok=True)

    def load_serie(skad , co, ile_pomin = 0, kolumny = ['RR_interval', 'Num_contraction']):
        csv_path = os.path.join(skad, co )
        print ( skad, co)
        seria = pd.read_csv(csv_path, sep='\t', header = None,
                        skiprows= ile_pomin, names= kolumny)
        if skad == SKAD_POBIERAC[2]:
            seria = pd.read_csv(csv_path, sep='\t',  decimal=',' )
        return seria

    if p >= len(pliki):
        break
    pomin = 5
    kolumny = ['time','R_peak','Resp','SBP','cus']
    seria = load_serie(skad = czytamy, co = plik, ile_pomin = pomin, kolumny = kolumny)
    print(seria.info())
        
    wyniki = open(KATALOG_DANYCH + "/wyniki.txt", mode = 'w')


#%%

    # ogólna informacja kompletnosci danych
    print("\n Ogólna informacja o kompletnosci danych:")
    print('\n-->brak danych w danej kolumnie:\n')

    print(seria.isnull().sum())
    print(seria.describe())


#%%

#kwantowanie
    print(seria['R_peak'].value_counts().sort_index())


#%%

#wizualizacja danych : DataFrame.plot
    seria.hist(bins = 50, figsize = (9,6))
    plt.tight_layout()
    plt.title("Histogramy wartości")
    plt.savefig(os.path.join(KATALOG_WYKRESOW,'histogramy.jpg'), dpi = 300 ) 
    plt.show() # sprawdzić jak się plik zapisal


#%%

# Współzależność serii do czasu
    korelacja = seria.corr()

    print('\nKorelacja między serią a czasem\n') 
    print( korelacja["time"].sort_values(ascending = False))
#wyniki.write(korelacja["time"].sort_values(ascending = False))


#%%


# Wyznaczenie czasowych własności sygnału interwałów RR
# RR - interwały (dotyczy zwolnień i przyspieszeń rytmu serca)

# Szukamy chwilii, w której wystąpił R_peak

    R = seria['time'][seria['R_peak'] == 1]

# Mnożę wartości R razy 1000, aby otrzymać milisekundy. Potem łatwiej będzie 
# otrzymywać wyniki dla prawdopodobieństwa pNN50 itp

    R = R * 1000

# RR jest to interwał, czyli "odległość" między R_peakami,
# stąd liczymy pochodną. Na zerowym miejscu pojawia nam się wartość 
# nan, czyli bierzemy od pierwszego miejsca

    RR = R.diff()[1:]
    print("RR:\n ", RR)


# Obliczamy wartość odchylenia standardowego, uzyjemy do tego wbudowanej funkcji
    SDNN = RR.std()
    print("SDNN: ", SDNN)


# Obliczamy RMSSD, korzystając ze wzoru poznanego na zajęciach
# jest to różnica kolejnych sygnałów do kwadratu stąd biorę znów
# pochodną tylko tym razem z RR

    N = len(RR)
    RMSSD = np.sqrt(1/N * np.square(np.diff(RR)).sum())
    print("RMSDD: ", RMSSD)


# Obliczenie pNN50 -> pNN50 = |x_{i} - x_{i-1}| > 50 ms
# chcemy obliczyć prawdopodobieństwo, że odległość między kolejnymi RR 
# była dłuższa niż 50 ms
# zliczamy ilość przypadków

    RR_diff = np.diff(RR)
    NN50 = len(RR_diff[np.abs(RR_diff) > 50])

# dzielimy przez moc naszego zbioru

    pNN50 = NN50 / N
    print("pNN50: ", pNN50)

    
# Analogicznie dla pNN20

    NN20 = len(RR_diff[np.abs(RR_diff) > 20])
    pNN20 = NN20 / N
    print("pNN20: ", pNN20)


#%%


# Wyznaczenie czasowych własności sygnału ciśnienia krwi
# Średnie ciśnienie skurczowe

# SBP - ciśnienie skurczowe, występuje chwile po R
# prawidłowe to 120 - 140 / 80

# wartości przed R_peakiem wskazują nan lub wartość 0, dlatego 
# 'wyciągamy' wartości autentycznych skurczów, które są > 0

    SBP = seria['SBP'][seria['SBP'] > 0]
    SBP_mean = SBP.mean()
    print("MSBP: ", SBP_mean)


#%%


#Kroczaca średnia - wygladzenie danych
    Resp_roll_mean = seria['Resp'].rolling(800).mean()
    Resp_roll_mean


#%%


    plt.plot(Resp_roll_mean)


#%%


# Podział fali oddechowej na wdechy i wydechy
# Resp - zmienna opisująca ruch klatki piersiowej, krzywa oddechu

#Tworzę pomocniczą kolumnę
#seria['pozycja'] = seria['Resp'].diff()[1:]
    seria['pozycja'] = Resp_roll_mean.diff()[1:]

#Biorąc wdech klatka się podnosi, czyli pochodna 'Resp' > 0

    Wdech = seria['pozycja'][seria['pozycja'] >= 0]

#Biorąc wydech klatka się obniża, czyli wysokość maleje -> pochodna 'Resp' < 0

    Wydech = seria['pozycja'][seria['pozycja'] < 0]


#%%


    start = 10000
    koniec = 50000

    fig, ax = plt.subplots(figsize = (8, 6))
    plt.plot(seria['time'][start : koniec], Resp_roll_mean.where(seria['pozycja'] >= 0)[start : koniec],
        color = "darkblue", label = "Wdech")
    plt.plot(seria['time'][start : koniec], Resp_roll_mean.where(seria['pozycja'] < 0)[start : koniec],
        color = "red", label = "Wydech")

    plt.title("Podział na wdechy i wydechy " + plik)
    ax.legend()
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Podział na wdechy i wydechy'+plik+'.jpg'), dpi = 300 )
    plt.show()


#%%

# Do poprawy
# zamienić zliczenia przyśpieszeń i zwolnień, rytmu serca  oraz  
# spadki i wzrosty ciśnienia na ich procentowy udział  w całości


#%%

# Poszukiwanie wzorców (własności symbolicznych sygnału)

# Ilość wystąpień pików R przy wdechu i przy wydechu
    ilosc = len(seria['pozycja'])
    Wdechy_R = seria['pozycja'][(seria['pozycja'] > 0) & (seria['R_peak'] == 1)].count()
    Wdechy_R = Wdechy_R / ilosc

    
    Wydechy_R = seria['pozycja'][(seria['pozycja'] < 0) & (seria['R_peak'] == 1)].count()
    Wydechy_R = Wydechy_R / ilosc

    
# Ilość przyśpieszeń a i zwolnień d rytmu serca przy wdechu i wydechu
# Jeśli RR_diff() < 0 <- przyspieszenie a, [RR maleje]
# Jeśli RR_diff() > 0 <- zwolnienie d,

    czy_wdech_w_czasie_RR = []
    for i in RR.index:
        if seria['pozycja'][i] > 0:
            czy_wdech_w_czasie_RR.append(True)
        else:
            czy_wdech_w_czasie_RR.append(False)
  
#przy wdechu

    count_wdech_a = 0
    count_wdech_d = 0
    ilos = len(RR_diff)
    
    for i in range(len(RR_diff)):
        if((RR_diff[i] < 0) & (czy_wdech_w_czasie_RR[i] == True)):
            count_wdech_a += 1
        if((RR_diff[i] > 0) & (czy_wdech_w_czasie_RR[i] == True)):
            count_wdech_d +=1
    
    count_wdech_a = count_wdech_a / ilosc
    count_wdech_d = count_wdech_d / ilosc
    print(("\nIlość przyśpieszeń a i zwolnień d rytmu serca przy wdechu\n"))
    print("przyspieszenia: \n")
    print(count_wdech_a, "\n") 
    print("zwolnienia: \n")
    print(count_wdech_d, "\n")


# przy wydechu

    count_wydech_a = 0
    count_wydech_d = 0

    for i in range(len(RR_diff)):
        if((RR_diff[i] < 0) & (czy_wdech_w_czasie_RR[i] == False)):
            count_wydech_a += 1
        if((RR_diff[i] > 0) & (czy_wdech_w_czasie_RR[i] == False)):
            count_wydech_d +=1

    count_wydech_a = count_wydech_a / ilosc   
    count_wydech_d = count_wydech_d / ilosc        
    print(("\nIlość przyśpieszeń a i zwolnień d rytmu serca przy wydechu\n"))
    print("przyspieszenia: \n")
    print(count_wydech_a, "\n") 
    print("zwolnienia: \n")
    print(count_wydech_d, "\n")

    
# Ilość wzrostów ^ i spadków v ciśnienia skurczowego SBP przy wdechu i wydechu
    
    czy_wdech_w_czasie_skurczu = []
    SBP_diff = np.diff(SBP)
    for i in SBP.index:
        if seria['pozycja'][i] > 0:
            czy_wdech_w_czasie_skurczu.append(True)
        else:
            czy_wdech_w_czasie_skurczu.append(False)
        
        
# przy wdechu

    count_wdech_wzrost = 0
    count_wdech_spadek = 0
    ilosc = len(SBP_diff)

    for i in range(len(SBP_diff)):
        if((SBP_diff[i] > 0) & (czy_wdech_w_czasie_skurczu[i] == True)):
            count_wdech_wzrost += 1
        if((SBP_diff[i] < 0) & (czy_wdech_w_czasie_skurczu[i] == True)):
            count_wdech_spadek += 1
    
    count_wdech_wzrost = count_wdech_wzrost / ilosc
    count_wdech_spadek = count_wdech_spadek / ilosc
    print("\nIlość wzrostów ^ i spadków v ciśnienia skurczowego SBP przy wdechu\n") 
    print("wzrosty: \n")
    print(count_wdech_wzrost, "\n")
    print("spadki: \n")
    print(count_wdech_spadek, "\n") 


# przy wydechu

    count_wydech_wzrost = 0
    count_wydech_spadek = 0
    for i in range(len(SBP_diff)):
        if((SBP_diff[i] > 0) & (czy_wdech_w_czasie_skurczu[i] == False)):
            count_wydech_wzrost += 1
        if((SBP_diff[i] < 0) & (czy_wdech_w_czasie_skurczu[i] == False)):
            count_wydech_spadek += 1
        
    count_wydech_wzrost = count_wydech_wzrost / ilosc
    count_wydech_spadek = count_wydech_spadek / ilosc
    print("\nIlość wzrostów ^ i spadków v ciśnienia skurczowego SBP przy wydechu\n")
    print("wzrosty: \n")
    print(count_wydech_wzrost, "\n") 
    print("spadki: \n")
    print(count_wydech_spadek, "\n") 
        


#%%


#wykresy Poincare
#wykresy Poincare dla RR

    RR1 = RR[:len(RR)-1]
    RR2 = RR[1:]
    plt.plot(RR1,RR2,'bx',markersize = 4)
    plt.title("Rozrzut danych: " + plik)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Rozrzut_danych'+plik+'.jpg'), dpi = 300 )
    plt.show()


#%%

#WIZUALIZACJA

    start = 10000
    koniec = 30000

    seria['R_peak'] = seria['R_peak'] * seria['Resp']
    seria['R_peak'].where(seria['R_peak'] > 0, np.nan, inplace = True)
    
    plt.plot(seria['time'][start : koniec], seria['Resp'][start : koniec],'g.',
            markersize = 4 )
    plt.plot(seria['time'][start : koniec], seria['R_peak'][start : koniec],'rx',
            markersize = 10)
    plt.title("Oddech " + plik)
    plt.savefig(os.path.join(KATALOG_WYKRESOW, 'Oddech ' + plik + '.jpg'), dpi = 300 ) 
    plt.show()

#%%

    if pacjent[4] == '1':
        czy_chory = 0
    else:
        czy_chory = 1
    columns = ['pacjent', 'czy_chory', 'SDNN', 'RMSSD', 'pNN50', 'pNN20', 'SBP_mean', 'R_peaks_inhale', 'R_peaks_exhale', 'R_a_inhale', 'R_a_exhale', 'R_d_inhale', 'R_d_exhale', 'SBP_up_inhale', 'SBP_up_exhale', 'SBP_down_inhale', 'SBP_down_exhale']
    wiersz = [plik, czy_chory, SDNN, RMSSD, pNN50, pNN20, SBP_mean, Wdechy_R, Wydechy_R, count_wdech_a, count_wydech_a, count_wdech_d, count_wydech_d, count_wdech_wzrost, count_wydech_wzrost,count_wdech_spadek, count_wydech_spadek]
    wyniki.write(str(wiersz))
    wyniki.close()
    if pacjent[4] == '1':
        data_zdrowi.append(wiersz)
    else:
        data_chorzy.append(wiersz)
df_zdrowi = pd.DataFrame(data_zdrowi, columns = columns)
df_chorzy = pd.DataFrame(data_chorzy, columns = columns)
df_zdrowi.to_csv('zdrowi.csv', encoding='utf-8', index = False)
df_chorzy.to_csv('chorzy.csv', encoding='utf-8', index = False)




