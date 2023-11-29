# -*- coding: utf-8 -*-
"""
@author: Radosław Różyński
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df_chorzy = pd.read_csv('chorzy.csv')
df_zdrowi = pd.read_csv('zdrowi.csv')

df_wszyscy = pd.concat([df_chorzy, df_zdrowi], ignore_index = True)

#%% Standardowe oznaczenia przetwarzanych danych i ogólny ogląd

X , y  = df_wszyscy.drop(['pacjent','czy_chory'],  axis=1), df_wszyscy['czy_chory']
print('dane do klasyfikacji:\n',X,'\nrozmiar danych:', X.shape )
print('znane etykiety:\n', y ,'\nrozmiar etykiety', y.shape)
y_chorzy = (y == 1)
print('binarne etykiety:\n', y_chorzy ,'\nrozmiar etykiety', y_chorzy.shape)

#%% skalowanie
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)


#%% TRAIN and TEST sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X train', X_train.shape, 'X test', X_test.shape)
print('y train', y_train.shape, 'y test', y_test.shape)

y_train_chorzy = (y_train == 1)
y_test_chorzy = (y_test == 1)


#%% PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D_train = pca.fit_transform(X_train)
print('\nProcent wariancji objasniony przez zmienne w PCA')
print(pca.explained_variance_ratio_)
X2D_test = pca.transform(X_test)

#zastosowanie pca do ograniczenia wymiarowoci do zmiennych objasniajacych 95% wariancji
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

#%% plot dla pca ograniczonego do 2 zmiennych
import seaborn as sns
sns.set()
Pca1 = X2D_test[:,0]
Pca2 = X2D_test[:,1]
ypca = y_test
pcaall = {'PC1': Pca1,
    'PC2': Pca2,
    'target': ypca}
pca_df  = pd.DataFrame(pcaall)
target_names = { 0:'zdrowy',
                 1:'chory'}
pca_df['target'] = pca_df['target'].map(target_names)
sns.lmplot(
    x='PC1', 
    y='PC2', 
    data=pca_df, 
    hue='target', 
    legend=True
    )
plt.title('2D PCA Graph')
plt.show()

#%% WYSZUKIWANIE NAKLEPSZYCH WARTOŚCI HIPERPARAMETROW popularna metoda
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier

moje_cv = 3
params = {
    'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'alpha' : [ 0.001, 0.01, 0.1],
    'penalty' :['l2','l1']#,'elasticnet','none']
        }


sgd_clf= SGDClassifier(max_iter =100)
grid = GridSearchCV ( sgd_clf, param_grid =params, cv=moje_cv, scoring ='f1',
                     return_train_score=True)

grid.fit(X_train, y_train_chorzy)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)
    
#%% Ostateczne przetwarzanie
from sklearn.metrics import precision_score, recall_score, f1_score

final_model = grid.best_estimator_


final_model_param = final_model.get_params()
final_training = final_model.fit(X_train,y_train_chorzy)

final_predictions = final_training.predict(X_test)

final_f1 = f1_score(y_test_chorzy,final_predictions)
final_recall = recall_score(y_test_chorzy,final_predictions)
final_precision = precision_score(y_test_chorzy,final_predictions)

print('f1 najlepszego rozwiazania SGDClassifier ',"%.2f"%final_f1)
print('recall najlepszego rozwiazania SGDClassifier ',"%.2f"%final_recall)
print('presision najlepszego rozwiazania SGDClassifier ',"%.2f"%final_precision)
print('parametry najlepszego modelu', final_model.get_params())


#%%  ZACZYNAMY WALIDACJE

from sklearn.model_selection import cross_val_score

moje_cv = 3
napis= 'dokladnosc: ilosc dobrych predykcji'
dokladnosc_klasyfikacji = cross_val_score(final_model, X_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="accuracy")
print(napis, dokladnosc_klasyfikacji)

napis = 'prezycja : TP/(TP+NP) ilosc dobrych predykcji w klasie pozytywnej'
precyzja_klasyfikacji = cross_val_score(final_model, X_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="precision")
print(napis, precyzja_klasyfikacji)


napis= 'recall : TP(TP+FN)  ile instancji klasy pozytywnej zostalo dobrze rozpoznane'
czulosc_klasyfikacji = cross_val_score(final_model, X_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="recall")

print(napis, czulosc_klasyfikacji)

#%% WSTEPNE PODSUMOWANIE

print("srednia i std dokladnosci %.3f"% dokladnosc_klasyfikacji.mean(), 
      " +/- %.3f"% dokladnosc_klasyfikacji.std())

print("srednia i std    precyzji  %.3f"% precyzja_klasyfikacji.mean() ,
      " +/- %.3f"%precyzja_klasyfikacji.std()  )


print("srednia i std    czulosci %.3f"%czulosc_klasyfikacji.mean(),
      " +/- %.3f"%czulosc_klasyfikacji.std())


#%% Konstrukcja tablica pomylek ( macierz błędow   )

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(final_model, X_train, y_train_chorzy, cv= moje_cv)

#  predykcja jest uzyskana z walidacji krzyżowej, czyli na zbiorach walidacyjnych

#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

confusion =  confusion_matrix(y_train_chorzy, y_train_predict)

(tn,fp,fn,tp) =  confusion_matrix(y_train_chorzy, y_train_predict).ravel()

disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn,fp],[fn,tp]]),
                               display_labels=[0,1])

disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.grid(False)
plt.show()

print('confusion matrix: \n', confusion, '\n')
print( 'TN =', tn, '\tFP =', fp)
print( 'FN =', fn, '\tTP =', tp)
print('\nprecision =  %.3f'%(tp/(tp + fp)) )
print('recall = %.3f'%(tp/(tp + fn)) )

from sklearn.metrics import precision_score, recall_score

print('\nchorzy wsrod sklasyfikowanych jako chorzy to %.2f'% precision_score(y_train_chorzy,y_train_predict))
print('\nchorzy rozpoznano w %.2f przypadkach  '% recall_score(y_train_chorzy,y_train_predict))

#%%  KOMPROMIS
#użycie metody decision_function() glownego obiektu
y_scores = cross_val_predict(final_model, X_train, y_train_chorzy, cv=5,
                             method="decision_function")
                        # domyslnie jest method='predict'
                        
#%% https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision-recall

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_chorzy, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-10, 10])
plt.savefig("precision_recall_vs_threshold_plot",dpi=300)
plt.show()

#%%

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-0.1, 1])
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(recalls, precisions )
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision')
plt.show()

#%% model dla PCA gdzie zredukowalismy wymiar do ilosci zmiennych pokrywajacych 95% wariancji 
print('PCA\n\n')
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


X_reduced_train, X_reduced_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

print('X_reduced train', X_reduced_train.shape, 'X_reduced test', X_reduced_test.shape)
print('y train', y_train.shape, 'y test', y_test.shape)


moje_cv = 3
params = {
    'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'] ,
    'alpha' : [ 0.001, 0.01, 0.1],
    'penalty' :['l2','l1']
        }

sgd_clf= SGDClassifier(max_iter =100)
grid = GridSearchCV (sgd_clf, param_grid =params, cv=moje_cv, scoring ='f1',
                     return_train_score=True)

grid.fit(X_reduced_train, y_train_chorzy)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)
    
final_model = grid.best_estimator_


final_model_param = final_model.get_params()
final_training = final_model.fit(X_reduced_train,y_train_chorzy)

final_predictions = final_training.predict(X_reduced_test)

final_f1 = f1_score(y_test_chorzy,final_predictions)
final_recall = recall_score(y_test_chorzy,final_predictions)
final_precision = precision_score(y_test_chorzy,final_predictions)

print('f1 najlepszego rozwiazania SGDClassifier ',"%.2f"%final_f1)
print('recall najlepszego rozwiazania SGDClassifier ',"%.2f"%final_recall)
print('presision najlepszego rozwiazania SGDClassifier ',"%.2f"%final_precision)

moje_cv = 3
napis= 'dokladnosc: ilosc dobrych predykcji'
dokladnosc_klasyfikacji = cross_val_score(final_model, X_reduced_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="accuracy")
print(napis, dokladnosc_klasyfikacji)

napis= 'prezycja : TP/(TP+NP) ilosc dobrych predykcji w klasie pozytwnej'
precyzja_klasyfikacji = cross_val_score(final_model, X_reduced_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="precision")
print(napis, precyzja_klasyfikacji)


napis= 'recall : TP(TP+FN)  ile instancji klasy pozytwnej zostalo dobrze rozpoznane'
czulosc_klasyfikacji = cross_val_score(final_model, X_reduced_train,  y_train_chorzy, 
                                              cv=moje_cv, scoring="recall")

print(napis, czulosc_klasyfikacji)

#%% WSTEPNE PODSUMOWANIE

print("srednia i std dokladnosci %.3f"% dokladnosc_klasyfikacji.mean(), 
      " +/- %.3f"% dokladnosc_klasyfikacji.std())

print("srednia i std    precyzji  %.3f"% precyzja_klasyfikacji.mean() ,
      " +/- %.3f"%precyzja_klasyfikacji.std()  )


print("srednia i std    czulosci %.3f"%czulosc_klasyfikacji.mean(),
      " +/- %.3f"%czulosc_klasyfikacji.std())

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(final_model, X_reduced_train, y_train_chorzy, cv= moje_cv)

confusion =  confusion_matrix(y_train_chorzy, y_train_predict)

(tn,fp,fn,tp) =  confusion_matrix(y_train_chorzy, y_train_predict).ravel()

disp = ConfusionMatrixDisplay(confusion_matrix=np.array([[tn,fp],[fn,tp]]),
                               display_labels=[0,1])
disp = disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
plt.grid(False)
plt.show()

print('confusion matrix: \n', confusion, '\n')
print( 'TN =', tn, '\tFP =', fp)
print( 'FN =', fn, '\tTP =', tp)
print('\nprecision =  %.3f'%(tp/(tp + fp)) )
print('recall = %.3f'%(tp/(tp + fn)) )

from sklearn.metrics import precision_score, recall_score
print('\nchorzy wsrod sklasyfikowanych jako chorzy to %.2f'% precision_score(y_train_chorzy,y_train_predict))
print('\nchorzy rozpoznano w %.2f przypadkach  '% recall_score(y_train_chorzy,y_train_predict))

#%%  KOMPROMIS
#użycie metody decision_function() glownego obiektu
y_scores = cross_val_predict(final_model, X_reduced_train, y_train_chorzy, cv=5,
                             method="decision_function")
                        # domyslnie jest method='predict'
                        
#%% https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision-recall

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_chorzy, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-5, 7])
plt.savefig("precision_recall_vs_threshold_plot",dpi=300)
plt.show()

#%%

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-2, 2])
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(recalls, precisions )
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall vs Precision')
plt.show()
