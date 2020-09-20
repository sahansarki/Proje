#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:41:47 2020

@author: sahan
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statistics import mode
from sklearn import metrics
from sklearn import tree
import pickle
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def prune_index(inner_tree, index, threshold): 
    """
    Ağaçtaki nodelardaki value değerleri sol ve sağ daki örnek değer sayılarını verir.
    Bunlardan bir tanesi eğer verilen threshold-eşik'den küçükse o ağacı yaprağa dönüştür.
    Bizim örneğimizde threshold 'umuz 50.
    """
    if inner_tree.value[index].min() < threshold:
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)




    
def ayirma(var,var_odeyenler,var_odememyenler):
    """
    Verilen bağımsız değişkeni o satırdaki Loan Status değerine göre
    kredi ödeyip - ödemeyenler şeklinde ayırıp bunları döndürdük.
    var -> bağımsız değişkenimiz(dataframe.values şeklinde aldığımız için liste şeklinde.)
    """
    j = 0
    for i in var:
        if(loan_status[j] == "Fully Paid"):
            var_odeyenler.append(i)
        else :
            var_odememyenler.append(i)
        j = j + 1
    return var_odeyenler,var_odememyenler

def eksikveri_numeric(var):
    """
    var , yukardaki fonksiyonumuz gibi bağımsız değişkenimiz.
    
    bağımsız değişkenimizin sütununda i'yi gezdirip nan değil ise list2 listemize atıyoruz.
    Sonra list2 ye attığımız değerleri toplayıp uzunluğunu bölerek ortalama değeri buluyoruz.
    Daha sonra bu bulduğumuz ortalama değeri nan olan yerlere koyup listemizi döndürüyoruz.
    """
    list2 = []
    j = 0
    for i in var:
        if(np.isnan(i)):
            continue
        else:
            list2.append(i)
    ortalama = sum(list2) / len(list2)
    
    for i in var:
        if(np.isnan(i)):
            var[j] = ortalama
        j = j + 1
    return var

def ks_test(var_odeyenler,var_odemeyenler,var_ismi):
    """
    Numeric değerlerimiz için karşılaştırma yaptık.
    ks_2samp() kullanarak bir bağımsız değişken için krediyi ödeyenler ile ödemeyenler arasında 
    karşılaştırma yaptık ve bize 2 değer döndü.
    p_value değerine bakarak arasındaki ilişkiyi yorumladık.
    """
    alpha=0.05 #Sabit değer.Değiştirilebilir.
    
    print(var_ismi + ' için Kolmogorov test sonuçları;')
    KS_statistic , p_value = st.ks_2samp(var_odeyenler,var_odemeyenler)
    print(st.ks_2samp(var_odeyenler,var_odemeyenler))
    if p_value<=alpha:
        print("H0 hipotezi reddedildi, bu iki değişken arasında ilişki var. \n")
    else:
        print("H0 hipotezi kabul edildi, bu iki değişken arasında bir ilişki yok. \n")
    
    print("\n")
    
    
def chi_square(var1,var2):
    """
    pd.crosstab diyerek bağımsız değişkenimizin barındırdığı özelliklere kaç tanesi
    krediyi ödeyip - ödemediğini gösteren bir tablo oluşturup yazdırdık.
    
    Oluşan bu tabloyu chi2_contingency() fonksiyonuna atadık ve bize 4 değer döndürdü.
    p_value değerini bakarak ilişkilerini yorumladık.
    """
    contingency_table = pd.crosstab(var1,var2)
    print('contingency_table :-\n',contingency_table)



    chi_square_statistic,p_value,dof,expected=st.chi2_contingency(contingency_table)
    
    alpha=0.05
    
    print('Significance level: ',alpha)
    print('Degree of Freedom: ',dof)
    print('chi-square statistic:',chi_square_statistic)
    print('p-value:',p_value)
    
    
    if p_value<=alpha:
        print("H0 hipotezi reddedildi, bu iki değişken arasında ilişki var. \n")
    else:
        print("H0 hipotezi kabul edildi, bu iki değişken arasında bir ilişki yok. \n")


def box_compare(kredi_odeyenler,kredi_odemeyenler,var_name):
    """
    krediyi ödeyenler ile ödemeyenleri bağımsız değişkenimize bakarak box şeklinde çizdirdik.
    """
    data = [kredi_odeyenler,kredi_odemeyenler]
    
    fig = plt.figure(figsize =(10,7)) 
    plt.boxplot(data,sym = 'b+')
    plt.title(var_name + " Box Tablosu")
# CSV'de tanımlı sütunlar

veriler = pd.read_csv('sample_data.csv', sep=";")

#değişkenleri alıyoruz.
loan_status = veriler["Loan Status"].values
credit_score = veriler["Credit Score"].values
current_loan_amount = veriler["Current Loan Amount"].values
monthly_debt = veriler["Monthly Debt"].values
credit_history = veriler["Years of Credit History"].values
annual_income = veriler["Annual Income"].values
number_of_open_accounts = veriler["Number of Open Accounts"].values
current_credit_balance = veriler["Current Credit Balance"].values
current_job = veriler["Years in current job"].values

"""
Kredi skorunda 4 haneli olan numaraları 3 haneye düşürdük.
"""


j = 0
for i in credit_score:
    if(i > 1000):
        credit_score[j] = i / 10
    j = j+1
    
    
    

    
#Eksik veriler

credit_score = eksikveri_numeric(credit_score)
annual_income = eksikveri_numeric(annual_income)



current_job_int = []
#current_job listesinin içeriğini değerlere göre numeric olarak oluşturduk
j = 0
for i in current_job:
    if(i == "< 1 year"):
        current_job_int.append(0)
    elif( i == "2 years"):
        current_job_int.append(1)
    elif( i == "3 years"):
        current_job_int.append(2)
    elif( i == "4 years"):
        current_job_int.append(3)
    elif( i == "5 years"):
        current_job_int.append(4)
    elif( i == "6 years"):
        current_job_int.append(5)
    elif( i == "7 years"):
        current_job_int.append(6)
    elif( i == "8 years"):
        current_job_int.append(7)
    elif( i == "9 years"):
        current_job_int.append(8)
    elif( i == "10+ years"):
        current_job_int.append(9)
    else:
        current_job_int.append(-1) #Eksik verilere -1 dedik.

en_cok_tekrar = mode(current_job)
#en çok tekrar eden sayımızı bulduk.
j = 0
for i in current_job_int:
    if(i == -1):
         current_job_int[j] = en_cok_tekrar
    j = j + 1
 
#-1 dediğimiz eksik verilere bu en çok tekrar eden değeri koyduk
#Sonra tekrardan listemizi eski halina döndürdük.
    
j = 0
for i in current_job_int:
    if(i == 0):
        current_job_int[j] = "< 1 year"
    elif(i == 1):
        current_job_int[j] = "2 years"
    elif(i == 2):
        current_job_int[j] = "3 years"
    elif(i == 3):
        current_job_int[j] = "4 years"
    elif(i == 4):
        current_job_int[j] = "5 years"
    elif(i == 5):
        current_job_int[j] = "6 years"
    elif(i == 6):
        current_job_int[j] = "7 years"
    elif(i == 7):
        current_job_int[j] = "8 years"
    elif(i == 8):
        current_job_int[j] = "9 years"
    elif(i == 9):
        current_job_int[j] = "10+ years"
    j = j + 1



kredi_odeyenler_score,kredi_odemeyenler_score = [] , []
kredi_odeyenler_amount,kredi_odemeyenler_amount = [] , []
kredi_odeyenler_debt,kredi_odemeyenler_debt = [] , []
kredi_odeyenler_history,kredi_odemeyenler_history = [] , []
kredi_odeyenler_annual,kredi_odemeyenler_annual = [] , []
kredi_odeyenler_account,kredi_odemeyenler_account = [] , []
kredi_odeyenler_balance,kredi_odemeyenler_balance = [] , []
kredi_odeyenler_current_job,kredi_odemeyenler_current_job = [] , []

#Verileri odeyen - odemeyen olarak böldük çünkü bağımlı değişkenimize göre karşılaştırma yapacağız.
kredi_odeyenler_score,kredi_odemeyenler_score = ayirma(credit_score,kredi_odeyenler_score,kredi_odemeyenler_score)
kredi_odeyenler_amount, kredi_odemeyenler_amount = ayirma(current_loan_amount,kredi_odeyenler_amount,kredi_odemeyenler_amount)
kredi_odeyenler_debt, kredi_odemeyenler_debt = ayirma(monthly_debt,kredi_odeyenler_debt,kredi_odemeyenler_debt)
kredi_odeyenler_history, kredi_odemeyenler_history = ayirma(credit_history,kredi_odeyenler_history,kredi_odemeyenler_history)
kredi_odeyenler_annual, kredi_odemeyenler_annual = ayirma(annual_income,kredi_odeyenler_annual,kredi_odemeyenler_annual)
kredi_odeyenler_account, kredi_odemeyenler_account = ayirma(number_of_open_accounts,kredi_odeyenler_account,kredi_odemeyenler_account)
kredi_odeyenler_balance, kredi_odemeyenler_balance = ayirma(current_credit_balance,kredi_odeyenler_balance,kredi_odemeyenler_balance)




ortalama_odeyenler = sum(kredi_odeyenler_score) / len(kredi_odeyenler_score)
ortalama_odemeyenler = sum(kredi_odemeyenler_score) / len(kredi_odemeyenler_score)

print(ortalama_odeyenler,ortalama_odemeyenler)


#Verileri box plot yardımı ile karşılaştırma.

#box_compare(kredi_odeyenler_score,kredi_odemeyenler_score,"Score")
box_compare(kredi_odeyenler_amount,kredi_odemeyenler_amount,"Amount")
#box_compare(kredi_odeyenler_debt,kredi_odemeyenler_debt,"Debt")
#box_compare(kredi_odeyenler_history,kredi_odemeyenler_history,"History")
#box_compare(kredi_odeyenler_annual,kredi_odemeyenler_annual,"Annual")
#box_compare(kredi_odeyenler_account,kredi_odemeyenler_account,"Account")
#box_compare(kredi_odeyenler_balance,kredi_odemeyenler_balance,"Balance")



veriler['Years in current job'] = current_job_int
#Eksik veri kalmadı.


sonuc = veriler.iloc[:,2:] #Eksiksiz bağımsız değişkenlerimizi aldık.

#Verileri ayırdık. %70 train %15 validasyon %15 test için.

x_train, x_test, y_train, y_test = train_test_split(sonuc , loan_status , test_size = 0.15 , random_state = 0)
x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train, test_size = 0.18 , random_state = 0) 



loan_status2 = y_train #Bağımlı Değişkenimiz.

#Numeric Veriler
#Yalnızca train'i özellik seçimi kullanacağımız için değişkenleri train dataframe'inden aldık.
credit_score2 = x_train["Credit Score"].values
current_loan_amount2 = x_train["Current Loan Amount"].values
monthly_debt2 = x_train["Monthly Debt"].values
credit_history2 = x_train["Years of Credit History"].values
annual_income2 = x_train["Annual Income"].values
number_of_open_accounts2 = x_train["Number of Open Accounts"].values
current_credit_balance2 = x_train["Current Credit Balance"].values


kredi_odeyenler_score,kredi_odemeyenler_score = ayirma(credit_score2,kredi_odeyenler_score,kredi_odemeyenler_score)
kredi_odeyenler_amount, kredi_odemeyenler_amount = ayirma(current_loan_amount2,kredi_odeyenler_amount,kredi_odemeyenler_amount)
kredi_odeyenler_debt, kredi_odemeyenler_debt = ayirma(monthly_debt2,kredi_odeyenler_debt,kredi_odemeyenler_debt)
kredi_odeyenler_history, kredi_odemeyenler_history = ayirma(credit_history2,kredi_odeyenler_history,kredi_odemeyenler_history)
kredi_odeyenler_annual, kredi_odemeyenler_annual = ayirma(annual_income2,kredi_odeyenler_annual,kredi_odemeyenler_annual)
kredi_odeyenler_account, kredi_odemeyenler_account = ayirma(number_of_open_accounts2,kredi_odeyenler_account,kredi_odemeyenler_account)
kredi_odeyenler_balance, kredi_odemeyenler_balance = ayirma(current_credit_balance2,kredi_odeyenler_balance,kredi_odemeyenler_balance)

    
print("*************Chi Square*****************\n")

chi_square(x_train["Home Ownership"],y_train)
chi_square(x_train["Term"],y_train)
chi_square(x_train["Years in current job"],y_train)
chi_square(x_train["Purpose"],y_train)


print("*************Chi Square*****************\n")


print("*************Kolmogorov Smirnov*****************\n")
ks_test(kredi_odeyenler_score,kredi_odemeyenler_score,"Credit Score")

ks_test(kredi_odeyenler_amount,kredi_odemeyenler_amount,"Current Loan Amount")
ks_test(kredi_odeyenler_debt,kredi_odemeyenler_debt,"Monthly Debt")
ks_test(kredi_odeyenler_history,kredi_odemeyenler_history,"Years of Credit History")
ks_test(kredi_odeyenler_annual,kredi_odemeyenler_annual,"Annual Income")
ks_test(kredi_odeyenler_account,kredi_odemeyenler_account,"Number of Open Accounts")
ks_test(kredi_odeyenler_balance,kredi_odemeyenler_balance,"Current Credit Balance ")

print("\n*************Kolmogorov Smirnov*****************")


x_train['Decision'] = y_train #x_train'e Decision kolonu olarak loan statusu koyduk.

category_col = ['Term','Years in current job','Home Ownership','Purpose','Decision']


for col in category_col: 
    #Her bir kategorik kolon için encoder ile numeric dönüşümü yaptık.
    x_train[col] = LabelEncoder().fit_transform(x_train[col]) 
    


regressor = DecisionTreeClassifier(criterion = "entropy")
regressor = regressor.fit(x_train.iloc[:,:11],x_train.iloc[:,11:]) 
#0-11 kolon arasını bağımsız değişken olarak koyduk
#11. kolonu da bağımsız değişken olarak


category_col = ['Term','Years in current job','Home Ownership','Purpose']

for col in category_col: 
    #Her bir kategorik kolon için encoder ile numeric dönüşümü yaptık.
    x_test[col] = LabelEncoder().fit_transform(x_test[col]) 
    x_validation[col] = LabelEncoder().fit_transform(x_validation[col])
    


y_test2 = pd.DataFrame(data = y_test , index = range(13278) , columns = ['Decision'])
"""
y_test'imiz bir dizi.Dolayısıyla aşağıdaki .apply() metodunu kullanamıyorduk
Bu yüzden dizinin boyu kadar (13278) satır olacak ve içeriği aynı olacak şekilde dataframe oluşturduk.
"""


y_test2 = y_test2.apply(LabelEncoder().fit_transform) #Oluşturduğumuz dataframe'e encoder uyguladık.
x_test_list = []
x_validation_list= []
for index, instance in x_test.iterrows():
        """
        x_test'deki her bir satır bir dizi olacak şekilde x_test_list'e atıyoruz.
        """
        x_test_list.append(instance)
  
for index, instance in x_validation.iterrows(): 
        """
        x_validation'deki her bir satır bir dizi olacak şekilde x_validation_list'e atıyoruz.
        """
        x_validation_list.append(instance)    
        

c45_predict = regressor.predict(x_test_list) #Bu örnekler listesine .predict() uygulayarak predict listesi oluşturuyoruz.
validation_predict = regressor.predict(x_validation_list) #validasyon değerlerin de test etmiş olduk.
"""
validation_predict değişkenini içinde 0 ve 1 (full paid - charged off) değerleri rastgele ayrılmış mı?
Yani aynı çıktı verilip verilmediği kontrolü için.
"""

c45_metric = metrics.confusion_matrix(y_test2,c45_predict)
print("Ağacı budamadan önceki karmaşıklık matrisi ve doğruluk değeri")
print(c45_metric)
print("Accuracy :",metrics.accuracy_score(y_test2,c45_predict))

"""
Ağacı çizdirmek için gerekli parametreleri oluşturduk feature_names - target_names.
"""
feature_names = x_train.columns[:11]
target_names = ['0','1']




prune_index(regressor.tree_, 0, 50) #Pruning işlemi

"""
Pruning işlemş uyguladıktan sonra ağacı tekrar tahmin-başarı testine sokuyoruz.
"""
c45_predict = regressor.predict(x_test_list)
c45_metric = metrics.confusion_matrix(y_test2,c45_predict)
validation_predict = regressor.predict(x_validation_list)
print("Ağacı budadıktan sonraki karmaşıklık matrisi ve doğruluk değeri")
print(c45_metric)
print("Accuracy :",metrics.accuracy_score(y_test2,c45_predict))

"""
Aşağıda ağacı çizdirip savefig() diyerek dosya dizinine kaydediyoruz.
"""

"""
Ağacı Pruning işleminde threshold değerini 1000 verdiğimde çizdirip ağacın yapısına baktım.
Ancak bu şekil yaptığımda charged off örnekleri olmadığı için daima full paid çıktısı veriyordu model
Bu yüzden threshold'u 50 yaptım ancak ağacı çizdirmedim çünkü budanmış bile olsa çok derin bir ağaç oluşuyor.

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(regressor, 
                   feature_names=feature_names,  
                   class_names=target_names,
                   filled=True,
                   rounded=True,
                   )

fig.savefig("decistion_tree.png")
"""

"""
flask kullanırken regresyonun .pkl dosyası lazım olacağı için aşağıdakini yazdık.
"""
pickle.dump(regressor, open('kredi.pkl', 'wb'))







