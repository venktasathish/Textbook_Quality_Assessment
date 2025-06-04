from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from nltk.stem import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.ensemble import RandomForestClassifier


main = Tk()
main.title("Textbook Quality Assessment using ML")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test, labels
global accuracy, precision, recall, fscore, rf_model, sc, tfidf_vectorizer
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset(): 
    global filename, X, Y, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels = ['Bad', 'Average', 'Good', 'Excellent']
    #plot graph of different labels found in dataset
    unique, count = np.unique(dataset['reviewer_rating'], return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize=(6,3))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Book Quality")
    plt.ylabel("Count")
    plt.title("Dataset Class Label Graph")
    plt.tight_layout()
    plt.show()        

def processDataset():
    global dataset, X, Y, sc, tfidf_vectorizer
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load("model/X.npy")
        Y = np.load("model/Y.npy")
    else:
        Y = dataset['reviewer_rating']
        reviews = dataset['review_description']
        books = dataset['book name']
        X = []
        for i in range(len(reviews)):
            name = books[i]
            data = name+" "+reviews[i]
            data = data.strip("\n").strip().lower()
            data = cleanText(data)#clean description
            X.append(data)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X", X)
        np.save("model/Y", Y)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=600)
    X = tfidf_vectorizer.fit_transform(X).toarray()
    temp = pd.DataFrame(X, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,"TF-IDF Word2Vec Values\n\n")
    text.insert(END,str(temp))
    

def splitDataset():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test, sc
    sc = StandardScaler()
    X = sc.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    text.insert(END,"Dataset Normalization & Shuffling Completed\n\n")
    text.insert(END,"Normalized Values = "+str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train & Test Split\n")
    text.insert(END,"80% dataset size used to train algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset size used to test algorithms : "+str(X_test.shape[0])+"\n")
    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)

def calculateMetrics(algorithm, predict, y_test):
    global labels
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()    

def runRandomForest():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, rf_model
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    rf_model = RandomForestClassifier() #create Random Forest object
    rf_model.fit(X_train, y_train)
    predict = rf_model.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def predict():
    text.delete('1.0', END)
    global rf_model, labels, sc, tfidf_vectorizer
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test = test.values
    for i in range(len(test)):
        data = test[i,0]
        data = data.strip().lower()
        data = cleanText(data)
        data = tfidf_vectorizer.transform([data]).toarray()
        data = sc.transform(data)
        predict = rf_model.predict(data)[0]
        predict = int(predict)
        predict = predict - 2
        print(predict)
        label = labels[predict]
        text.insert(END,"Book Details = "+str(test[i,0])+"\n")
        text.insert(END,"Predicted Book Quality = "+str(label)+"\n\n")
        
    
def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='Textbook Quality Assessment using ML')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Book Reviews Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=20,y=150)
processButton.config(font=ff)

splitButton = Button(main, text="Split Dataset Train & Test", command=splitDataset)
splitButton.place(x=20,y=200)
splitButton.config(font=ff)

rfButton = Button(main, text="Train Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=20,y=250)
rfButton.config(font=ff)

predictButton = Button(main, text="Predict Book Quality", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=350)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
