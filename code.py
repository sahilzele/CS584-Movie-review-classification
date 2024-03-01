import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

import regex as re
import contractions
from tqdm import tqdm
from nltk.stem import PorterStemmer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import GridSearchCV



'''importing test and traing files into data frames'''
    
Trainfilepath = "/Users/sahil/Documents/CS584/Asst1/TrainSet.txt"
Testfilepath = "/Users/sahil/Documents/CS584/Asst1/testSet.txt"

#creatting dataframe and importing datasets
Traindf = pd.read_csv(Trainfilepath,engine='python',delimiter='    ',header = None)
Testdf = pd.read_csv(Testfilepath,engine='python',delimiter='    ',header = None)

Traindf.columns = ['columns']

#Spliting score and reviews in the training df and renaming the columns
Traindf[['score','review']] = Traindf['columns'].str.split('\t', n= 1, expand = True)

#Importing stopwords
stp_words=stopwords.words('english')


''' Data cleaning for Training set'''
#creating a new list and appending cleaned data
processed_train_reviews=[]
for i in tqdm(Traindf['review']):
    #Regular expression that removes all the html tags pressent in the reviews
    i=re.sub('(<[\w\s]*/?>)',"",i)
    i=i.replace("-"," ")
    i=i.replace("#EOF","")
    #Expanding all the contractions present in the review to is respective actual form
    i=contractions.fix(i)
    #Removing all the special charactesrs from the review text
    i=re.sub('[^a-zA-Z0-9\s]+',"",i)
    #Removing all the digits present in the review text
    i=re.sub('\d+',"",i)
    #making the all text in same case
    i = i.lower()
    #Making all the review text to be of lower case as well as remvoing the stopwords and words of length less than 3
    processed_train_reviews.append(" ".join([j for j in i.split() if j not in stp_words and len(j)>=3]))

#creating a new df with cleaned data
cleaned_Train_df=pd.DataFrame({'review':processed_train_reviews,'sentiment':list(Traindf['score'])})


''' Data cleaning for Testing set''' 
Testdf.columns = ['reviews']
#creating a new list and appending cleaned data
processed_test_reviews=[]
for i in tqdm(Testdf['reviews']):
    #Regular expression that removes all the html tags pressent in the reviews
    i=re.sub('(<[\w\s]*/?>)',"",i)
    i=i.replace("-"," ")
    i=i.replace("#EOF","")
    #Expanding all the contractions present in the review to is respective actual form
    i=contractions.fix(i)
    #Removing all the special charactesrs from the review text
    i=re.sub('[^a-zA-Z0-9\s]+',"",i)
    #Removing all the digits present in the review text
    i=re.sub('\d+',"",i)
    #making the all text in same case
    i = i.lower()
    #Making all the review text to be of lower case as well as remvoing the stopwords and words of length less than 3
    processed_test_reviews.append(" ".join([j for j in i.split() if j not in stp_words and len(j)>=3]))

#creating a new df with cleaned data
cleaned_Test_df=pd.DataFrame({'reviews':processed_test_reviews})


#defining stem_text method for stemming 
stemmer = PorterStemmer()
def stem_text(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)

#updating both the datasets after stemming
cleaned_Train_df['review'] = cleaned_Train_df['review'].apply(stem_text)
cleaned_Test_df['reviews'] = cleaned_Test_df['reviews'].apply(stem_text)

#calculating tf-idf vectors
tfidf = TfidfVectorizer(max_features = 2500,ngram_range=(1,2))
tfidf_train_vector = tfidf.fit_transform(cleaned_Train_df['review'] ).toarray()
tfidf_test_vector = tfidf.transform(cleaned_Test_df['reviews'] ).toarray()
features = tfidf.get_feature_names_out()


'''Hyper parameter tuning'''
#declaring knn model object
knn=KNeighborsClassifier()

#defining the grid parameters for hyper parameter tuning having n_neighbours in range(10,690,30)
param_grid = {
    'n_neighbors': [10,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690],
    'weights': ['uniform', 'distance'],'metric': ['euclidean','manhattan']}

#tuning using RandomizedSearchCV
cv = RandomizedSearchCV(knn, param_grid, random_state=0,n_jobs=-1,scoring = 'accuracy',verbose=1)
cv.fit(tfidf_train_vector,cleaned_Train_df['sentiment'])

#tuning using GridSearchCV
#cv = GridSearchCV(estimator=knn, param_grid= param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#cv.fit(tfidf_train_vector,cleaned_Train_df['sentiment'])

#pulling the best parameters discovered by Hyper parameter tuning
best_params = cv.best_params_
#print(best_params)

#defining new knn object with best parameters and fitting the model with cleaned train dataset
best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'],metric=best_params['metric'])
best_knn.fit(tfidf_train_vector,cleaned_Train_df['sentiment'])

#checking cross validation results from hyper parameter tuning
acc = cross_val_score(best_knn,tfidf_train_vector,cleaned_Train_df['sentiment'], cv=5)
print(acc.mean())

#predicting and storing the result in output list
output = best_knn.predict(tfidf_test_vector)

#Output file creation 
file_path = '/Users/sahil/Documents/CS584/format.dat'
# Open the file in write mode ('w')
with open(file_path, 'w') as file:
    # Write each element of the list to the file
    for item in output:
        file.write(f'{item}\n')
        
