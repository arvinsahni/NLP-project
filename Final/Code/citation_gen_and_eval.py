import pandas as pd
import numpy as np
import random
import math
import nltk
import sys
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from operator import itemgetter
import pickle
import string

#dstem and tokenize words
stopwords = nltk.corpus.stopwords.words('english')
stopwords_punct = stopwords + list(set(string.punctuation))
stemmer = SnowballStemmer("english")
def stem_tokenizer(text):
    words = [word for word in nltk.word_tokenize(text)]
    words_no_stop = [word for word in words if not word in stopwords_punct]
    words_stemmed = [stemmer.stem(word) for word in words_no_stop]
    return words_stemmed

#clean up names (got creative in BigQuery) and generate joint text columns
cit_lists = pd.read_csv("patents_with_cites.csv")
print "data read in"
patents_with_cites = cit_lists[["pat_p_id", "pat_p_type", "pat_p_number", "pat_p_date", "pat_p_abstract", "pat_p_title", 
                             "pat_p_kind", "c_patents_cited"]]
data = patents_with_cites.rename(columns = {"pat_p_id": "p_id", "pat_p_type": "type", "pat_p_number": "number", "pat_p_date": "date", 
                           "pat_p_abstract" : "abstract", "pat_p_title" : "title", 
                             "pat_p_kind" : "kind", "c_patents_cited" : "Patents_Cited"})
data.fillna(" ", inplace = True)
data["date"]=pd.to_datetime(data["date"])
data["Patents_Cited"] = data['Patents_Cited'].map(lambda x : list(set(x.split(","))))
data["combined_text_plain"] = data["title"] + data["abstract"]
print "Tokenizing!"
data["combined_text"] = data["combined_text_plain"].apply(stem_tokenizer)

#training and test split
random.seed(10)
train=data.sample(frac=0.9,random_state=200)
test=data.drop(train.index)
data = train

print "Writing test to csv!"
test.to_csv("test_patents.csv")
print "Test written to csv!"

#topic modelling clusters 
print "Doing Topic Modelling!"
full_dict = corpora.Dictionary(data['combined_text'])
DT_matrix = [full_dict.doc2bow(doc) for doc in data['combined_text']]

#parralelized topic model
lda = LdaMulticore(DT_matrix, id2word=full_dict, num_topics=100) 
print "Topics Modelled!"


#this list comprehension is a crime against humanity
#it generates the list of probably topics and the most probable topic
topicList = [[i[0] for i in sorted(lda.get_document_topics(full_dict.doc2bow(doc),
			 minimum_probability = .05),key=itemgetter(1), reverse = True)]
             for doc in data['combined_text']]
#this hack fixes the very occasional patent that does not have a topic
topicList = [ i if len(i)>=1 else [0] for i in topicList]
data["all_topics"] = topicList
data["main_topic"] = [i[0] for i in topicList]

print "Writing to csv"
data.to_csv("topic_modeled_patents.csv")
print "written to csv"

#TFIDF Vectorization
vector = TfidfVectorizer(max_df=0.8, min_df=0.2, stop_words='english',
                                 use_idf=True, #tokenizer=stem_tokenizer,
                         ngram_range=(1,3))
print "Vectorized"
## we need to create a feature matrix for each main_topic
## column names will be values of cosine, jaccard, csine_second_order and a Yes/NO 
## rows will be tuples of patent_ids


## Find all unique topics in the dataframe
## run cosine similarity and tf and idf within the group only
## sort cosine similarity in descending order
## store list of docs in descending order of similarity

## calculate jaccard distance by first converting to dense form
## some values are NaN and are converted to sys.maxint before saving into features

## calculate 2nd order cosine similarity as specified in the cell above 


import time
start = time.clock()

print "Starting Similarities and Modelling"

main_topic_list=sorted(data.main_topic.unique())
#main_topic_list=[0]
print main_topic_list
for i in main_topic_list:
    #print "\n\ni=",i
    print i
    temp=data.loc[data['main_topic'] == i]
    print "data subsetted by topic"
    print "vector fitting"
    vector_fit = vector.fit_transform(temp.combined_text_plain)
    print "vector fit, doing cosine!"
    cosine_sim = cosine_similarity(vector_fit)
    print "Done cosine, starting jaccard"
    jaccard_dist=pairwise_distances(vector_fit.todense(), metric='jaccard')
    print "Done jaccard, doing second order"
    ## second-order cosine calculation
    cosine_sim_2_denom=np.sqrt((cosine_sim**2).sum(axis=1))
    cosine_sim_order2= np.zeros((cosine_sim.shape[0], cosine_sim.shape[0]))

    for p in range(cosine_sim.shape[0]):
        for q in range(cosine_sim.shape[0]):
            if cosine_sim_2_denom[p]!=0 and cosine_sim_2_denom[q]!=0:
                cosine_sim_order2[p,q]=np.dot(cosine_sim[p],cosine_sim[q])/(cosine_sim_2_denom[p]*cosine_sim_2_denom[q])
    print "Done second order!" 
    ## second-order cosine calculation end

    ## creating a feature matrix
    print "Creating feature matrix"
    feature_matrix= np.zeros((cosine_sim.shape[0]**2, 5))
    m=0
    for p in range(cosine_sim.shape[0]):   
        for q in range(cosine_sim.shape[0]):
            feature_matrix[m,1]=cosine_sim[p,q]
            if math.isnan(jaccard_dist[p,q]):
                feature_matrix[m,2]=sys.maxint
            else:
                feature_matrix[m,2]=jaccard_dist[p,q]
                
            feature_matrix[m,3]=cosine_sim_order2[p,q]
            if temp.iloc[q]['p_id'] in temp.iloc[p]['Patents_Cited']:
                feature_matrix[m,4]=1  
            else:
                feature_matrix[m,4]=0  
            ## override with pretend value for now
            #feature_matrix[m,4]=(p+q+1)%2  ## Actual Yes/No
            m=m+1
    print "Done feature matrix, starting regresion modelling"
    #print "feature",feature_matrix
    
    ## Logistic Regression Model
    X=feature_matrix[:,1:3]
    y=feature_matrix[:,4]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print "Model fit! Pickling!"
    ## saving model for use later for prediction
    filename = 'finalized_model'+str(i)+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    ## 
    print "pickeled, predicting"
    predicted = model.predict(X_test)
    probs = model.predict_proba(X_test)
    print "Done!"

print time.clock() - start

print "Finished Modelling"

#start prediction on test set!

data_prediction = test

#identify topics for test set based on trained models!
topicList_pred = [[i[0] for i in sorted(lda.get_document_topics(full_dict.doc2bow(doc),
				 minimum_probability = .05),key=itemgetter(1), reverse = True)]
             for doc in data_prediction['combined_text']]
#this hack fixes the very occasional patent that does not have a prediction
topicList_pred = [ i if len(i)>=1 else [0] for i in topicList_pred]
data_prediction["all_topics"] = topicList_pred
data_prediction["main_topic"] = [i[0] for i in topicList_pred]

data_prediction["date"]=pd.to_datetime(data_prediction["date"])
data["date"]=pd.to_datetime(data["date"])
data_prediction.count()

print "Starting Prediction!"

start = time.clock()

main_topic_list_pred=sorted(data_prediction.main_topic.unique())
#data_prediction["Predicted_Cites"] = None
predictions = pd.DataFrame()
for i in main_topic_list_pred:
    #print "\n\ni=",i

    temp=data.loc[data['main_topic'] == i]
    temp_pred=data_prediction.loc[data_prediction['main_topic'] == i]
    #the patents will never be in the full data now with proper train test split
    temp_combo=pd.concat([temp, temp_pred], ignore_index = True)
  
    
    vector_fit = vector.fit_transform(temp_combo.combined_text_plain)
    cosine_sim = cosine_similarity(vector_fit)
    jaccard_dist=pairwise_distances(vector_fit.todense(), metric='jaccard')
    
    ## second-order cosine calculation
    cosine_sim_2_denom=np.sqrt((cosine_sim**2).sum(axis=1))
    cosine_sim_order2= np.zeros((cosine_sim.shape[0], cosine_sim.shape[0]))

    for p in range(cosine_sim.shape[0]):
        for q in range(cosine_sim.shape[0]):
            if cosine_sim_2_denom[p]!=0 and cosine_sim_2_denom[q]!=0:
                cosine_sim_order2[p,q]=np.dot(cosine_sim[p],cosine_sim[q])/(cosine_sim_2_denom[p]*cosine_sim_2_denom[q])
    ## second-order cosine calculation end

    ## creating a feature matrix
    m_list=[]
    feature_matrix_pred= np.zeros((cosine_sim.shape[0]**2, 5))
    m=0
    for p in range(cosine_sim.shape[0]):   
        for q in range(cosine_sim.shape[0]):
            feature_matrix_pred[m,1]=cosine_sim[p,q]
            if math.isnan(jaccard_dist[p,q]):
                feature_matrix_pred[m,2]=sys.maxint
            else:
                feature_matrix_pred[m,2]=jaccard_dist[p,q]
                
            feature_matrix_pred[m,3]=cosine_sim_order2[p,q]
            if temp_combo.iloc[q]['p_id'] in temp_combo.iloc[p]['Patents_Cited']:
                feature_matrix_pred[m,4]=1  
            else:
                feature_matrix_pred[m,4]=0  
            
            if temp_combo.iloc[p]['p_id'] in list(temp_pred["p_id"]):
                m_list.append(m)
            m=m+1
    
    feature_matrix_pred_new=feature_matrix_pred[m_list,:]

    
    ## Logistic Regression Model
    X_test=feature_matrix_pred_new[:,1:3]
    
    # load the model from disk 
    filename = 'finalized_model'+str(i)+'.sav'

    
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    predicted=loaded_model.predict(X_test)

    
    temp_pid_pred=[]
    temp_patent_id =[]
    for k in range(len(predicted)/cosine_sim.shape[0]):
        #print "k=",k
        temp_temp=[]

        for r in range((k*cosine_sim.shape[0]),((k+1)*cosine_sim.shape[0])):
            #print "r=",r
            if (int(predicted[r])==1 and (temp_combo.iloc[r%cosine_sim.shape[0]]["date"]<=temp_pred.iloc[k]["date"])):
                temp_temp.append(temp_combo.iloc[r%cosine_sim.shape[0]]["p_id"])
                #temp_pid_pred[k/cosine_sim.shape[0]].append(temp_combo.iloc[(k%cosine_sim.shape[0])]["p_id"])
        temp_pid_pred.append(temp_temp)
        temp_patent_id.append(temp_pred.iloc[k]["p_id"])
    preds_data_frame = pd.DataFrame({'patent_id': temp_patent_id, 'Predicted_Patents': temp_pid_pred})
    predictions = pd.concat([preds_data_frame, predictions])
        
print time.clock() - start

print "Predictions Done"

for_eval = pd.merge(data_prediction, predictions, left_on = 'p_id', right_on = 'patent_id')

def average_precision(actual, predicted, limit = 20):
    '''
    actual is the list of cited patents, in no particular order
    predicted is the predicted list of citation, in order of likelyhood
    Limit is the number of predicted patents to check
    '''
    if len(predicted) > limit:
        predicted = predicted[:limit]
    score = 0.0
    num_correct = 0.0 
    for i in range(len(predicted)):
        if predicted[i] in actual:
            num_correct +=1
            score += num_correct/(i+1)
    if not actual:
        #incase a patent does not have any citations in the time period
        return 0.0

    return score / float(len(actual))
def MAP(actual, predicted):
    '''
    maps previous function across a series of list
    actual is a list of lists of the actual citations
    predicted is a list of lists of the predictions
    '''
    return np.mean([average_precision(act, pred) for act,pred in zip(actual, predicted)])   

predicted = for_eval["Patents_Cited"]
actual = for_eval["Predicted_Patents"]
print "The MAP is:", MAP(actual, predicted)












