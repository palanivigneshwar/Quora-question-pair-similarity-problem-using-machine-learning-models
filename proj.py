import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords,wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt
import time 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(r"train.csv")
df.head()

df.question1= df.question1.fillna('')

df.question2.isna().sum()

df.question2= df.question2.fillna('')

# lemmatizer = nltk.stem.WordNetLemmatizer()
# df['question1']=df['question1'].apply(lambda ques: ' '.join([lemmatizer.lemmatize(tkn) for tkn in list(ques.split())]))
# df['question2']=df['question2'].apply(lambda ques: ' '.join([lemmatizer.lemmatize(tkn) for tkn in list(ques.split())]))

df['question1'] = df['question1'].apply(lambda x: x.lower() )
df['question2'] = df['question2'].apply(lambda x: x.lower() )

df['question1'] = df['question1'].apply(lambda x: re.compile(r'<[^>]+>').sub('',x))
df['question2'] = df['question2'].apply(lambda x: re.compile(r'<[^>]+>').sub('',x))

def punctuation(x):
  a = x.maketrans(' ',' ',string.punctuation)
  a1 = x.translate(a)

  return a1

df['question1'] = df['question1'].apply(lambda x: punctuation(x) )
df['question2'] = df['question2'].apply(lambda x: punctuation(x) )

df['question1'] = df['question1'].apply(lambda x: re.sub(r'\d+','',x))
df['question2'] = df['question2'].apply(lambda x: re.sub(r'\d+','',x))

df['question1'] = df['question1'].apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*','',x))
df['question2'] = df['question2'].apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*','',x))

stopwords = set(stopwords.words('english'))

df['question1'] = df['question1'].apply(lambda x: ' '.join([s for s in x.split() if s not in stopwords]))
df['question2'] = df['question2'].apply(lambda x: ' '.join([s for s in x.split() if s not in stopwords]))

lemmatizer = WordNetLemmatizer()
df['question1'] = df['question1'].apply(lambda x: lemmatizer.lemmatize(x))
df['question2'] = df['question2'].apply(lambda x: lemmatizer.lemmatize(x))

def cosine(row,row1):
  word1 = row.split()
  word2 = row1.split()

  word1 = {s for s in word1 if not s in stopwords }
  word2 = {s for s in word2 if not s in stopwords }

  u = word1.union(word2)
  l1=[]
  l2=[]
  for w in u:
    if w in word1:
      l1.append(1)
    else:
      l1.append(0)

    if w in word2:
      l2.append(1)
    else:
      l2.append(0)

  l1=np.asarray(l1)
  l2=np.asarray(l2)

  num = np.dot(l1,l2.T)

  # sum1=0
  # sum2=0
  # for i in l1:
  #   sum1+=i*i
  # for j in l2:
  #   sum2+=j*j

  den = np.sqrt(np.sum(np.square(l1))) + np.sqrt(np.sum(np.square(l2)))

  cos = num/den

  return cos

df['cosine'] = df.apply(lambda x: cosine(x['question1'],x['question2']),axis=1)

from wordcloud import WordCloud
q1 = df['question1'].str.cat(sep=' ')

wordcloud = WordCloud().generate(str(q1))

plt.figure(figsize=(9,9))
plt.imshow(wordcloud)
plt.axis("off")

q2 = df['question2'].str.cat(sep=' ')

wordcloud1 = WordCloud().generate(str(q2))

plt.figure(figsize=(9,9))
plt.imshow(wordcloud1)
plt.axis("off")

df['question1'] = df['question1'].apply(lambda x: x.split())
df['question2'] = df['question2'].apply(lambda x: x.split())



# cosine(df.question1[6],df.question2[6])

A = df[df.is_duplicate==1]
A.head()

# cosine(df.question1[12],df.question2[12])

df.question1[8]

# import gzip
# with open(r"GoogleNews-vectors-negative300.bin.gz","rb") as f:
#     file = f.read()

# f.close()
# file
!pip install gensim

from gensim.models import KeyedVectors,Word2Vec

file = r"GoogleNews-vectors-negative300.bin"

S1 = df['question1'].values
S2 = df['question2'].values

S1

S = np.concatenate((S1,S2))
S.shape

model = Word2Vec(S,size=100,min_count=1,window=5)

def average_vectors(m,a):

  text = []

  for b in a:
    if b in m.wv.vocab:
      text.append(b)

  if(len(text)>0):
    k = np.mean(m[text],axis=0)
  else:
    k=[]

  return k

df['question1'] = df['question1'].apply(lambda x: average_vectors(model,x))
df['question2'] = df['question2'].apply(lambda x: average_vectors(model,x))

df.head()

# new_df = pd.DataFrame(vec1)
# new_df.head()

# vector = TfidfVectorizer(analyzer="word",token_pattern=r'\w{1,}',max_features=5000)

# vector.fit(pd.concat((df['question1'],df['question2'])).unique())

# df.drop()

# v1 = vector.transform(df['question1'].values)
# v2 = vector.transform(df['question2'].values)

# type(v2)
q1 = df['question1'].values
q2 = df['question2'].values


q3 = []
q4 = []
for w in q1:
  q3.append(w)

for w in q2:
  q4.append(w)

len(q3)

n_df = pd.DataFrame(q3)
n1_df = pd.DataFrame(q4)
map={}
for i in range(100):
  map[i] = i+100
n1_df.rename(columns=map,inplace=True)
n2_df = pd.concat((n_df,n1_df),axis=1)
n2_df.head()

print(n2_df.shape,df.shape)
n2_df.isnull().sum()

df1 = pd.concat((df,n2_df),axis=1)

df1 = pd.read_csv(r"preprocessed_data.csv")
df1.head()

# from sklearn.metrics.pairwise import cosine_similarity
import scipy

# cos = cosine_similarity(v1,v2)

# cos
# X = scipy.sparse.hstack((v1,v2))
# dfv1 = pd.DataFrame(nv1)
# dfv1.head()
X = df1.drop(['id','qid1','qid2','is_duplicate'],axis=1)
df2 = df1.drop(['id','question1','question2'],axis=1)

df2.duplicated().sum()

# df['cosine'] = df.apply(lambda x: cosine(x['question1'],x['question2']),axis=1) 
X.drop(['question1','question2'],axis=1,inplace=True)

X.head()

# df.drop(['id','qid1','qid2','question1','question2'],axis=1,inplace=True)
X.fillna(0,inplace=True)

X.isnull().sum()

# X=df.cosine
Y=df1.is_duplicate

from sklearn.preprocessing import StandardScaler,MinMaxScaler
X = StandardScaler().fit_transform(X)
X = MinMaxScaler().fit_transform(X)

from sklearn.decomposition import PCA
p = PCA(2)
reduced = p.fit_transform(X)

plt.scatter(reduced[:,0],reduced[:,1],c=Y,cmap='Paired')

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.7,random_state=0)

X_train.shape

xtrain1 = p.fit_transform(X_train)
xtest1 = p.fit_transform(X_test)

from sklearn.metrics import accuracy_score,log_loss

def plot_dec_boundary(c,x):
    plt.figure(figsize=(10,10))
    # positive_label = reduc[Y==1]
    # negative_label = reduc[Y==-1]
    # plt.scatter(positive_label[:,0],positive_label[:,1],label="1",s=20)
    # plt.scatter(negative_label[:,0],negative_label[:,1],label="-1",s=20)
    plt.scatter(x[:,0],x[:,1],c=Y_train,s=20,cmap="Paired")
    plt.legend()
    ax = plt.gca()

    x1 = ax.get_xlim()
    y2 = ax.get_ylim()

    x3 = np.linspace(x1[0], x1[1], 50)
    y3 = np.linspace(y2[0], y2[1], 50)
    x4,y4 = np.meshgrid(x3,y3)
    new_arr = np.vstack([x4.ravel(),y4.ravel()]).T
    out = c.decision_function(new_arr).reshape(x4.shape)

    ax.contour(
        x4, y4, out, colors="k", levels = [0,1], alpha=0.5, linestyles = ["-","--"]
    )


    plt.show()

def plot_dec_boundary_svc(c,x):
    plt.figure(figsize=(10,10))
    plt.scatter(x[:,0],x[:,1],c=Y_train,s=20,cmap="Paired")
    plt.legend()
    ax = plt.gca()

    x1 = ax.get_xlim()
    y2 = ax.get_ylim()

    x3 = np.linspace(x1[0], x1[1], 50)
    y3 = np.linspace(y2[0], y2[1], 50)
    x4,y4 = np.meshgrid(x3,y3)
    new_arr = np.vstack([x4.ravel(),y4.ravel()]).T
    out = c.decision_function(new_arr).reshape(x4.shape)

    ax.contour(
        x4, y4, out, colors="k", levels = [-1,0,1], alpha=0.5, linestyles = ["--","-","--"]
    )


    plt.show()

"""### Decision tree"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,log_loss

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,Y_train)

import joblib
# joblib.dump(clf,"dt")
clf11 = joblib.load('dt')

pred_train = clf11.predict(X_train)
pred = clf11.predict(X_test)

print("training accuracy :",accuracy_score(Y_train,pred_train))
print("testing accuracy :",accuracy_score(Y_test,pred))

pred_train_prob = clf11.predict_proba(X_train)
pred_prob = clf11.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob))
print("log loss of test set : ",log_loss(Y_test,pred_prob))

print(classification_report(Y_train,pred_train))

print(classification_report(Y_test,pred))

path = clf11.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas,impurities = path.ccp_alphas,path.impurities

ccp_alphas.shape

import time
from sklearn.tree import DecisionTreeClassifier
dts = []
for ccp in ccp_alphas[11100:16100:1000]:
  start = time.time()
  clf1 = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp)
  clf1.fit(X_train,Y_train)
  dts.append(clf1)
  print(time.time()-start)

import joblib
# joblib.dump(dts,"dts")
dts = joblib.load('dts')

train_acc = [accuracy_score(Y_train,clf2.predict(X_train)) for clf2 in dts]
test_acc = [accuracy_score(Y_test,clf2.predict(X_test)) for clf2 in dts]
train_acc_prob = [log_loss(Y_train,clf2.predict_proba(X_train)) for clf2 in dts]
test_acc_prob = [log_loss(Y_test,clf2.predict_proba(X_test)) for clf2 in dts]

print(train_acc)
print(test_acc)
print(train_acc_prob)
print(test_acc_prob)

plt.plot(ccp_alphas[15500:16000:100],train_acc,label="training")
plt.plot(ccp_alphas[15500:16000:100],test_acc,label="testing")
plt.legend()
plt.xlabel('ccp_alphas')
plt.ylabel('accuracy')

plt.plot(ccp_alphas[15500:16000:100],train_acc_prob,label="training")
plt.plot(ccp_alphas[15500:16000:100],test_acc_prob,label="testing")
plt.legend()
plt.xlabel('ccp_alphas')
plt.ylabel('logloss')

dec_tree = pd.DataFrame()
dec_tree['ccp_alpha'] = ccp_alphas[15500:16000:100]
dec_tree['train accuracy'] = train_acc
dec_tree['test accuracy'] = test_acc
dec_tree['train log loss'] = train_acc_prob
dec_tree['test log loss'] = test_acc_prob
dec_tree


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,log_loss

clf = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alphas[15500:16000:100][4])
clf.fit(X_train,Y_train)

pred_train = clf.predict(X_train)
pred = clf.predict(X_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train))
print("testing accuracy : ",accuracy_score(Y_test,pred))

pred_train_prob = clf.predict_proba(X_train)
pred_prob = clf.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob))
print("log loss of test set : ",log_loss(Y_test,pred_prob))

print(classification_report(Y_train,pred_train))

print(classification_report(Y_test,pred))

"""### XGBOOST"""

import joblib

from xgboost import XGBClassifier

clf1 = XGBClassifier(max_depth=10, n_estimators=80,use_label_encoder=False)
clf1.fit(X_train,Y_train)

pred_train1 = clf1.predict(X_train)
pred1 = clf1.predict(X_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train1))
print("testing accuracy : ",accuracy_score(Y_test,pred1))
pred_train_prob6 = clf1.predict_proba(X_train)
pred_prob6 = clf1.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob6))
print("log loss of test set : ",log_loss(Y_test,pred_prob6))

print(classification_report(Y_train,pred_train1))

print(classification_report(Y_test,pred1))

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import time
start = time.time()
max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60,55]
train5 = []
test5 = []
clf5 = []
for i in range(5):
    clf2 = XGBClassifier(max_depth=max_depth1[i], n_estimators=n_estimators1[i],use_label_encoder=False)
    clf2.fit(X_train,Y_train)

    pred_train1 = clf2.predict(X_train)
    pred1 = clf2.predict(X_test)
    clf5.append(clf2)
    train5.append(accuracy_score(Y_train,pred_train1))
    test5.append(accuracy_score(Y_test,pred1))
    print(time.time()-start)

train5 = [accuracy_score(Y_train,c.predict(X_train)) for c in clf5]
test5 = [accuracy_score(Y_test,c.predict(X_test)) for c in clf5]
log_train5 = [log_loss(Y_train,c.predict_proba(X_train)) for c in clf5]
log_test5 = [log_loss(Y_test,c.predict_proba(X_test)) for c in clf5]

print(train5)
print(test5)
print(log_train5)
print(log_test5)

xg = pd.DataFrame()
max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60]
xg['max_depth'] = max_depth1
xg['n_estimators'] = n_estimators1
xg['train accuracy'] = train5
xg['test accuracy'] = test5
xg['train log loss'] = log_train5
xg['test log loss'] = log_test5
xg

max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60]
plt.plot(max_depth1,train5,label="trainig accuracy")
plt.plot(max_depth1,test5,label="testing accuracy")
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('accuracy')

plt.show()

max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60]
plt.plot(max_depth1,log_train5,label="trainig log loss")
plt.plot(max_depth1,log_test5,label="testing log loss")
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('logloss')

plt.show()

plt.plot(n_estimators1,train5,label="trainig accuracy")
plt.plot(n_estimators1,test5,label="testing accuracy")
plt.legend()
plt.xlabel('no. of estimators')
plt.ylabel('accuracy')

plt.show()

plt.plot(n_estimators1,log_train5,label="trainig log loss")
plt.plot(n_estimators1,log_test5,label="testing log loss")
plt.legend()
plt.xlabel('no. of estimators')
plt.ylabel('logloss')

plt.show()

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import accuracy_score

clf2 = LogisticRegression(max_iter=600,C=0.3)
clf2.fit(X_train,Y_train)
pred_train2 = clf2.predict(X_train)
pred2 = clf2.predict(X_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train2))
print("test accuracy : ",accuracy_score(Y_test,pred2))
pred_train_prob2 = clf2.predict_proba(X_train)
pred_prob2 = clf2.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob2))
print("log loss of test set : ",log_loss(Y_test,pred_prob2))

print(classification_report(Y_train,pred_train2))

print(classification_report(Y_test,pred2))

from sklearn.linear_model import LogisticRegression,SGDClassifier
start = time.time()
max_iter1 = [100,200,250,300,600]
c_val = [1,0.8,0.7,0.5,0.3]
train10 = []
test10 = []
clf15 = []
for i in range(5):
    clf2 = LogisticRegression(max_iter = max_iter1[i],C=c_val[i])
    clf2.fit(X_train,Y_train)
    pred_train2 = clf2.predict(X_train)
    pred2 = clf2.predict(X_test)
    clf15.append(clf2)
    train10.append(accuracy_score(Y_train,pred_train2))
    test10.append(accuracy_score(Y_test,pred2))
    print(time.time()-start)

train_acc1 = [accuracy_score(Y_train,clf2.predict(X_train)) for clf2 in clf15]
test_acc1 = [accuracy_score(Y_test,clf2.predict(X_test)) for clf2 in clf15]
train_acc_prob1 = [log_loss(Y_train,clf2.predict_proba(X_train)) for clf2 in clf15]
test_acc_prob1 = [log_loss(Y_test,clf2.predict_proba(X_test)) for clf2 in clf15]

print(train_acc1)
print(test_acc1)
print(train_acc_prob1)
print(test_acc_prob1)

log = pd.DataFrame()
log['max_iter'] = max_iter1
log['C'] = c_val
log['train accuracy'] = train_acc1
log['test accuracy'] = test_acc1
log['train log loss'] = train_acc_prob1
log['test log loss'] = test_acc_prob1
log

plt.plot(max_iter1,train_acc1,label="trainig accuracy")
plt.plot(max_iter1,test_acc1,label="testing accuracy")
plt.legend()
plt.xlabel('max_iter')
plt.ylabel('accuracy')

plt.show()

plt.plot(max_iter1,train_acc_prob1,label="trainig log loss")
plt.plot(max_iter1,test_acc_prob1,label="testing log loss")
plt.legend()
plt.xlabel('max_iter')
plt.ylabel('logloss')
plt.show()

plt.plot(c_val,train_acc1,label="trainig accuracy")
plt.plot(c_val,test_acc1,label="testing accuracy")
plt.legend()
plt.xlabel('C')
plt.ylabel('accuracy')

plt.show()

plt.plot(c_val,train_acc_prob1,label="trainig logloss")
plt.plot(c_val,test_acc_prob1,label="testing logloss")
plt.legend()
plt.xlabel('C')
plt.ylabel('logloss')

plt.show()

clf2

clf2.fit(xtrain1,Y_train)
plot_dec_boundary(clf2,xtrain1)

"""### Random Forest"""

from sklearn.ensemble import RandomForestClassifier
import time
start = time.time()
max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60,55]
train6 = []
test6 = []
clf6 = []
for i in range(5):
    clf3 = RandomForestClassifier(n_estimators=n_estimators1[i],max_depth=max_depth1[i])

    clf3.fit(X_train,Y_train)
    pred_train3 = clf3.predict(X_train)
    pred3 = clf3.predict(X_test)
    
    clf6.append(clf3)
    train6.append(accuracy_score(Y_train,pred_train3))
    test6.append(accuracy_score(Y_test,pred3))
    print(time.time()-start)

train6 = [accuracy_score(Y_train,clf2.predict(X_train)) for clf2 in clf6]
test6 = [accuracy_score(Y_test,clf2.predict(X_test)) for clf2 in clf6]
log_train6 = [log_loss(Y_train,clf2.predict_proba(X_train)) for clf2 in clf6]
log_test6 = [log_loss(Y_test,clf2.predict_proba(X_test)) for clf2 in clf6]

print(train6)
print(test6)
print(log_train6)
print(log_test6)

max_depth1 = [10,20,30,40,50]
n_estimators1 = [80,75,70,65,60]
rf = pd.DataFrame()
rf['max_depth'] = max_depth1
rf['n_estimators'] = n_estimators1
rf['train accuracy'] = train6
rf['test accuracy'] = test6
rf['train log loss'] = log_train6
rf['test log loss'] = log_test6
rf

plt.plot(max_depth1,train6,label="trainig accuracy")
plt.plot(max_depth1,test6,label="testing accuracy")
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('accuracy')

plt.show()

plt.plot(max_depth1,log_train6,label="trainig log loss")
plt.plot(max_depth1,log_test6,label="testing log loss")
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('logloss')
plt.show()

plt.plot(n_estimators1,train6,label="trainig accuracy")
plt.plot(n_estimators1,test6,label="testing accuracy")
plt.legend()
plt.xlabel('no. of estimators')
plt.ylabel('accuracy')

plt.show()

plt.plot(n_estimators1,log_train6,label="trainig log loss")
plt.plot(n_estimators1,log_test6,label="testing log loss")
plt.legend()
plt.xlabel('no. of estimators')
plt.ylabel('logloss')
plt.show()


clf31 = RandomForestClassifier(n_estimators=55,max_depth=50)

clf31.fit(X_train,Y_train)
pred_train3 = clf31.predict(X_train)
pred3 = clf31.predict(X_test)

clf6.append(clf3)

print("training accuracy : ",accuracy_score(Y_train,pred_train3))
print("test accuracy : ",accuracy_score(Y_test,pred3))
pred_train_prob3 = clf31.predict_proba(X_train)
pred_prob3 = clf31.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob3))
print("log loss of test set : ",log_loss(Y_test,pred_prob3))

print(classification_report(Y_train,pred_train3))

print(classification_report(Y_test,pred3))

"""### SVC (Linear)"""

from sklearn.linear_model import LogisticRegression,SGDClassifier
clf3 = SGDClassifier(loss='hinge')
clf3.fit(X_train,Y_train)

pred_train3 = clf3.predict(X_train)
pred3 = clf3.predict(X_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train3))
print("test accuracy : ",accuracy_score(Y_test,pred3))
from sklearn.calibration import CalibratedClassifierCV
clf10 = CalibratedClassifierCV(base_estimator=clf3, cv="prefit")
clf10.fit(X_train,Y_train)
pred_train_prob3 = clf10.predict_proba(X_train)
pred_prob3 = clf10.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob3))
print("log loss of test set : ",log_loss(Y_test,pred_prob3))

print(classification_report(Y_train,pred_train3))

print(classification_report(Y_test,pred3))

clf3 = SGDClassifier(loss='hinge')
clf3.fit(xtrain1,Y_train)

clf3.fit(xtrain1,Y_train)
plot_dec_boundary_svc(clf3,xtrain1)


"""### SVC (RBF)"""

from sklearn.kernel_approximation import RBFSampler

rbf = RBFSampler(random_state=0,n_components=2500)
x_rbf_train = rbf.fit_transform(X_train)
x_rbf_test = rbf.fit_transform(X_test)

clf4 = SGDClassifier()
clf4.fit(x_rbf_train,Y_train)

pred_train4 = clf4.predict(x_rbf_train)
pred4 = clf4.predict(x_rbf_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train4))
print("test accuracy : ",accuracy_score(Y_test,pred4))
clf11 = CalibratedClassifierCV(base_estimator=clf4, cv="prefit")
clf11.fit(x_rbf_train,Y_train)
pred_train_prob3 = clf11.predict_proba(x_rbf_train)
pred_prob3 = clf11.predict_proba(x_rbf_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob3))
print("log loss of test set : ",log_loss(Y_test,pred_prob3))

print(classification_report(Y_train,pred_train4))

print(classification_report(Y_test,pred4))

rbf1 = RBFSampler(random_state=0)
x_pc_rbd_train = rbf1.fit_transform(xtrain1)
x_pc_rbd_test = rbf1.fit_transform(xtest1)
x_pc_rbd_train = p.fit_transform(x_pc_rbd_train)
x_pc_rbd_test = p.fit_transform(x_pc_rbd_test)
clf20 = SGDClassifier()
clf20.fit(x_pc_rbd_train,Y_train)

x_pc_rbd_train.shape

plot_dec_boundary_svc(clf20,x_pc_rbd_train)

xtest1.shape



Y1 = Y_train.values
Y1.shape

"""### MLP"""

from sklearn.neural_network import MLPClassifier
hl = [400,300,250,200,100]
max_iter6 = [200,250,300,350,400,500]
mlp=[]
for i in range(5):
    start = time.time()
    clf12 = MLPClassifier(random_state=0,hidden_layer_sizes=hl[i],max_iter=max_iter6[i])
    clf12.fit(X_train, Y_train)
    mlp.append(clf12)
    print(time.time()-start)

train9 = [accuracy_score(Y_train,clf2.predict(X_train)) for clf2 in mlp]
test9 = [accuracy_score(Y_test,clf2.predict(X_test)) for clf2 in mlp]
log_train9 = [log_loss(Y_train,clf2.predict_proba(X_train)) for clf2 in mlp]
log_test9 = [log_loss(Y_test,clf2.predict_proba(X_test)) for clf2 in mlp]

print(train9)
print(test9)
print(log_train9)
print(log_test9)

hl = [400,300,250,200,100]
max_iter6 = [200,250,300,350,400]
mlp2 = pd.DataFrame()
mlp2['hidden layers'] = hl
mlp2['max_iter'] = max_iter6
mlp2['train accuracy'] = train9
mlp2['test accuracy'] = test9
mlp2['train log loss'] = log_train9
mlp2['test log loss'] = log_test9
mlp2

plt.plot(hl,train9,label="trainig accuracy")
plt.plot(hl,test9,label="testing accuracy")
plt.legend()
plt.xlabel('hidden layer')
plt.ylabel('accuracy')

plt.show()

plt.plot(hl,log_train9,label="trainig logloss")
plt.plot(hl,log_test9,label="testing logloss")
plt.legend()
plt.xlabel('hidden layer')
plt.ylabel('logloss')

plt.show()

plt.plot(max_iter6,train9,label="trainig accuracy")
plt.plot(max_iter6,test9,label="testing accuracy")
plt.legend()
plt.xlabel('max iter')
plt.ylabel('accuracy')

plt.show()

plt.plot(max_iter6,log_train9,label="trainig logloss")
plt.plot(max_iter6,log_test9,label="testing logloss")
plt.legend()
plt.xlabel('max iter')
plt.ylabel('logloss')

plt.show()

from sklearn.neural_network import MLPClassifier
clf12 = MLPClassifier(random_state=0,hidden_layer_sizes=300,max_iter=250)
clf12.fit(X_train, Y_train)

pred_train3 = clf12.predict(X_train)
pred3 = clf12.predict(X_test)

print("training accuracy : ",accuracy_score(Y_train,pred_train3))
print("test accuracy : ",accuracy_score(Y_test,pred3))
from sklearn.calibration import CalibratedClassifierCV
clf13 = CalibratedClassifierCV(base_estimator=clf12, cv="prefit")
clf13.fit(X_train,Y_train)
pred_train_prob3 = clf13.predict_proba(X_train)
pred_prob3 = clf13.predict_proba(X_test)
print("log loss of train set : ",log_loss(Y_train,pred_train_prob3))
print("log loss of test set : ",log_loss(Y_test,pred_prob3))

print(classification_report(Y_train,pred_train3))

print(classification_report(Y_test,pred3))

from matplotlib.colors import ListedColormap
def plot_dec(s,clf12,xtrain1,xtest1):
    step = 0.02
    figure = plt.figure(figsize=(25,10))
    xl, xh = xtrain1[:, 0].min() - 0.5, xtrain1[:, 0].max() + 0.5
    yl, yh = xtrain1[:, 1].min() - 0.5, xtrain1[:, 1].max() + 0.5
    xc, yc = np.meshgrid(np.arange(xl, xh, step), np.arange(yl, yh, step))
    name=s
    color = ListedColormap(["#FF0000", "#0000FF"])
    
    fig,ax = plt.subplots()
    ax.scatter(xtrain1[:, 0], xtrain1[:, 1], c=Y_train, cmap=color, edgecolors="k")
    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_xticks(())
    ax.set_yticks(())
    clf12.fit(xtrain1, Y_train)
    score = clf12.score(xtest1, Y_test)
    
    if hasattr(clf12, "decision_function"):
        A = clf12.decision_function(np.c_[xc.ravel(), yc.ravel()])
    else:
        A = clf12.predict_proba(np.c_[xc.ravel(), yc.ravel()])[:, 1]
        
    A = A.reshape(xc.shape)
    cm = plt.cm.RdBu
    ax.contourf(xc, yc, A, cmap=cm, alpha=0.8)
    
    ax.scatter(
        xtrain1[:, 0], xtrain1[:, 1], c=Y_train, cmap=color, edgecolors="k"
    )
#     # Plot the testing points
    
    ax.set_xlim(xc.min(), xc.max())
    ax.set_ylim(yc.min(), yc.max())
    ax.set_xticks(())
    ax.set_yticks(())
    
    ax.set_title(name)
    plt.tight_layout()
    plt.show()

plot_dec("random forest",clf31,xtrain1,xtest1)

plot_dec("MLP",clf12,xtrain1,xtest1)

plot_dec("logistic regression",clf2,xtrain1,xtest1)

plot_dec("SVC (linear)",clf3,xtrain1,xtest1)

plot_dec("xgboost",clf1,xtrain1,xtest1)

plot_dec("decision tree",clf,xtrain1,xtest1)

plot_dec("SVC(RBF)",clf20,x_pc_rbd_train,x_pc_rbd_test)
