import itertools
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import preprocessing
%matplotlib inline

#Load Dataset
url = "https://raw.githubusercontent.com/ParthanOlikkal/KNearestNeighbor/master/teleCust1000t.csv"
df = pd.read_csv(url)
df.head()

#Data Visualisation
#number of data in cuscat  
df['cuscat'].value_counts()
df.hist(column = 'income', bins = 50)

#Feature Set
df.columns

#Converting pandas data frame to numpy arrays
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retired', 'gender', 'reside']] . values    #.astype(float)
X[0:5]

#Labels 
y = df['custcat'].values
y[0:5]

#Normalize Data
X = preprocessing.StandardScalar().fit(X).transform(X.astype(float))
X[0:5]

#Train/Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

#Classification - KNN
from sklearn.neighbors import KNeighborsClassifier

#Training with k=4
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train)
neigh

#Prediction
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy evaluation - Jaccard_similarity_score function
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat)))


#How to know the right value of K : repeat the process from 1 to 10 and plot a graph
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(1, Ks):

	#Train Model and Predict
	neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
	yhat =neigh.predict(X_test)
	mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)
	std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#Plot model accuracy
plt.plot(range(1,Ks), mean_acc,'g')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha = 0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.xlabel('Number of neighbor (K)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()

print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
