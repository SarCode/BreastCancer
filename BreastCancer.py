# Support Vector Machine (SVM)

# Importing the libraries

import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Cancer.csv')

#Replacing ? value with mode of column
x=dataset['Bare Nuclei'].mode()
x
dataset.replace({"?":1}, inplace=True)


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred)
x=cm1.sum()
y=cm1[0][0]+cm1[1][1]
accuracy=y/x
print("SVM Accuracy :",accuracy*100)
print("SVM : predicted benign tumor but actually have malignant tumor : 3")

#####################################################################
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)
x=cm2.sum()
y=cm2[0][0]+cm2[1][1]
accuracy=y/x
print("Random Forrest Accuracy :",accuracy*100)
print("Random Forrest : predicted benign tumor but actually have malignant tumor : 2")
