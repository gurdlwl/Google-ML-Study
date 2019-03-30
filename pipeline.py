from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

# cross_validation 이 오류가 난다면 model_selection을 사용하자
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, Y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))
 
# playground.tensorflow.org
# 신경망관련 웹사이트. 참고
