import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))


# viz code 시각화 하는 코드
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# 오류 발생시
# not pound ~
#   => pip(conda) install ~
# GraphViz\'s executables not found
#   => 환경변수에서 path에 c\..\Anaconda3\Library\bin\graphviz를 넣어준다.

print(test_data[1], test_target[1])
print(iris.feature_names, iris.target_names)
