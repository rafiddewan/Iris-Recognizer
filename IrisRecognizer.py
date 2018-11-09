import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.LinearClassifier(n_classes = 3)
classifier.fit(iris.data, iris.target)
score = metrics.acurracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy %f" % score)
