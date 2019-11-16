import numpy as np
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="log", alpha=0.1, max_iter=100, shuffle=True, fit_intercept=True)

classes = []
def strtonum(x):
    if x not in classes:
        classes.append(x)
    return classes.index(x)

data = np.loadtxt("./iris.data", delimiter=',', converters={4: strtonum})

x = data[:, :4]
y = data[:, 4]

clf.fit(x,y)
print(clf.score(x,y))