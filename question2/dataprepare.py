import numpy as np

classes = []
def strtonum(x):
    if x not in classes:
        classes.append(x)
    return classes.index(x)


data = np.loadtxt("./iris.data", delimiter=',', converters={4: strtonum})
np.savez("iris", data=data[:, :4], label=np.array([data[:, 4]]).T)
