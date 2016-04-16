from sklearn import svm

def train(train_data,train_label,k):
    X = train_data
    y = train_label
    clf = svm.SVC(kernel=k)
    clf.fit(X, y)
    return clf

def test(test_data,clf):
    res = clf.predict(test_data)
    return res
