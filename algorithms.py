from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.svm import SVC 
import numpy as np

def params_4_dec_tree(data):
    print('SELECTING PARAMETERS FOR DECISION TREE \n')
    features = list(data.columns)
    features.remove('class')
    X = data[features]
    y = data['class']
    dt = tree.DecisionTreeClassifier()
    #Select best parameters for the decision tree
    max_feat = len(features)
    param = {'max_depth': range(3,15), 'max_features': range(1,max_feat), 'criterion':['gini','entropy']}
    rs = RandomizedSearchCV(dt, param, n_iter=25,n_jobs=-1)
    rs.fit(X,y)

    print("Best parameters for decision tree: ", rs.best_params_)
    print("Best score for decision tree: ", rs.best_score_)

    return rs.best_params_

def apply_decision_tree(params, train_data, test_data):
    #Params is a dict with three pairs (key,value): 'max_depth', 'max_features' and 'criterion'
    #Returns the values predicted by the algorithm and the actual labels
    print('\n APPLYING DECISION TREE \n')
    features = list(train_data.columns)
    features.remove('class')
    X_train = train_data[features]
    y_train = train_data['class']
    X_test = test_data[features]
    y_test = test_data['class']
    dt = tree.DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'], max_features=params['max_features'])
    clf = dt.fit(X_train,y_train)

    y_predic = clf.predict(X_test)

    return [y_predic, y_test]
def params_4_svm(data):
    print('SELECTING PARAMETERS FOR SVM \n ')
    features = list(data.columns)
    features.remove('class')
    X = data[features]
    y = data['class']
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    degree = [ i for i in range(3,5)]
    gamma = ['scale','auto']
    c = [i for i in np.arange(0.1,10,0.5)]
    params = {'C': c, 'kernel' : kernels, 'degree': degree, 'gamma' : gamma }
    svm = SVC()
    rs = RandomizedSearchCV(svm, params, n_iter=25, n_jobs=5)
    rs.fit(X,y)
    print("Best parameters for SVM: ", rs.best_params_)
    print("Best score for SVM: ", rs.best_score_)

    return rs.best_params_

def apply_svm(params, train_data, test_data):
    print('\n APPLYING SVM \n ')
    features = list(train_data.columns)
    features.remove('class')
    X_train = train_data[features]
    y_train = train_data['class']
    X_test = test_data[features]
    y_test = test_data['class']
    svm = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'], gamma=params['gamma'])
    clf = svm.fit(X_train,y_train)
    y_predic = clf.predict(X_test)

    return [y_predic, y_test]