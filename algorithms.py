from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt


def evaluate_results(y_pred, y_test, estim):
    acc = metrics.accuracy_score(y_test, y_pred)*100
    f1 = metrics.f1_score(y_test, y_pred)*100
    c = metrics.confusion_matrix(y_test,y_pred)
    #cm = metrics.confusion_matrix(y_test, y_pred, labels=estim.classes_)
    fig = plt.figure()
    plt.matshow(c)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('conf_matrix/confusion_matrix_{}.jpg'.format(estim))

    return [acc,f1]
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
    print("Best score for decision tree: {}".format(rs.best_score_*100))

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
    cdt = dt.fit(X_train,y_train)
    y_predic = cdt.predict(X_test)
    [acc,f1]= evaluate_results(y_test, y_predic, dt)
    print("Accuracy achieved applying Decision Tree: ", acc )
    print("F1 score applying Decision Tree: ", f1 )
    return [y_predic, y_test]
def params_4_svm(data):
    print('SELECTING PARAMETERS FOR SVM \n ')
    features = list(data.columns)
    features.remove('class')
    scaler = StandardScaler()
    X = data[features]
    #scaler.fit(X)
    y = data['class']
    #X_transf = scaler.transform(X)
    #y_tranf = scaler.transform(y)
    #'precomputed'
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degree = [ i for i in range(3,5)]
    gamma = ['scale','auto']
    c = [i for i in np.arange(0.1,10,0.5)]
    params = {'C': c, 'kernel' : kernels, 'degree': degree, 'gamma' : gamma }
    svm = SVC(max_iter=1000)
    rs = RandomizedSearchCV(svm, params, n_iter=25, n_jobs=-1)
    rs.fit(X,y)
    print("Best parameters for SVM: ", rs.best_params_)
    print("Best score for SVM: {}".format(rs.best_score_*100))

    return rs.best_params_

def apply_svm(params, train_data, test_data):
    print('\n APPLYING SVM \n ')
    features = list(train_data.columns)
    features.remove('class')
    X_train = train_data[features]
    y_train = train_data['class']
    X_test = test_data[features]
    y_test = test_data['class']
    svm = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'], gamma=params['gamma'], max_iter=1000)
    csvm = svm.fit(X_train,y_train)
    y_predic = csvm.predict(X_test)
    [acc,f1] = evaluate_results(y_test, y_predic, svm)
    print("Accuracy achieved applying SVM: ", acc )
    print("F1 score applying SVM: ", f1 )
    return [y_predic, y_test]

def params_4_random_forest(data):
    print('SELECTING PARAMETERS FOR RANDOM FOREST \n ')
    features = list(data.columns)
    features.remove('class')
    #scaler = StandardScaler()
    X = data[features]
    y = data['class']
    max_feat = len(features)
    rf = RandomForestClassifier()
    #Select best parameters for the decision tree
    #param = {'n_estimators': range(100,200),'max_depth': range(3,15), 'max_features': range(1,max_feat), 'criterion':['gini','entropy']}
    param = {'n_estimators': range(100,200), 'criterion':['gini','entropy']}
    rs = RandomizedSearchCV(rf, param, n_iter=30,n_jobs=-1)
    rs.fit(X,y)

    print("Best parameters for random forest: ", rs.best_params_)
    print("Best score for random forest: {}".format(rs.best_score_*100))

    return rs.best_params_

def apply_random_forest(params, train_data, test_data):
    print('\n APPLYING RANDOM FOREST \n ')
    features = list(train_data.columns)
    features.remove('class')
    X_train = train_data[features]
    y_train = train_data['class']
    X_test = test_data[features]
    y_test = test_data['class']
    rf = RandomForestClassifier(n_estimators= params['n_estimators'], criterion=params['criterion'], n_jobs=-1)
    crf = rf.fit(X_train,y_train)
    y_predic = crf.predict(X_test)
    [acc,f1] = evaluate_results(y_test, y_predic, rf)
    print("Accuracy achieved applying Random Forest: ", acc)
    print("F1 score applying Random Forest: ", f1 )

    return [y_predic, y_test]

def params_4_neural_network(data):
    print('SELECTING PARAMETERS FOR MLP CLASSIFIER \n ')
    features = list(data.columns)
    features.remove('class')
    X = data[features]
    y = data['class']
    num_feat = len(features)
    nnc = MLPClassifier()
    batch_size = [64, 128, 256, 512]
    epochs = [120, 140, 160]
    hidden_layer_sizes = [(3*num_feat, 2*num_feat,), (4*num_feat, 3*num_feat, 2*num_feat,), (6*num_feat,5*num_feat,4*num_feat, 3*num_feat, 2*num_feat,)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    learning_rate = ['constant', 'invscaling', 'adaptive']
    solver = ['lbfgs', 'sgd', 'adam']
    momentum = [0.7, 0.8, 0.9]
    params = {'hidden_layer_sizes':hidden_layer_sizes, 'activation': activation, 'solver': solver, 'learning_rate':learning_rate,
                'batch_size':batch_size,'momentum' :momentum, 'max_iter':epochs}
    rs = RandomizedSearchCV(nnc,params,n_iter=70,n_jobs=-1)
    rs.fit(X,y)

    
    print("Best parameters for MLPClassifier: ", rs.best_params_)
    print("Best score for MLPClassifier: {}".format(rs.best_score_*100))

    return rs.best_params_
def apply_neural_network(params, train_data, test_data):
    print('\n APPLYING MLPCLASSIFIER (NEURAL NETWORK) \n ')
    features = list(train_data.columns)
    features.remove('class')
    X_train = train_data[features]
    y_train = train_data['class']
    X_test = test_data[features]
    y_test = test_data['class']
    #{'solver': 'adam', 'momentum': 0.8, 'max_iter': 120, 'learning_rate': 'constant', 'hidden_layer_sizes': 86, 'batch_size': 256, 'activation': 'relu'}
    nnc = MLPClassifier(solver= params['solver'], momentum=params['momentum'], max_iter=params['max_iter'],learning_rate=params['learning_rate'],
                hidden_layer_sizes = params['hidden_layer_sizes'],batch_size=params['batch_size'], activation=params['activation'])
    cnn = nnc.fit(X_train,y_train)
    y_predic = cnn.predict(X_test)
    [acc,f1] = evaluate_results(y_test, y_predic, nnc)
    print("Accuracy achieved applying MLP Classifier: ", acc)
    print("F1 score applying Random Forest: ", f1 )
    return [y_predic, y_test]
    