import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import pandas as pd
import csv #not needed unless you use ReadCsv()
import time

#TODO: 2nd data set
#Learning Curves
#2 hyperparameters to tune
#Cross-validation: scores = cross_val_score(clf, X, y, cv=5)
#Show validation curves

DEFAULT_TEST_SIZE = 0.2


def Wallclock(dataset=1):
    neu_model = MLPClassifier(random_state=1,hidden_layer_sizes = (50,), activation='logistic' )
    dt_model = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.005, max_depth=13)
    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='rbf'))
    boost_model = AdaBoostClassifier(n_estimators=120,learning_rate=0.9)
    neigh_models = KNeighborsClassifier(n_neighbors=8, weights='distance')
    
    models = [neu_model,dt_model,svm_model,boost_model,neigh_models]
    model_names = ["Neural_Nets_CV","DT_CV", "SVM_CV","Boosting_CV","KNN_CV"]

    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    
    i=0
    train_sizes = [0.1, 0.2, 0.3, 0.4, .5, 0.6, 0.7, 0.8, 0.9]
    for model in models:
        print("\n model: ",model_names[i])
        start = time.time()
        model.fit(X_train,y_train)
        end = time.time()
        print("train clock time in ms: ", (end-start)*1000)

        
        start = time.time()
        model.predict(X_test)
        end = time.time()
        print("test clock time in ms : ", (end-start)*1000)

        train_score = 100*model.score(X_train, y_train)
        test_score = 100*model.score(X_test, y_test)
        print ("train score", train_score)
        print("test score", test_score)
        i+=1
        

def All_Algos_Learning_Curves_CV(dataset=1):
    neu_model = MLPClassifier(random_state=1)
    dt_model = tree.DecisionTreeClassifier(random_state=0)
    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    boost_model = AdaBoostClassifier(n_estimators=100)
    neigh_models = KNeighborsClassifier(n_neighbors=3)
    
    models = [neu_model,dt_model,svm_model,boost_model,neigh_models]
    model_names = ["Neural_Nets_CV","DT_CV", "SVM_CV","Boosting_CV","KNN_CV"]

    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    
    i=0
    train_sizes = [0.1, 0.2, 0.3, 0.4, .5, 0.6, 0.7, 0.8, 0.9]
    for model in models:
        train_size_abs, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes)

        # learning curve reported in percentages
        lrn_crv = [[train_size*100, np.mean(train_score)*100, np.mean(test_score)*100] for train_size, train_score, test_score in zip(train_sizes, train_scores, test_scores)]
        count=0
        print("\n model: ",model_names[i])
        for line in lrn_crv:
            if count >= 10:
                break
            print("training size, train score, test score",line[0], line[1], line[2])
            count += 1
            
        plot_learning_curve_errors(lrn_crv, model_names[i], dataset)
        i += 1
    

    


def Neural_Net(dataset=1):
    neu_model = MLPClassifier(random_state=1)
    #plot learning curve
    lrn_crv = run_algo(neu_model,dataset)
    plot_learning_curve(lrn_crv, "Neural_Network",dataset)
    plot_learning_curve_errors(lrn_crv,"Neural_Network",dataset)


    # tune hyperparameter
    hidden_layers = []
    models = []

    for h in range (10,200,10):
        model = MLPClassifier(random_state=1, hidden_layer_sizes=(h,))
        models.append(model)
        hidden_layers.append(h)

    plot_hyperparmeter_performance(models, "hidden layer sizes", np.array(hidden_layers), "Neural_Nets",dataset)


    # tune hyperparameter activation
    activations = ['identity', 'logistic', 'tanh', 'relu']
    models = []

    for a in activations:
        model = MLPClassifier(random_state=1, activation=a)
        models.append(model)
  

    plot_hyperparmeter_performance(models, "activation functions", np.array(activations), "Neural_Nets",dataset)
        


def DT(dataset=1):

    #plot hyp
    max_depths = []
    models = []

    for depth in range (1,21,1):
        model = tree.DecisionTreeClassifier(random_state=0, max_depth = depth)
        models.append(model)
        max_depths.append(depth)

    plot_hyperparmeter_performance(models, "max depths", np.array(max_depths), "Decision_Tree",dataset)

    
    


def SVM(dataset=1):
    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    #plot learning curve 
    lrn_crv = run_algo(svm_model,dataset)
    plot_learning_curve(lrn_crv, "SVM",dataset)
    plot_learning_curve_errors(lrn_crv,"SVM",dataset)

    #tune hyper param kernel

    #kernels_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    kernels_list = ['rbf','sigmoid','linear', 'poly']
    models = []
    for k in kernels_list:
        model = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel=k))
        models.append(model)
        

    plot_hyperparmeter_performance(models, "kernels", np.array(kernels_list), "SVM",dataset)

    #tune cache size
    cache_sizes = []
    models = []
    for size in range(150,260,10):
        model = make_pipeline(StandardScaler(), SVC(gamma='auto',cache_size=size ))
        models.append(model)
        cache_sizes.append(size)
        

    plot_hyperparmeter_performance(models, "cache_size", np.array(cache_sizes), "SVM",dataset)

    
    #tune degree
    degrees = []
    models = []
    for d in range(1,15,1):
        model = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='poly',degree=d))
        models.append(model)
        degrees.append(d)
        

    plot_hyperparmeter_performance(models, "degree-for-poly", np.array(degrees), "SVM",dataset)

    
    
    
    



def Boosting(dataset=1):
    boost_model = AdaBoostClassifier(n_estimators=100)
    #plot learning curve for an optimal k. Why is this k optimal?
    lrn_crv = run_algo(boost_model,dataset)
    plot_learning_curve(lrn_crv, "Boosting",dataset)
    plot_learning_curve_errors(lrn_crv,"Boosting",dataset)

    #tune hyper param n_estimators

    n_estimators_list = []
    models = []
    for n in range(1,200,1):
        model = AdaBoostClassifier(n_estimators=n)
        models.append(model)
        n_estimators_list.append(n)

    plot_hyperparmeter_performance(models, "n_estimators", np.array(n_estimators_list),"Boosting",dataset)


    #tune hyper param learning rate

    learn_rates = []
    models = []
    for r_percent in range(10,110,10):
        rate = float(r_percent)/100
        model = AdaBoostClassifier(learning_rate = rate)
        models.append(model)
        learn_rates.append(rate)

    plot_hyperparmeter_performance(models, "learning_rate", np.array(learn_rates), "Boosting",dataset)
    
    



def Knn(dataset=1):

    neigh = KNeighborsClassifier(n_neighbors=3)

    #plot learning curve for an optimal k. Why is this k optimal?
    lrn_crv = run_algo(neigh,dataset)
    plot_learning_curve(lrn_crv, "KNN",dataset)
    plot_learning_curve_errors(lrn_crv, "KNN",dataset)

    #tune hyper param k

    neighs = []
    ks = []
    for k in range(1,100,1):
        neigh = KNeighborsClassifier(n_neighbors=k)
        neighs.append(neigh)
        ks.append(k)

    plot_hyperparmeter_performance(neighs, "k", np.array(ks),"KNN",dataset)

    #tune weights on distance
    neighs = []
    wts = ['uniform','distance']
    for wt in wts:
        neigh = KNeighborsClassifier(weights = wt)
        neighs.append(neigh)        

    plot_hyperparmeter_performance(neighs, "weight_on_distance", np.array(wts),"KNN",dataset)
        


def plot_hyperparmeter_performance(models, hyp_name, hyps, algo_name="",dataset=1):
    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for model in models:
        model.fit(X_train,y_train)
    
    # Accuracy vs hyperparameter for training and testing sets
    train_scores = [model.score(X_train, y_train) for model in models]
    test_scores = [model.score(X_test, y_test) for model in models]

    fig, ax = plt.subplots()
    ax.set_xlabel(hyp_name)
    ax.set_ylabel("accuracy")
    title = algo_name + " : Accuracy vs " + hyp_name + " for Dataset - " + str(dataset)
    ax.set_title(title)
    ax.plot(hyps, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(hyps, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    file_name = algo_name +"_accurracy_vs_"+hyp_name+ "_Dataset_" + str(dataset)
    plt.savefig(file_name)
    #plt.show()

        

    


def run_algo(model,dataset=1):
    X, y, dataset_name = GetData(dataset)
    test_percs = []
    lrn_crv = []
    
    scale = 4
    for i in range(9*scale, 1, -1):
        test_percs.append(float(i)/(10*scale))
        
    for test_perc in test_percs:
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_perc, random_state=0)
        train_perc = 100*round(1-test_perc,4)
        
        print("-----\nTraining percentage = ", train_perc)
        print("\n")
        
        
        model.fit(X_train, y_train)
        train_score = 100*model.score(X_train, y_train)
        test_score = 100*model.score(X_test, y_test)

        print("Train score: ",train_score)
        print("Test score: ",test_score)

        lrn_crv.append([train_perc,train_score,test_score])
        
    return lrn_crv


def plot_learning_curve(lrn_crv,algo_name,dataset=1):
    crv = np.array(lrn_crv)
    x_data = crv[:,0] #train_percentages
    y_data = crv[:,1:] # train and test scores
    figure_name = algo_name + "Dataset-" + str(dataset) + ".jpg"
    
    fig, ax = plt.subplots()
    title = algo_name + " : Learning Curve - "+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Training Data Percentage (%) --> ")
    plt.ylabel("Score (%) --> ")
    plt.plot(x_data, y_data)
    plt.legend(['train score', 'test score'])
    plt.savefig(figure_name)
    #plt.show()

def plot_learning_curve_errors(lrn_crv,algo_name,dataset=1):
    crv = np.array(lrn_crv)
    x_data = crv[:,0] #train_percentages
    y_data = np.ones_like(crv[:,1:])*100 - crv[:,1:] # train and test error rate
    figure_name = algo_name + "Dataset-" + str(dataset) + "_error.jpg"
    
    fig, ax = plt.subplots()
    title = algo_name + " : Learning Curve showing Error Rates - " + "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Training Data Percentage (%) --> ")
    plt.ylabel("Error Rate (%) --> ")
    plt.plot(x_data, y_data)
    plt.legend(['train', 'cross-validation'])
    plt.savefig(figure_name)
    #plt.show()


def plot_data(X,y,name="data"):
    fig, ax = plt.subplots()
    title = name
    plt.title(title)
    plt.xlabel("Xs --> ")
    plt.ylabel("ys --> ")
    for i in range(21):
        plt.plot(X[:,i], y,linestyle='None',marker='o')        
        figname=str(i)
        plt.legend(figname)
        plt.savefig(figname)
    #plt.show()
    
    
    


    
    
# Decision Tree Pruning Citation:
# https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

def DecisionTree(dataset=1):
    #X, y = load_breast_cancer(return_X_y=True)
    X, y, dataset_name = GetData(dataset)
    test_percs = []
    lrn_crv = []
    
    scale = 4
    for i in range(9*scale, 1, -1):
        test_percs.append(float(i)/(10*scale))
        
    for test_perc in test_percs:
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_perc, random_state=0)
        train_perc = 100*round(1-test_perc,4)
        
        print("-----\nTraining percentage = ", train_perc)
        print("\n")
        
        clf = tree.DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        train_score = 100*clf.score(X_train, y_train)
        test_score = 100*clf.score(X_test, y_test)

        print("Train score: ",train_score)
        print("Test score: ",test_score)

        lrn_crv.append([train_perc,train_score,test_score])

        """
        if train_perc > 85:
            tree.plot_tree(clf)
        """

        if train_perc > 89 and train_perc < 91:
            #Prune Tree
            print("Pruning for train_perc% = ",train_perc)
            path = clf.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            fig, ax = plt.subplots()
            ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
            ax.set_xlabel("effective alpha")
            ax.set_ylabel("total impurity of leaves")
            ax.set_title("Total Impurity vs effective alpha for training set")
            plt.savefig("impurity of leaves vs alpha")

            #train a decision tree using the effective alphas
            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, y_train)
                clfs.append(clf)
            print(
                "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                    clfs[-1].tree_.node_count, ccp_alphas[-1]
                )
            )

            # number of nodes and tree depth decreases as alpha increases
            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]

            node_counts = [clf.tree_.node_count for clf in clfs]
            depth = [clf.tree_.max_depth for clf in clfs]
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[0].set_title("Number of nodes vs alpha")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            ax[1].set_title("Depth vs alpha")
            fig.tight_layout()
            plt.savefig("nodes, tree depth vs alpha")

            # Accuracy vs alpha for training and testing sets
            train_scores = [clf.score(X_train, y_train) for clf in clfs]
            test_scores = [clf.score(X_test, y_test) for clf in clfs]

            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            title = "Accuracy vs alpha for training and testing sets" + " for Dataset-" + str(dataset)
            ax.set_title(title)
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            fig_name = "Decision_Tree_accurracy_vs_alpha" + "_Dataset_"+str(dataset)
            plt.savefig(fig_name)
            #plt.show()
            
            

    # TODO: Explore one more hyperparameter: e.g. depth of tree
    # TODO: Second dataset run
    # TODO: Plot tree with ccp_alpha optimized
    """
    crv = np.array(lrn_crv)
    x_data = crv[:,0] #train_percentages
    y_data = crv[:,1:] # train and test scores
    figure_name = "Decision_Tree.jpg"
    
    fig, ax = plt.subplots()
    plt.title("Decision Tree : Learning Curve")
    plt.xlabel("Training Data Percentage (%) --> ")
    plt.ylabel("Score (%) --> ")
    plt.plot(x_data, y_data)
    plt.legend(['train score', 'test score'])
    plt.savefig(figure_name)
    plt.show()
    """
    
    plot_learning_curve(lrn_crv,"Decision_Tree",dataset)
    plot_learning_curve_errors(lrn_crv,"Decision_Tree",dataset)
    
    """
    ax.set_title("Decision Tree : Learning Curve")
    ax.set_xlabel("Training Data Percentage (%) --> ")
    ax.set_ylabel("Score (%) --> ")
    ax.plot(x_data, y_data)
    ax.legend(['train score', 'test score'])
    plt.savefig(figure_name)
    plt.show()
    """
    
    



def GetData(dataset = 1):
    """
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    """
    if dataset == 1:
        name = "fetal_health"
    else:
        name = "mobile_price"
    name = name + ".csv"
    df = pd.read_csv(name)
    data = df.to_numpy() # rows and columns just like .csv
    X = data[:,0:-1]
    y = np.transpose(data)[-1]
    #print("data",data)
    #print("X",X)
    #print("y",y)
    return (X,y,name)





def ReadCsv():    
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    



    

    


if __name__=="__main__":
    print("One day you'll look back on this and smile. \
    There will be tears, but they will be tears of joy")
    #DecisionTree(2)
    #DT(2) #for hyper
    #Knn(2)
    #Boosting(2)
    #SVM(2)
    #Neural_Net(2)
    #DecisionTree(1)
    #DT(1) #for hyper
    #Knn(1)
    #Boosting(1)
    #SVM(1)
    #Neural_Net(1)
    #All_Algos_Learning_Curves_CV(1)
    #All_Algos_Learning_Curves_CV(2)
    Wallclock(1)
    Wallclock(2)
    #ReadCsv()
    #X,y,name=GetData()
    #plot_data(X,y,"data1")
