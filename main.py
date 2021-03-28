# General packages
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets as ds
from scipy.stats import randint
from zipfile import ZipFile
from ecg.load_data import load_data

# Classifiers and kernels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score

# Regularization
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.feature_selection import SelectFromModel

# Model selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#Classifiers
from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection 
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm


#Learning curve
def plot_learning_curve(estimator, title, X, y, axes, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores  = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt


    # Data loading functions

def preprocessing_data(data, n_features=10):
    data_points= data.drop(['label'], axis=1).to_numpy()
    data_labels= data['label'].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(data_points, data_labels, test_size=0.3)

    #Scaling
    scaler = preprocessing.RobustScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    #PCA
    pca = PCA(n_components=n_features)
    pca = pca.fit(x_train_scaled)
    x_train_trans = pca.transform(x_train_scaled)
    x_test_trans = pca.transform(x_test_scaled)
    explained_variance = pca.explained_variance_ratio_

    # x_train_trans_df = pd.DataFrame(x_train_trans, columns=['PC1', 'PC2', 'PC3', 'PC4'])
    # y_train_df = pd.DataFrame(y_train, columns=['label'])
    # xy_plot = pd.concat([x_train_trans_df, y_train_df], axis=1)
    # pc_pairplot = sns.pairplot(xy_plot, hue='label', palette='tab10')
    
    return x_train_trans, y_train, x_test_trans, explained_variance, data_points, data_labels, x_train


if __name__ == "__main__":
    _, data = load_data()

    x_train_trans, y_train, _, explained_variance, data_points, data_labels, x_train = preprocessing_data(data)
    # plot learning curves
    X = x_train_trans
    Y = y_train
    print(explained_variance)
    

    # Aantal features 
    svc = svm.SVC(kernel="linear")
    rfecv = feature_selection.RFECV(estimator=svc, step=1, cv=model_selection.StratifiedKFold(4),scoring='roc_auc')
    rfecv.fit(x_train, y_train)
    # Plot number of features VS. cross-validation scores

    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()




    # clsfs = [LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),GaussianNB(), 
    #          LogisticRegression(),SGDClassifier(),KNeighborsClassifier(), DecisionTreeClassifier()]
    # num = 0
    # fig = plt.figure(figsize=(24,8*len(clsfs)))
    # ax = fig.add_subplot(7, 2, num+1)
    # ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y, 
    #            s=25, edgecolor='k', cmap=plt.cm.Paired)

    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # num = 1
    # for clf in clsfs:
    #     # Split data in training and testing
    #     title = str(type(clf))
    #     ax = fig.add_subplot(7, 3, num + 1)
    #     plot_learning_curve(clf, title, X, Y, ax, ylim=(0.3, 1.01), cv=cv)
    #     num += 1
    # plt.show()