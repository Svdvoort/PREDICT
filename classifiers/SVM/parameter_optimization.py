from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import scipy


def random_search_parameters(features, labels, random_state, N_jobs, N_folds, test_size,
                             scoring_method, N_search_iter, C_loc, C_scale,
                             degree_loc, degree_scale, coef_loc, coef_scale,
                             gamma_loc, gamma_scale,
                             class_weight):

    param_grid = {'kernel': ['poly'],
                  'C': scipy.stats.uniform(loc=C_loc, scale=C_scale),
                  'degree': scipy.stats.uniform(loc=degree_loc, scale=degree_scale),
                  'coef0': scipy.stats.uniform(loc=coef_loc, scale=coef_scale),
                  'class_weight': [class_weight],
                  'gamma': scipy.stats.uniform(loc=gamma_loc, scale=gamma_scale)}

    clf = SVC(class_weight=class_weight, max_iter=1e7)

    print('Doing the fitting')

    cv = StratifiedShuffleSplit(n_splits=N_folds,
                                test_size=test_size,
                                random_state=random_state)

    random_search = RandomizedSearchCV(clf,
                                       param_distributions=param_grid,
                                       n_iter=N_search_iter,
                                       scoring=scoring_method,
                                       n_jobs=N_jobs,
                                       verbose=1,
                                       cv=cv,
                                       iid=False)
    random_search.fit(features, labels)

    print(random_search.best_score_)
    print(random_search.best_params_)

    return random_search
