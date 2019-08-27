import numpy as np
import scipy.stats as st


def compute_confidence(metric, N_train, N_test, alpha=0.95):
    """
    Function to calculate the adjusted confidence interval

    metric: numpy array containing the result for a metric for the different cross validations
    (e.g. If 20 cross-validations are performed it is a list of length 20 with the calculated accuracy for
    each cross validation)
    N_train: Integer, number of training samples
    N_test: Integer, number of test_samples
    alpha: float ranging from 0 to 1 to calculate the alpha*100% CI, default 95%
    """

    N_iterations = len(metric)

    metric_average = np.mean(metric)
    S_uj = 1.0 / (N_iterations - 1) * np.sum((metric_average - metric)**2.0)

    metric_std = np.sqrt((1.0/N_iterations + N_test/N_train)*S_uj)

    CI = st.t.interval(alpha, N_iterations-1, loc=metric_average, scale=metric_std)

    # print("Metric " + str(alpha*100) + " % CI:" + str(CI))

    return CI
