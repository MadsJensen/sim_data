import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
from statsmodels.tools.eval_measures import mse, rmse


# Function to fit higher order polynomial on epochs
def make_sim_data(epochs, baseline=None, order=10):
    """Doc string

    params:
    -------
    epochs : Epoch object
    baseline : tuble
        Either tuble with the baseline of the epochs or None, if None the
        baseline from the epochs is used. Defualts to None.
    order : int
        The order of the polynomial fitted to the orignal epochs.


    returns:
    -------
    """
    epochs.load_data()  # load data into memory.
    # create array for the simulated data.
    sim_data = np.zeros_like(epochs.get_data())

    # Find the end of baseline, to only fit polynomial on the actual data and
    # baseline.
    bs_index = np.abs(epochs.times + 0).argmin()

    # Loop over each channel and each epoch and fit polynomial
    for k in range(epochs.get_data().shape[0]):
        for j in range(epochs.get_data().shape[1]):
            poly_params = np.poly1d(
                np.polyfit(epochs.times[bs_index:],
                           epochs.get_data()[k, j, bs_index:], order))
            sim_data[k, j, bs_index:] = poly_params(epochs.times[bs_index:])

    # Create epochs object with the simulated data.
    sim_epochs = mne.EpochsArray(
        sim_data,
        epochs.info,
        tmin=epochs.tmin,
        events=epochs.events,
        event_id=epochs.event_id)

    return sim_epochs


plt.style.use("seaborn")


def AIC(LL, k):
    aic = (-2 * np.log(LL)) + (2 * k)
    return aic


def search_for_best_order(epochs, orders, plot=False):
    """
    Keyword Arguments:
    epochs : Epochs object
        The epochs to fit.
    orders : numpy array
        Numpy array with the orders to test.
    """

    mse_res = []
    rmse_res = []
    r2_res = []
    loglik_res = []
    aic_res = []

    for order in orders:
        print("Working on order: %d" % order)
        sim_data = make_sim_data(epochs, order=order)
        mse_res.append(mse(epochs.get_data(), sim_data.get_data()).mean())
        rmse_res.append(rmse(epochs.get_data(), sim_data.get_data()).mean())
        r2_res.append(
            r2_score(epochs.get_data().reshape(-1),
                     sim_data.get_data().reshape(-1)))
        LL = -np.sum(
            stats.norm.logpdf(epochs.get_data(), loc=sim_data.get_data()))
        loglik_res.append(LL)
        aic_res.append(AIC(LL, k=order + 1))

        best_order = orders[np.asarray(mse_res).argmin() + 1]

    if plot:
        plt.figure()
        plt.plot(orders, rmse_res, 'ko')
        plt.plot(orders[np.asarray(rmse_res).argmin()],
                 rmse_res[np.asarray(rmse_res).argmin()], 'ro')
        plt.title("RMSE")

        plt.figure()
        plt.plot(orders, mse_res, 'ko')
        plt.plot(orders[np.asarray(mse_res).argmin()],
                 mse_res[np.asarray(mse_res).argmin()], 'ro')
        plt.title("MSE")

        plt.figure()
        plt.plot(orders, r2_res, 'ko')
        plt.plot(orders[np.asarray(r2_res).argmax()],
                 r2_res[np.asarray(r2_res).argmax()], 'ro')
        plt.title("r^2")

        plt.figure()
        plt.plot(orders, loglik_res, 'ko')
        plt.plot(orders[np.asarray(loglik_res).argmin()],
                 loglik_res[np.asarray(loglik_res).argmin()], 'ro')
        plt.title("LogLikelihood")

        plt.figure()
        plt.plot(orders, aic_res, 'ko')
        plt.title("AIC")

    return best_order, mse_res, rmse_res, r2_res, aic_res, loglik_res