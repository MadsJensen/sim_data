import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
from statsmodels.tools.eval_measures import mse, rmse


# Function to fit higher order polynomial on epochs
def make_sim_data(data, baseline=None, order=10):
    """Doc string

    Parameters
    ----------
    data : epochs or TF object
    baseline : tuple
        Either tuple with the baseline of the epochs or None, if None the
        baseline from the epochs is used. Defaults to None.
    order : int
        The order of the polynomial fitted to the original epochs.


    return
    ------
    """

    if isinstance(data, mne.epochs.Epochs) or \
            isinstance(data, mne.epochs.EpochsArray):
        sim_data = _make_sim_data_epochs(data, baseline=baseline, order=order)
    elif isinstance(data, mne.time_frequency.AverageTFR):
        sim_data = _make_sim_data_tf(data, baseline, order)
    else:
        print("data missing")
        return

    return sim_data


def _make_sim_data_epochs(epochs, baseline, order):
    """
    """
    epochs.load_data()  # load data into memory.
    # create array for the simulated data.
    sim_data = np.zeros_like(epochs.get_data())

    # Find the end of baseline, to only fit polynomial on the actual data and
    # baseline.
    if baseline:
        bs_index = np.abs(epochs.times + baseline[1]).argmin()
    else:
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


def _make_sim_data_tf(data, baseline, order):
    """
    """

    # create array for the simulated data.
    sim_data = np.zeros_like(data.data)

    # Find the end of baseline, to only fit polynomial on the actual data and
    # baseline.
    if baseline:
        bs_index = np.abs(data.times + baseline[1]).argmin()
    else:
        bs_index = np.abs(data.times + 0).argmin()

    # Loop over each channel and each epoch and fit polynomial
    for k in range(data.data.shape[0]):
        for j in range(data.data.shape[1]):
            poly_params = np.poly1d(
                np.polyfit(data.times[bs_index:],
                           data.data[k, j, bs_index:], order))
            sim_data[k, j, bs_index:] = poly_params(data.times[bs_index:])

    # Create epochs object with the simulated data.
    sim_tf = mne.time_frequency.AverageTFR(
        info=data.info,
        data=sim_data,
        times=data.times,
        freqs=data.freqs,
        nave=data.nave,
        method=data.method)

    return sim_tf


def _calc_aic(loglik, k):
    """Calculate the AIC based on log likelihood and the numbers of parameters
    in the model.
    """
    aic = (-2 * np.log(loglik)) + (2 * k)
    return aic


def _calc_mse_epo(data, sim_data):
    return mse(data.get_data(), sim_data.get_data()).mean()


def _calc_mse_tf(data, sim_data):
    return mse(data.data, sim_data.data).mean()


def _calc_rmse_epo(data, sim_data):
    return rmse(data.get_data(), sim_data.get_data()).mean()


def _calc_rmse_tf(data, sim_data):
    return rmse(data.data, sim_data.data).mean()


def _calc_loglike_tf(data, sim_data):
    return -np.sum(stats.norm.logpdf(data.data, loc=sim_data.data))


def _calc_loglike_epo(data, sim_data):
    return -np.sum(stats.norm.logpdf(data.get_data(), loc=sim_data.get_data()))


def _r2_score_epo(data, sim_data):
    return r2_score(data.get_data().reshape(-1),
                    sim_data.get_data().reshape(-1))


def _r2_score_tf(data, sim_data):
    return r2_score(data.data.reshape(-1), sim_data.data.reshape(-1))


def search_for_best_order(data, orders, baseline=None, plot=False):
    """This runs make_sim_data for each order provided and calculated different
    measure of fit to see which order polynomial provides the best result.
    
    Parameters
    ----------
    data : Epochs object
        The epochs to fit.
    orders : numpy array
        Numpy array with the orders to test.
    baseline : tuple
        Either tuple with the baseline of the epochs or None, if None the
        baseline from the epochs is used. Defaults to None.
    plot : bool
        If True then plots of the measures of fits will be made. Defaults to
        False.

    return
    ------
    """
    mse_res = []
    rmse_res = []
    r2_res = []
    loglik_res = []
    aic_res = []

    for order in orders:
        print("Working on order: %d" % order)
        if isinstance(data, mne.epochs.Epochs):
            sim_data = _make_sim_data_epochs(data, baseline=baseline,
                                             order=order)
            mse_res.append(_calc_mse_epo(data, sim_data))
            rmse_res.append(_calc_rmse_epo(data, sim_data))
            loglik_res.append(_calc_loglike_epo(data, sim_data))
            aic_res.append(_calc_aic(loglik_res[-1], k=order))
            r2_res.append(_r2_score_epo(data, sim_data))
        elif isinstance(data, mne.time_frequency.AverageTFR):
            sim_data = _make_sim_data_tf(data, baseline, order)
            mse_res.append(_calc_mse_tf(data, sim_data))
            rmse_res.append(_calc_rmse_tf(data, sim_data))
            loglik_res.append(_calc_loglike_tf(data, sim_data))
            aic_res.append(_calc_aic(loglik_res[-1], k=order))
            r2_res.append(_r2_score_tf(data, sim_data))
        else:
            print("data missing")
            return

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

        # plt.figure()
        # plt.plot(orders, r2_res, 'ko')
        # plt.plot(orders[np.asarray(r2_res).argmax()],
        #          r2_res[np.asarray(r2_res).argmax()], 'ro')
        # plt.title("r^2")
        #
        # plt.figure()
        # plt.plot(orders, loglik_res, 'ko')
        # plt.plot(orders[np.asarray(loglik_res).argmin()],
        #          loglik_res[np.asarray(loglik_res).argmin()], 'ro')
        # plt.title("LogLikelihood")
        #
        # plt.figure()
        # plt.plot(orders, aic_res, 'ko')
        # plt.title("AIC")

    best_order = orders[np.asarray(mse_res).argmin()]
    # return best_order, np.asarray(mse_res), np.asarray(rmse_res), \
    #     np.asarray(r2_res), np.asarray(aic_res), np.asarray(loglik_res)
    return best_order, np.asarray(mse_res), np.asarray(rmse_res), \
           np.asarray(aic_res), np.asarray(loglik_res)