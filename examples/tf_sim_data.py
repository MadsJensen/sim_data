"""
This example is based on the MNE-python example:
http://martinos.org/mne/stable/auto_examples/time_frequency/plot_time_frequency_simulated.html

@author: mje
@email mads [] cfin.au.dk
"""
import mne
import numpy as np
from mne import create_info, EpochsArray
from mne.datasets import somato
from mne.time_frequency import (tfr_multitaper)

import sim_data as sd  # Might need some tweaking to work!

sfreq = 1000.0
ch_names = ['SIM0001', 'SIM0002']
ch_types = ['grad', 'grad']
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = int(sfreq)  # 1 second long epochs
n_epochs = 40
seed = 42
rng = np.random.RandomState(seed)
noise = rng.randn(n_epochs, len(ch_names), n_times)

# Add a 50 Hz sinusoidal burst to the noise and ramp it.
t = np.arange(n_times, dtype=np.float) / sfreq
signal = np.sin(np.pi * 2. * 50. * t)  # 50 Hz sinusoid signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.  # Hard windowing
on_time = np.logical_and(t >= 0.45, t <= 0.55)
signal[on_time] *= np.hanning(on_time.sum())  # Ramping
data = noise + signal

reject = dict(grad=4000)
events = np.empty((n_epochs, 3), dtype=int)
first_event_sample = 100
event_id = dict(sin50hz=1)
for k in range(n_epochs):
    events[k, :] = first_event_sample + k * n_times, 0, event_id['sin50hz']

epochs = EpochsArray(data=data, info=info, events=events, event_id=event_id,
                     reject=reject)

freqs = np.arange(5., 100., 3.)

# You can trade time resolution or frequency resolution or both
# in order to get a reduction in variance

# (1) Least smoothing (most variance/background fluctuations).
n_cycles = freqs / 2.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=-1., vmax=3.,
           title='Sim: Least smoothing, most variance')

# (2) Less frequency smoothing, more time smoothing.
n_cycles = freqs  # Increase time-window length to 1 second.
time_bandwidth = 4.0  # Same frequency-smoothing as (1) 3 tapers.
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=-1., vmax=3.,
           title='Sim: Less frequency smoothing, more time smoothing')

# Use real example data
data_path = somato.data_path()
raw_fname = data_path + '/MEG/somato/sef_raw_sss.fif'

# let's explore some frequency bands
iter_freqs = [
    ('Theta', 4, 7),
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
    ("Gamma", 30, 45)
]

# set epoching parameters
event_id, tmin, tmax = 1, -0.5, 0.5
baseline = None

# get the header to extract events
raw = mne.io.read_raw_fif(raw_fname, preload=False)
events = mne.find_events(raw, stim_channel='STI 014')

frequency_map = list()

for band, fmin, fmax in iter_freqs:
    # (re)load the data to save memory
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.pick_types(meg='grad', eog=True)  # we just look at gradiometers

    # bandpass filter and compute Hilbert
    raw.filter(fmin, fmax, n_jobs=1,  # use more jobs to speed up.
               l_trans_bandwidth=1,  # make sure filter params are the same
               h_trans_bandwidth=1,  # in each band and skip "auto" option.
               fir_design='firwin')
    raw.apply_hilbert(n_jobs=1, envelope=False)

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=baseline,
                        reject=dict(grad=4000e-13, eog=350e-6), preload=True)
    # remove evoked response and get analytic signal (envelope)
    epochs.subtract_evoked()  # for this we need to construct new epochs.
    epochs = mne.EpochsArray(
        data=np.abs(epochs.get_data()), info=epochs.info, tmin=epochs.tmin)
    # now average and move on
    frequency_map.append(((band, fmin, fmax), epochs))

beta_data = frequency_map[2][1]
beta_sim = sd.make_sim_data(beta_data, order=17)
alpha_data = frequency_map[1][1]
alpha_sim = sd.make_sim_data(alpha_data, order=17)
theta_data = frequency_map[0][1]
theta_sim = sd.make_sim_data(theta_data, order=17)
gamma_data = frequency_map[3][1]
gamma_sim = sd.make_sim_data(theta_data, order=17)

beta_sim.average().plot_joint()
beta_data.average().plot_joint()
beta_diff = mne.combine_evoked([beta_data.average(), -beta_sim.average()],
                               weights="equal")
beta_diff.plot_joint()

gamma_sim.average().plot_joint()
gamma_data.average().plot_joint()
gamma_diff = mne.combine_evoked([gamma_data.average(), -gamma_sim.average()],
                                weights="equal")
gamma_diff.plot_joint()

theta_sim.average().plot_joint()
theta_data.average().plot_joint()
theta_diff = mne.combine_evoked([theta_data.average(), -theta_sim.average()],
                                weights="equal")
theta_diff.plot_joint()

alpha_sim.average().plot_joint()
alpha_data.average().plot_joint()
alpha_diff = mne.combine_evoked([alpha_data.average(), -alpha_sim.average()],
                                weights="equal")
alpha_diff.plot_joint()
