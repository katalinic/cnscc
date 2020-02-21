"""
Created on Wed Apr 22 15:21:11 2015

@author: rkp

Code to compute spike-triggered average.
"""

from __future__ import division
import numpy as np


def compute_sta(stim, rho, num_timesteps):
    """
    Compute the spike-triggered average from a stimulus and spike-train.

    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA

    Returns:
        spike-triggered average for specified number of timesteps before spike
    """

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Compute the spike-triggered average of the spikes found.
    # To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of
    # the event (inclusive). Each of these vectors defines a list of
    # samples that is contained within a window of 300 ms before each
    # spike. The average of these vectors should be completed in an
    # element-wise manner.
    #
    # Your code goes here.
    indexer = np.repeat(spike_times, num_timesteps)
    indexer = np.reshape(indexer, (-1, num_timesteps))
    indexer -= np.arange(num_timesteps - 1, -1, -1)

    spike_windows = stim[indexer]
    del indexer

    return np.mean(spike_windows, axis=0)
