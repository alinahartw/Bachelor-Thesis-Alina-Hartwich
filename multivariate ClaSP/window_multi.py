from numba import prange
import numpy as np

from claspy.window_size import suss

def suss_for_each_dim(time_series):
    """
    Calculates the suss window size for each dimensions of the time series.

    Parameter
    --------
    time_series : Time series for which the window size is to be calculated.

    Returns
    -------
    window_pro_dim : np.array(shape=(time_series.shape[0])) the suss window size is stored for each dimension .
    """
    window_pro_dim = np.zeros(shape=(time_series.shape[0]), dtype=np.int64)  

    #if TS is one dimensional, return suss window_size calculation
    if  time_series.shape[0] == 1:
        window_pro_dim[0] =suss(time_series[0])//2
        return window_pro_dim
    
    #if TS is multidimenaional, calculate suss per dimension 
    for dim in prange(time_series.shape[0]):
        ts = time_series[dim]
        window_pro_dim[dim] = suss(ts)//2
    return window_pro_dim 

def max_dim(time_series):
    """
    Calculates the suss window size for each dimensions of the time series. 
    Afterwards uses the maximal window size of all calculated ones.

    Parameter
    --------
    time_series : Time series for which the window size is to be calculated

    Returns
    -------
    window_pro_dim : np.array(shape=(time_series.shape[0])) the maximal suss window size of all dimensions
    """
    window_pro_dim = np.zeros(shape=(time_series.shape[0]), dtype=np.int64)  

    #calculate suss per dimension 
    for dim in prange(time_series.shape[0]):  
        ts = time_series[dim]
        window_pro_dim[dim] = suss(ts)//2

    #use maximal suss value for all dimensions
    window_max = int(np.max(window_pro_dim))
    for i in range(0, len(window_pro_dim)):
        window_pro_dim[i] = window_max

    return window_pro_dim

def min_dim(time_series):
    """
    Calculates the suss window size for each dimensions of the time series. 
    Afterwards use the minimal window size of all calculated ones.

    Parameter
    --------
    time_series : Time series for which the window size is to be calculated

    Returns
    -------
    window_pro_dim : np.array(shape=(time_series.shape[0])) the minimal suss window size is stored for each dimension 
    """
    window_pro_dim = np.zeros(shape=(time_series.shape[0]), dtype=np.int64)  

    #calculate suss per dimension 
    for dim in prange(time_series.shape[0]):  
        ts = time_series[dim]
        window_pro_dim[dim] = suss(ts)//2

    #use minimal suss value for all dimensions
    window_min = int(np.min(window_pro_dim))
    for i in range(0, len(window_pro_dim)):
        window_pro_dim[i] = window_min
    
    return window_pro_dim

def median_dim(time_series):
    """
    Calculates the suss window size for each dimensions of the time series. 
    Afterwards use the median window size of all calculated ones.

    Parameter
    --------
    time_series : Time series for which the window size is to be calculated

    Returns
    -------
    window_pro_dim : np.array(shape=(time_series.shape[0])) the median of all suss window sizes is stored for each dimension 
    """
    window_pro_dim = np.zeros(shape=(time_series.shape[0]), dtype=np.int64)  

    #calculate suss per dimension 
    for dim in prange(time_series.shape[0]):  
        ts = time_series[dim]
        window_pro_dim[dim] = suss(ts)//2

    #use median of all suss values for all dimensions
    window_median = int(np.median(window_pro_dim))
    for i in range(0, len(window_pro_dim)):
        window_pro_dim[i] = window_median
    
    return window_pro_dim

def arithme_dim(time_series):
    """
    Calculates the suss window size for each dimensions of the time series. 
    Afterwards use the arithmetic mean window size of all calculated ones.

    Parameter
    --------
    time_series : Time series for which the window size is to be calculated.

    Returns
    -------
    window_pro_dim : np.array(shape=(time_series.shape[0])) the arithmetic mean of all suss window sizes is stored for each dimension 
    """
    window_pro_dim = np.zeros(shape=(time_series.shape[0]), dtype=np.int64)  

    #calculate suss per dimension 
    for dim in prange(time_series.shape[0]):  
        ts = time_series[dim]
        window_pro_dim[dim] = suss(ts)//2

    #use arithmetic mean of suss values for all dimensions
    window_arithmetisches_mittel = np.sum(window_pro_dim)//time_series.shape[0]
    for i in range(0, len(window_pro_dim)):
        window_pro_dim[i] = window_arithmetisches_mittel
    
    return window_pro_dim


_WINDOW_SIZE_MAPPING = {
    "suss": suss_for_each_dim,
    "max" : max_dim,
    "min" : min_dim,
    "median" : median_dim,
    "arith" : arithme_dim
}

def map_window_size_methods_multi(window_size_method):
    """
    Maps the given window size method to its corresponding implementation.

    Parameters
    ----------
    window_size_method : str
        The name of the window size method to be mapped.

    Returns
    -------
    function
        The implementation of the given window size method.

    Raises
    ------
    ValueError
        If the given window size method is not a valid window size method.
        Valid implementations include: 'suss', 'fft', and 'acf'.

    """
    if window_size_method not in _WINDOW_SIZE_MAPPING:
        raise ValueError(
            f"{window_size_method} is not a valid window size method. Implementations include: {', '.join(_WINDOW_SIZE_MAPPING.keys())}")

    return _WINDOW_SIZE_MAPPING[window_size_method]



