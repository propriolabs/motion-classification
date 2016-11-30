import numpy as np
import statsmodels.api as sm
import _ucrdtw

def featurize(ts,dat,templates):
    """
    Get the features from the raw tennis data
    :param ts: numpy array of size (samples,sensor readings) which contains the raw data
    :param dat: tuple with start and stop indices to slice data
    :return: 
    :rtype : 
    """
    #start and stop indices 
    s=dat[0]
    t=dat[1]
    #slices the time series 
    data=ts[s:t,:]
    length=1.0*len(data[:,0])
    #this handles the edge case in which the swings is at the very end of the collection
    
    #computes total acceleration
    a=np.sqrt(np.sum(np.array(data**2,dtype=float),axis=1))
    #finds the index which corresponds to the maximum acceleration
    max_index=8
    model1x=sm.OLS(range(1,9),ts[s:s+max_index,0])
    model1y=sm.OLS(range(1,9),ts[s:s+max_index,1]) 
    model1z=sm.OLS(range(1,9),ts[s:s+max_index,2]) 
    reg1=np.array([model1x.fit().params[0],model1y.fit().params[0],model1z.fit().params[0]])
    model2x=sm.OLS(range(1,9),ts[s+max_index:t,0])
    model2y=sm.OLS(range(1,9),ts[s+max_index:t,1]) 
    model2z=sm.OLS(range(1,9),ts[s+max_index:t,2]) 
    reg2=np.array([model2x.fit().params[0],model2y.fit().params[0],model2z.fit().params[0]])
    dist=np.zeros((len(templates)))
    for j,template in enumerate(templates):
        dist[j]=_ucrdtw.ucrdtw(data, template, 0.5, False)[1]
    #cross correlation between each sensor reading
    correlation=np.corrcoef(np.c_[data,a],rowvar=0)
    return np.hstack([
    np.max(data,axis=0),\
    np.min(data,axis=0),np.mean(a),\
    np.std(a),\
    np.min(a),np.max(a),
    reg1,reg2,#reg3,reg4,
    np.sign(ts[8,0:2]),
    (np.argmax(dat,axis=0)-np.argmin(dat,axis=0))/length,
    np.array([correlation[0,1],correlation[0,2],correlation[1,2],correlation[0,3],correlation[1,3],correlation[2,3]]),
    dist,
    np.sum(np.diff(np.sign(data),axis=0)>0,axis=0)/length,\
    np.sum(np.diff(np.sign(data),axis=0)<0,axis=0)/length])



def rolling_window(data, window):
    """
    Creates a rolling window 
    data=np.array([[1,2,3],[4,5,6],[7,8,9]])
    window=2
    rolling_window(data,window) returns [[[1,2,3],[4,5,6]],[[4,5,6],[7,8,9]]
    :param data: numpy array (n sensors x n observations)
    :return: numpy array (n sensors x window size x observations / window size)
    :rtype : numpy array
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)  

def featurize_sliding(data,window_size):
    """
    Get the rolling statistical features
    :param data: numpy array (n sensors x n observations) raw data
    :param window_size: int size of the window
    :return: features for each window
    :rtype : numpy array
    """
    tmp=rolling_window(data,window_size)
    rolling_feats=np.apply_along_axis(featurize,2,tmp)
    return rolling_feats

def autocorrelation(data):
    """
    Get the autocorrelation
    :param data: numpy array / pandas series which contains 1-D time series
    :return: the autocorrelation of the time series 
    :rtype : 1-D numpy array
    """
    N = len(data)
    fvi = np.fft.fft(data, n=2*N)
    acf = np.real( np.fft.ifft(fvi * np.conjugate(fvi) )[:N])
    acf = acf/N
    return acf[:N+1]

def wavelet_regression(data):
    """
    Get the wavelet coefficients (which fully describe data.) using the multilevel discrete wavelet 
    transform. Thresholding sets many of the coefficients to zero by assuming that they are noise.  
    The original signal is recovered by applying the inverse discrete wavelet transform to the 
    thresholded coefficients. 
    :param data: 1-D numpy array which contains raw data
    :return: thresholded wavelet reconstruction of data
    :rtype : numpy 1-D array 
    """
    data=np.array(data)
    true_coefs = pywt.wavedec(data, 'db8', level=11, mode='per')
    noisy_coefs = pywt.wavedec(data, 'db8', level=11, mode='per')

    sigma = stand_mad(noisy_coefs[-1])
    uthresh = sigma*np.sqrt(2*np.log(len(data)))

    denoised = noisy_coefs[:]

    denoised[1:] = (pywt.thresholding.soft(i, value=uthresh) for i in denoised[1:])
    return pywt.waverec(denoised, 'db8', mode='per')
