import pandas as pd
import numpy as np
import multiprocessing
from tqdm import trange, tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
#----------------------------------------------------------------------- eCDF ------------------------------------------------------------------------

class eCDF_model():
        def __init__(self, k, r, ARL0, hist_data):
            """
            Initializes the SPC module class based on CDF method
            """
            self.k = k
            self.r = r
            self.ARL0 = ARL0
            self.hist_data_sorted = np.sort(hist_data,axis=0)

        def fit(self, ic_list, H_ub, H_lb, eps1=2, eps2=0.001, max_iter=100,verbose=0):

            """
            Fitting the model, meaning to find the desired threshold based on a k value and other parameters

            Inputs
            ----------
            ic_list: Historical in-control data list
            H_ub: upper bound of h for bisection method
            H_lb: lower bound of h for bisection method
            eps1: threshold for ARL0 convergence (abs(ARL0-cal_ARL0) < eps1)
            eps2: threshold for h convergence (abs(h_ub - h_lb) < eps2)
            max_iter: maximum number of iterations for convergence

            Returns
            ----------
            H: the found threshold for the set of parameters
            ARL0i: the calculated ARL0 for the found threshold
            RL0_list: the list of run lengths for each observation
            """

            num_cores = multiprocessing.cpu_count()
            print(f"Number of CPU cores: {num_cores}")
            
            i = 0

            ARL0i = 0 # for intializing the while loop

            while abs(ARL0i - self.ARL0) > eps1 and abs(H_lb - H_ub) > eps2 and i < max_iter:
                H = 0.5 * (H_ub + H_lb)

                ARL0i, RL0_list = self.get_ARL(ic_list, H)
                
                if ARL0i < self.ARL0:
                    H_lb = H
                if ARL0i > self.ARL0:
                    H_ub = H
                
                i += 1

            self.H = H
            self.RL0_ = RL0_list

            return H, ARL0i, RL0_list
        
        def monitor(self, data):

            """
            Monitors new data points and checks whether they fall within the control limits.
            
            Inputs:
            ------------
            new_data (float): The new data point to monitor.
            
            Returns:
            ------------
            Runlength as the total number of timesteps that raises the alarm

            """

            n_dst = data.shape[1]
            max_RL = data.shape[0]
            sum_topw = np.zeros(max_RL)
            RL = 0
            detected_fault_samples = []
            n_alarms = 0
            samples_indicies = []
            sum_topr = 0
            W_plus = np.zeros([n_dst])
            W_minus = np.zeros([n_dst])
            W = np.zeros([n_dst])

            while RL < max_RL:

                for j in range(n_dst):
                    
                    mu = self.mu_calc(self.hist_data_sorted[:,j], data[RL,j])
                    w_ = np.log(1 - mu)
                    w_p = np.log(mu)

                    W_plus[j] = max(W_plus[j] - w_ - self.k, 0)
                    W_minus[j] = max(W_minus[j] - w_p - self.k, 0)
                    W[j] = max(W_plus[j], W_minus[j])
                
                sorted_list = np.sort(W[:])[::-1]

                sum_topr = np.sum(sorted_list[:self.r])
                sum_topw[RL] = sum_topr

                if sum_topr > self.H: 
                    n_alarms += 1
                    detected_fault_samples.append(RL)

                RL += 1

            return n_alarms, sum_topw[:RL-1], detected_fault_samples
            
        
        def get_ARL(self,data_list, H):

            """
        
            setting up parallel computing environment for calculating ARL
        
            Parameters
            ----------
            data : data stream for analysis
            h: constant threshold for top-r approach
            quantile_result: quantile vector

            Returns
            -------
            out : ARL and its standard deviation
        
            """
            T = len(data_list)
            num_cores = multiprocessing.cpu_count()
            
            output = Parallel(n_jobs = num_cores - 1)(delayed(self.cdf)(data_list[i], H) for i in tqdm(range(T), desc='Calculating RL0s'))
            output_list = [output[i][0] for i in range(T)]
            RL_array = np.asarray(output_list)
            ARL = np.mean(RL_array)

            print(f'Calculated ARL0 = {ARL:.2f} for H = {H:.2f} and k = {self.k}')
            
            return ARL, RL_array
            
        def cdf(self,data, H):
            
            """
            The main algorithm that implements the eCDF method

            Inputs
            ----------
            data: the observed measurements for Phase II analysis
            H: in-control threshold to raise the alarm, related to the pre-specified in-control ARL (ARL0)

            Returns
            -------
            i-1: Out-of-control run length
            sum_topw[:i-1]: The top-r statistics sum for the observations
            """

            n_dst = data.shape[1]
            n_l = data.shape[0]
            max_RL = n_l
            
            sum_topw = np.zeros(max_RL)
            RL = 0
            W_plus = np.zeros([n_dst])
            W_minus = np.zeros([n_dst])
            W = np.zeros([n_dst])
            sum_topr = 0
            
            while sum_topr <= H and RL < max_RL:

                for j in range(n_dst):
                    
                    mu = self.mu_calc(self.hist_data_sorted[:,j], data[RL,j])
                    w_ = np.log(1 - mu)
                    w_p = np.log(mu)

                    
                    W_plus[j] = max(W_plus[j] - w_ - self.k, 0)
                    W_minus[j] = max(W_minus[j] - w_p - self.k, 0)
                    W[j] = max(W_plus[j], W_minus[j])


                sorted_list = np.sort(W[:])[::-1]
                sum_topr = np.sum(sorted_list[:self.r])
                sum_topw[RL] = sum_topr
                
                RL += 1
        
            return RL - 1, sum_topw[:RL - 1]
            
        def mu_calc(self, xh_j_sorted, x_j):
            """
            Bayesian estimation of the cumulative distribution function (CDF) 

            Inputs
            ----------
                xh_j_sorted: sorted historical in-control data of datastream x_j
                x_j: new sample for datastream x_j
            Returns
            ----------
                mu: Bayesian estimation of the cdf
            """
            if np.ndim(xh_j_sorted) == 1:
                mu = (np.searchsorted(xh_j_sorted, x_j) + 1)/(len(xh_j_sorted)+2)
                return mu
            else:
                # number of historical samples
                n = xh_j_sorted.shape[0]
                cdf = np.zeros(n)
                for i in range(n):
                    cdf[i] = np.searchsorted(xh_j_sorted[i], x_j[i])
                
                # Eqn 3
                mu = (cdf+1)/(n+2) 
                return mu
        
        def plot_histogram(self,array, bins=10):
            """
            Plots a histogram for a given NumPy array.
            
            Parameters:
            array (numpy.ndarray): The input array for the histogram.
            bins (int): Number of bins for the histogram (default: 10).
            """
            plt.hist(array, bins=bins, edgecolor='black')
            plt.title("Histogram of Calculated RL0")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

        def set_h(self,H):
            """
            only being used when the user itself wants to specify the threshold (manually setting H)
            """
            self.H = H


