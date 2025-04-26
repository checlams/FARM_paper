import numpy as np 
import pandas as pd  
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from random import sample
import pickle
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from scipy.signal import find_peaks
from numpy import trapz
from scipy.signal import find_peaks
from tqdm import tqdm
import pywt
from sklearn.decomposition import PCA
#=================================================================================================#
def load_ic_data(n_samples=None):
    '''
    Loads the required data from the dataset directory

    Returns:
    list: List of in-control (np.ndarray) data
    '''
    load_path = './dataset'
        
    with open(f'{load_path}/MATLAB GUI/ic_list.pkl', 'rb') as file:
        ic_list= pickle.load(file)
    if n_samples is not None:
        ic_list = ic_list[:n_samples]

    print(f'a set of {len(ic_list)} in-control simulations are loaded!')
    return ic_list



def prepare_train(datalist, test_size=0.2):

    ic_train, ic_test = train_test_split(datalist, test_size=test_size, random_state=42)
    print(f'Number of train simulations: {len(ic_train)}')
    means = list()
    stds = list()

    X_train = list()

    for i, x in enumerate(ic_train):
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        X_train.append(x_scaled)
        means.append(scaler.mean_)
        stds.append(scaler.scale_)
    
    print('ic-train Normalized!')
    
    means_concat = np.vstack(means)
    train_means = np.mean(means_concat, axis=0)
    
    std_concat = np.vstack(stds)
    train_stds = np.mean(std_concat, axis=0)

    return X_train, ic_test, train_means, train_stds

def load_oc(fault=None):

    '''
    Loads the required data from the dataset directory

    Returns:
    list: List of in-control (np.ndarray) data
    '''
    load_path = './dataset'

    n_sim = 500

    with open(f'{load_path}/MATLAB GUI/faulty_data.pkl', 'rb') as file:
        oc_list_all = pickle.load(file)
    if fault is not None:
        oc_list = oc_list_all[(fault-1)*n_sim:(fault)*n_sim]
        print(f'a set of {len(oc_list)} out-of-control simulations for fault {fault} is loaded!')

    else:
        oc_list = oc_list_all
        print(f'a set of {len(oc_list)} out-of-control simulations for all faults is loaded!')

    return oc_list

def logarithm(cov):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.log(d))
    logcov = np.dot(np.dot(V, D), V.T)
    return logcov

def sqrroot(cov):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.sqrt(d))
    sqrroot = np.dot(np.dot(V, D), V.T)
    return sqrroot

def expstep(cov,step):
    d, V = np.linalg.eigh(cov)
    D = np.diag(np.exp(d*step))
    expstep = np.dot(np.dot(V, D), V.T)
    return expstep

def mat_op(operation,d,V):
    return np.dot(V*operation(d),V.T)

def T_Project(geomean_inv_sqrt,cov):
        newmat = geomean_inv_sqrt.dot(cov).dot(geomean_inv_sqrt)
        T_cov = logarithm(newmat)
        return T_cov

def reimannian_mapping(data_list):

    '''
    data_list: list of timeseries datapoints
    '''
    # this function only works when given a list of timeseries and return the corresponding mapped covariance matrix

    Covs = []
    for i in range(len(data_list)):
        baseline = data_list[i]
        Cov = ledoit_wolf(baseline)[0]
        Covs.append(Cov)

    #Initialize the Matrix Mean:
    Covs = np.array(Covs)
    geomean = np.mean(Covs,axis = 0)
    #Initialize the gradient descent step size and loss:
    step = 1
    norm_old = np.inf

    #Set tolerance:
    tol = 1e-8
    norms = []

    for n in range(100):

        #Compute the gradient
        geo_eval,geo_evec = np.linalg.eigh(geomean)
        geomean_inv_sqrt = mat_op(np.sqrt,1. / geo_eval,geo_evec)

        #Project matrices to tangent space and compute mean and norm:
        mats= [geomean_inv_sqrt.dot(cov).dot(geomean_inv_sqrt) for cov in Covs]
        log_mats = [logarithm(mat) for mat in mats]
        meanlog = np.mean(log_mats,axis = 0)
        norm = np.linalg.norm(meanlog)

        #Take step along identified geodesic to minimize loss:
        geomean_sqrt = sqrroot(geomean)
        geomean = geomean_sqrt.dot(expstep(meanlog,step)).dot(geomean_sqrt)

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm

        elif norm > norm_old:
            step = step / 2.
            norm = norm_old

        if tol is not None and norm / geomean.size < tol:
            break
            
        norms.append(norm)

    geo_eval,geo_evec = np.linalg.eigh(geomean)

    geomean_inv_sqrt = mat_op(np.sqrt,1. / geo_eval, geo_evec)

    T_covs = [T_Project(geomean_inv_sqrt,cov) for cov in Covs]

    print(f'{len(T_covs)} covariances are mapped')

    return T_covs, geomean_inv_sqrt

def covariance_calc(data_list):
    '''
    data_list: list of timeseries datapoints
    '''
    Covs = []
    for i in range(len(data_list)):
        baseline = data_list[i]
        Cov = ledoit_wolf(baseline)[0]
        Covs.append(Cov)
    return Covs

def plot_conf(conf_matrix, figure_size=(15, 15), label_fontsize=18, number_fontsize=14, decimal_places=2):
    # Display the confusion matrix with a larger figure size
    fig, ax = plt.subplots(figsize=figure_size)  # Set figure size
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues', ax=ax)  # Pass the colormap name as a string

    # Modify labels to start from 1
    disp.ax_.set_xticks(range(len(conf_matrix)))  # Set the number of ticks
    disp.ax_.set_yticks(range(len(conf_matrix)))  # Set the number of ticks
    disp.ax_.set_xticklabels(range(1, len(conf_matrix)+1), fontsize=24)
    disp.ax_.set_yticklabels(range(1, len(conf_matrix)+1), fontsize=24)
    
    # Customize font sizes for axes labels
    ax.set_xlabel("Predicted Label", fontsize=label_fontsize)
    ax.set_ylabel("True Label", fontsize=label_fontsize)
    
    # Customize the font size and format of the text in the matrix
    for text in disp.text_.ravel():  # Access text elements in the matrix
        value = float(text.get_text())  # Get the current value
        if value == 0 or value < 0.005:
            formatted_value = "0"
        else:
            formatted_value = f"{value:.{decimal_places}f}"  # Format to specified decimal places
        text.set_text(formatted_value)  # Update the text
        text.set_fontsize(number_fontsize)  # Set font size
    
    # Customize the font size of the colormap ticks
    colorbar = disp.im_.colorbar  # Access the colorbar
    colorbar.ax.tick_params(labelsize=14)  # Adjust font size for colorbar ticks
    
    # Adjust layout for better display
    plt.tight_layout()

def SVM_fault_diag(X, y, patience_time, RL_list,ic_mean, ic_std, t_shift,windowing=False, call_all=False, ic_append=False, 
                    include_stats=False, scaling=False, compression=False, mapping=True, tuning=True):
    
    oc_list_trunc = []
    print(f'Patience time is set to {patience_time}')
    faults = [i for i in range(1, 21)]
    for fault in faults:
        oc_fault = X[(fault-1)*500:(fault)*500]
        RL_fault = RL_list[(fault-1)*500:(fault)*500]
        for i, oc in enumerate(oc_fault):
            oc_scaled = (oc - ic_mean) / ic_std
            time_length = RL_fault[i] + patience_time
            oc_list_trunc.append(oc_scaled[:time_length,:])

    if len(oc_list_trunc) != 10000:
        raise ValueError('Length of oc_list_trunc is not 10000')
    
    if ic_append:
        ic_oc_list = []
        ic_list = load_ic_data(n_samples=len(oc_list_trunc))

        for i, oc in enumerate(oc_list_trunc):
            ic_part = (ic_list[i][:t_shift,:] - ic_mean)/ ic_std

            # Concatenate ic_part and oc_part along the appropriate axis
            oc = np.concatenate((ic_part, oc), axis=0)

    if call_all:
        mapped_faulty, geo_mean = reimannian_mapping(X)
    else:
        if windowing:
            num_rows = [arr.shape[0] for arr in oc_list_trunc]

            window_size = round(np.mean(num_rows))

            print(f'Window size = {window_size}')

            # Prepare the data
            XX = []
            for i, oc in enumerate(oc_list_trunc):
                if oc.shape[0] < window_size:
                    XX.append(oc)
                else:
                    XX.append(oc[-window_size:,:])

            if mapping:
                print('Mapping the windowed data to the tangent space')
                mapped_faulty, geo_mean = reimannian_mapping(XX)
            else:
                print('No mapping. Returning windowed data')

                mapped_faulty = covariance_calc(XX)
        else:
            print('No windowing. Mapping truncated timeseries covariances')
            mapped_faulty, geo_mean = reimannian_mapping(oc_list_trunc)



    flat_mapped_faulty = [np.ndarray.flatten(cov) for cov in mapped_faulty]
    flat_mapped_faulty = np.array(flat_mapped_faulty)

    print(f'The whole data has shape: {flat_mapped_faulty.shape}')
    
    if scaling:
        print('Scaling the data')
        scaler = StandardScaler()
        flat_mapped_faulty = scaler.fit_transform(flat_mapped_faulty)
        print('Data has been scaled')


    if include_stats:
        df_covs = pd.DataFrame(flat_mapped_faulty)

        num_columns = len(flat_mapped_faulty[0])  # Number of columns based on the first row of data
        df_covs = pd.DataFrame(flat_mapped_faulty, columns=[f'feature_{i}' for i in range(num_columns)])
        print(f'Dataframe with covariances has been created with shape: {df_covs.shape}')
        
        means = []
        stds = []
        medians = []
        variances = []
        ranges = []
        maxes = []
        areas = []
        n_peaks = []

        print('Computing stats metrics for each fault')
        for fault in tqdm(faults):
            fault_stats = safire_dict[f'global state fault {fault}']

            RL_list_fault = RL_dict[f'fault {fault}']

            trunc_stats = []
            for i,stat_plot in enumerate(fault_stats):
                time_to_trunc = RL_list_fault[i] + patience_time
                trunc_stats.append(stat_plot[:time_to_trunc])


            for stat_plot in trunc_stats:
                # compute metric
                mean = np.mean(stat_plot)
                means.append(mean)

                std = np.std(stat_plot)
                stds.append(std)

                median = np.median(stat_plot)
                medians.append(median)

                variance = np.var(stat_plot)
                variances.append(variance)

                range_ = np.max(stat_plot) - np.min(stat_plot)
                ranges.append(range_)

                max_ = np.max(stat_plot)
                maxes.append(max_)

                area = trapz(stat_plot)
                areas.append(area)

                peaks, _ = find_peaks(stat_plot)
                n_peak = len(peaks)
                n_peaks.append(n_peak)

        # Create a DataFrame from the lists
        data = {
            'means': means,
            'stds': stds,
            'medians': medians,
            'variances': variances,
            'ranges': ranges,
            'maxes': maxes,
            'areas': areas,
            'n_peaks': n_peaks,
        }
        df_stats = pd.DataFrame(data)

        # Create a StandardScaler object
        scaler = StandardScaler()


        columns_to_scale = ['means', 'stds', 'medians', 'variances', 'ranges', 'maxes', 'areas']

        df_stats_scaled = pd.DataFrame(scaler.fit_transform(df_stats[columns_to_scale]), columns=columns_to_scale)

        # Concatenate the scaled features with the 'n_peaks' column
        df_stats_scaled['n_peaks'] = df_stats['n_peaks']

        train_data = pd.concat([df_covs, df_stats_scaled], axis=1)
        train_data.columns = train_data.columns.astype(str)
        print(f'Dataframe with stats metrics has been created with shape: {train_data.shape}')
    else:
        train_data = flat_mapped_faulty
    
    
    # Training the model
    X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, random_state=42, stratify=y)

    # Define the parameter grid
    param_grid = {
        'C': [1, 10, 100],
        'kernel': ['rbf'],
        'gamma': ['auto', 'scale', 0.001, 0.01, 0.1, 1]
    }
    
    if compression:
        # Create a PCA object
        pca = PCA(n_components=0.95)
        print('Performing PCA with 95% variance explained')
        print(f'number of features after PCA: {pca.n_components_}')
        
        # Fit the PCA model to the training data
        pca.fit(X_train)

        # Transform the training and testing data using the PCA model
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    
    if tuning:
        # Perform grid search
        strat_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=strat_cv, n_jobs=-1, verbose=2, scoring='accuracy')
        
        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Print the best parameters and best score
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Score: ", grid_search.best_score_)
        print("Best Estimator: ", grid_search.best_estimator_)

        best_estimator = grid_search.best_estimator_
    
    else:
        best_estimator = SVC(C=10, gamma=0.001, kernel='rbf')
        # Calculate the train accuracy
        best_estimator.fit(X_train, y_train)

    accuracy_train = best_estimator.score(X_train, y_train)
    # accuracy_train = svc_model.score(X_train, y_train)
    print("Train Accuracy: ", accuracy_train)

    # Test the model
    accuracy_test = best_estimator.score(X_test, y_test)
    print(f'Test Accuracy: {accuracy_test}')

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, best_estimator.predict(X_test), normalize='true')

    return best_estimator, cm, accuracy_train, accuracy_test

     
def isSPD(mat):
    # Check if the matrix is symmetric positive definite
    return np.all(np.linalg.eigvals(mat) > 0)

