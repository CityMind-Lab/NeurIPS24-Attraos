import numpy as np
import time
import pandas as pd
import random

def CCMethod(data, max_d):

    N = len(data)
    S_mean = np.zeros([1, max_d]) #
    Sdelta_mean = np.zeros([1, max_d])
    Scor = np.zeros([1, max_d])
    sigma = np.std(data) #
    r_division = list(range(1,5))
    m_list = list(range(2,6))
    for t in range(1, max_d+1):
        S = np.zeros([len(m_list), len(r_division)]) # (4,4)
        S_delta = np.zeros([1, len(m_list)]) # (1,4)
        for m in m_list:
            for r_d in r_division:
                r = sigma/2 * r_d
                sub_data = subdivide(data, t) #
                s = 0
                Cs1 = np.zeros([t]) #
                Cs = np.zeros([t])
                for tau in range(1, t+1): # 1,..,t
                    N_t = int(N/t)
                    Y = sub_data[:, tau-1]
                    for i in range(N_t-1):
                        for j in range(i, N_t):
                            d1 = np.abs(Y[i] -Y[j])
                            if r>d1:
                                # Cs1[tau] += 1
                                Cs1[tau-1] += 1
                    Cs1[tau-1] = 2*Cs1[tau-1] / (N_t*(N_t-1))
                    Z = reconstruction(Y,m,1) # (m, N_t-(m-1))
                    M = N_t-(m-1)
                    Cs[tau-1] = correlation_integral(Z, M, r)
                    s += Cs[tau-1] - Cs1[tau-1]**m
                S[m-2, r_d-1] = s/tau
            S_delta[0,m-2] = max(S[m-2,:])-min(S[m-2,:])
        S_mean[0, t-1] = np.mean(np.mean(S))
        Sdelta_mean[0, t-1]=np.mean(S_delta)
        Scor[0, t-1] = np.abs(S_mean[0, t-1]+Sdelta_mean[0, t-1])
    return Sdelta_mean, Scor

def subdivide(data, t): # t=t+1

    n = len(data)
    Data = np.zeros([int(n/t), t])
    for i in range(t):
        assert t != 0
        for j in range(int(n/t)):
            Data[j, i] = data[i+ j*t]
    return Data

def reconstruction(data, m, tau):

    n = len(data)
    M = n - (m-1) * tau
    Data = np.zeros([m, M])
    for j in range(M):
        for i in range(m):
            Data[i, j] = data[i*tau+j]
    return Data

def correlation_integral(Z, M , r):

    sum_H = 0
    for i in range(M-1):
        for j in range(i+1, M):
            d = np.linalg.norm(Z[:, i]-Z[:, j],ord=np.inf)
            if r>d:
                sum_H += 1
    C_I = 2*sum_H/(M*(M-1))
    return C_I
def get_tau(Sdelta_mean):
    Sdelta_mean = Sdelta_mean.reshape(-1)
    for i in range(1, len(Sdelta_mean)-1):
        if Sdelta_mean[i] < Sdelta_mean[i-1] and Sdelta_mean[i]<Sdelta_mean[i+1]:
            tau = i
            return tau+1
        else:
            continue
def get_tw(Scor):
    Scor = Scor.reshape(-1)
    return np.argmin(Scor)+1

def get_embedding_m(tau, t_w):
    assert tau != 0
    return int(t_w / tau) + 1


if __name__ == "__main__":
    random.seed(2021)
    length = 720
    name = 'ETTm2'

    if name == 'ETTh1':
        stock_data = pd.read_csv('dataset/ETT-small/ETTh1.csv')
        vars = ['LULL', 'LUFL', 'HUFL', 'OT']
    elif name == 'ETTh2':
        stock_data = pd.read_csv('dataset/ETT-small/ETTh2.csv')
        vars = ['LULL', 'MUFL', 'LUFL', 'HUFL', 'OT']
    elif name == 'ETTm1':
        stock_data = pd.read_csv('dataset/ETT-small/ETTm1.csv')
        vars = ['LULL', 'LUFL', 'HUFL', 'OT']
    elif name == 'ETTm2':
        stock_data = pd.read_csv('dataset/ETT-small/ETTm2.csv')
        vars = ['OT', 'MUFL', 'LUFL', 'HUFL']
    elif name == 'Weather':
        stock_data = pd.read_csv('dataset/weather/weather.csv')
        vars = ['wd (deg)', 'OT']
    elif name == 'Exchange_rate':
        stock_data = pd.read_csv('dataset/exchange_rate/exchange_rate.csv')
        vars = ['0', '1', '2', '3', '4', '5', '6', 'OT']
    elif name == 'Traffic':
        stock_data = pd.read_csv('dataset/traffic/traffic.csv')
        vars = ['223', '481', '665', '806', 'OT']
    elif name == 'ECL':
        stock_data = pd.read_csv('dataset/electricity/electricity.csv')
        vars = ['4', '23', '193', '123', 'OT']
    elif name == 'ILL':
        stock_data = pd.read_csv('dataset/illness/national_illness.csv')
        vars = ['AGE 0-4', 'AGE 5-24', 'OT']

    total_length = len(stock_data)
    start_index = random.randint(0, total_length - length)

    for var in vars:
        data = stock_data.iloc[start_index: start_index + length, :][var].to_numpy()
        max_d = 32
        start = time.time()
        Sdelta_mean, Scor = CCMethod(data, max_d)
        print(f"cost time:{time.time() - start},")
        tau = get_tau(Sdelta_mean)
        t_w = get_tw(Scor)
        m = get_embedding_m(tau, 1)
        print(f"tau:{tau},tw:{t_w},m:{m}")