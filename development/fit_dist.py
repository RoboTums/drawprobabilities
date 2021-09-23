import pandas as pd
import scipy.stats as sp
from sympy import divisors
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter

def point_table_to_hist(point_df,num_bins):
    heights = []
    points_per_bin = point_df.shape[0]//num_bins
    for bin_i in range(points_per_bin,point_df.shape[0],points_per_bin):
        last_i = max(0,bin_i - points_per_bin)
        height = point_df.iloc[last_i:bin_i,:].Y.mean()
        heights.append(height)
    return np.array(heights), np.linspace(point_df.X.min(),point_df.X.max(),num_bins)
    

if __name__ == '__main__':
    point_df = pd.read_csv('points.csv',index_col=0)
    # you have to renomalize this or you wont get anywhere near a PDF
    point_df['Y'] = point_df['Y']/point_df['Y'].sum()
    #best bin divisor is the largest divisor. if prime then we drop the last row and do the same.
    if point_df.shape[0] % 2 == 0 and point_df.shape[0] >= 4:
        num_bins = divisors(point_df.shape[0])[-2] 
    else:
        point_df.drop(point_df.shape[0]-1,inplace=True)
        num_bins = divisors(point_df.shape[0])[-2] 
    #we make a scipy stats rv because its the easiest to sample from
    hist = point_table_to_hist(point_df,num_bins=num_bins)
    hist_dist = sp.rv_histogram(hist)
    #we sample 1k variates to see which ones are the best. 
    samples = hist_dist.rvs(size=1000)
    #@TODO: let yourself choose how many samples you want AND which distributions to test. 
    f = Fitter(samples,bins=10)
    f.fit()
    score_df = f.summary()
    print(score_df)
    score_df.to_csv('result_dist.csv')
    plt.show()
    exit(0)
    