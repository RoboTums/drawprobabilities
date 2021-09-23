from fitter import Fitter
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


if __name__ == '__main__':
	point_df = pd.read_csv('points.csv',index_col=0)
	f = Fitter(point_df)
	f.fit()
	f.summary()
