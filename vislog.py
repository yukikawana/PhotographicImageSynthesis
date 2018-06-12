import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

window = 1000
ind=int(sys.argv[1])
if(len(sys.argv)>2):
    window=int(sys.argv[2])


frame = test_df = pd.read_csv('log.csv', header=None, skiprows=16, sep=" ")
print(frame.ix[0,0])
frame.ix[:,ind].rolling(window=window).mean().plot()
plt.show()
