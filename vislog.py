import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

window = 1000
fl=sys.argv[1]
ind=int(sys.argv[2])
if(len(sys.argv)>3):
    window=int(sys.argv[3])


frame = test_df = pd.read_csv(fl, header=None, skiprows=17, sep=" ")
print(frame.ix[0,0])
frame.ix[:,ind].rolling(window=window).mean().plot()
plt.show()
