import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

ind=int( sys.argv[1])


frame = test_df = pd.read_csv('log.csv', header=None, skiprows=16, sep=" ")
print(frame.ix[0,0])
frame.ix[:,ind].plot()
plt.show()
