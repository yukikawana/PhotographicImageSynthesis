import dill
import math

#d = dill.load(open("dim_300.pkl","rb"))
d = dill.load(open("dim_150.pkl","rb"))
scaledic = {}
for n in range(1,6):
    name = "conv%d_2"%n
    print(name, d[name])
    scaledic[d[name][0]] = d[name][1]
scaledic[5] = math.ceil(31./2)
#scaledic[10] = math.ceil(31./2)
dill.dump(scaledic,open("scaledic.pkl", "wb"))
for s in scaledic:
    print(s,scaledic[s])
nex = 150
for i in range(0, 7):
    print(i, nex)
    nex = math.ceil(nex/(2))
