import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# On charge les donnees
df = pd.read_csv('cwndbis.csv', delimiter = ',')
x = []
y = []
for i in range(0,len(df)):
    x.append(df['time'].values[i])
    y.append(df['cwnd'].values[i])
y = np.array(y)
X = np.array(x)

plt.plot(X,y)
plt.show()
