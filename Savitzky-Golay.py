import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


df = pd.read_excel(r'data.xlsx', header=0)
data = df.values.astype(float)
print(df)

smoothed_data = savgol_filter(data[:,1:], window_length=25, polyorder=2, axis=0, deriv=0) # Savitzky-Golay 滤波器
print(smoothed_data.shape)
np.savetxt(r'光谱曲线.csv', smoothed_data,delimiter=',',fmt='%.8f')
