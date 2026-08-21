import numpy as np

time = [0, 1, 1.5, 2.5, 3,5]
t_diffs = np.diff(time)

print(time)
print(t_diffs)

weights = np.zeros(len(time))
weights[-1] = 0.5*(t_diffs[-1]+1)
weights[0] = 0.5*(t_diffs[0]+1)
for i in range(len(weights)-2):
    weights[i+1] = 0.5*(t_diffs[i] + t_diffs[(i+1)])

print(weights)