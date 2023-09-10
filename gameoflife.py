import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cellclass
from cellclass import Cell


# Initialization
N = 100
initial_states = np.random.choice([Cell.ON, Cell.OFF], N*N, p=[0.2, 0.8])
grid = np.array([Cell(state) for state in initial_states]).reshape(N, N)

# Set neighbors
for i in range(N):
    for j in range(N):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                grid[i, j].neighbors[di+1][dj+1]=grid[(i+di)%N, (j+dj)%N]

# Animation
fig, ax = plt.subplots()
img = ax.imshow(np.vectorize(lambda x: x.state)(grid), interpolation='nearest')
ani = animation.FuncAnimation(fig, cellclass.update, fargs=(img, grid, N, ), frames = 50,
                              interval=50, save_count=50)

plt.show()
