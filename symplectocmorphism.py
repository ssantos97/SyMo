from Code.data import generate_trajectory
from Code.models import pendulum
from Code.root_find import rootfind
from Code.models import get_field
import matplotlib.pyplot as plt
import numpy as np
import torch


plt.subplot(1,2,1)
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 1
DPI = 600


fig = plt.figure(figsize=(14, 5), dpi=DPI)
u=0
R = 9
gridsize = 25
kwargs = {'xmin': -R, 'xmax': R, 'ymin': -R, 'ymax': R, 'gridsize': gridsize, 'u': u}
field = get_field(**kwargs)
plt.subplot(1,2,1)
plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2)) 
x0 = [np.pi/3, np.pi]
N=10000
h=0.01
states, _ = generate_trajectory("pendulum", x0, N, h).get_trajectory()
for i, l in enumerate(np.split(states, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], (1/3)*l[:,1],color=color, linewidth=LINE_WIDTH)
plt.xlim(0, 6)
plt.ylim(-3, 3)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
plt.title("Phase Space for the Simple Pendulum (h = 0.01)", pad=10)

plt.subplot(1,2,2)

plt.quiver(field['x'][:,0], (1/3)*field['x'][:,1], field['dx'][:,0], (1/3)*field['dx'][:,1],
            cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2,.2,.2)) 
N=10000
h=0.1
plt.title("Numerical Damping (h = 0.1)", pad=10)
plt.xlabel("$q$", fontsize=14)
plt.ylabel("$p$", rotation=0, fontsize=14)
states, _ = generate_trajectory("pendulum", x0, N, h).get_trajectory()
for i, l in enumerate(np.split(states, LINE_SEGMENTS)):
    color = (float(i)/LINE_SEGMENTS, 0, 1-float(i)/LINE_SEGMENTS)
    plt.plot(l[:,0], (1/3)*l[:,1],color=color, linewidth=LINE_WIDTH)
plt.xlim(0, 6)
plt.ylim(-3, 3)    
fig_dir= './figures'
plt.savefig('{}/symplectomorphism_plot'.format(fig_dir, 'png'))

# %%
fig = plt.figure(DPI)
x0 = [np.pi/3, np.pi]
N=100
h=0.02
states, _ = generate_trajectory("pendulum", x0, N, h).get_trajectory()
plt.plot(states[:,0], (1/3)*states[:,1],color='b', linewidth=3)
noise = torch.rand(states.shape)
states +=np.array(noise)*0.5
plt.plot(states[:,0], (1/3)*states[:,1],color='r', linewidth=2)
plt.xlim(-3, 3)
plt.ylim(-3, 3)  
fig_dir= './figures'



