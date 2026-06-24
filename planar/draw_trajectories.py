import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from sims.visualization_utils import to_image_frame, visualize_point



goal_pos = np.array([11.5, 8.5])

dataset = 'univ'
plt.clf(), plt.cla()
fig, ax = plt.subplots()

image = cv2.imread(os.path.join(os.path.dirname(__file__), 'traj', '{}.png'.format(dataset)))
ax.imshow(image, cmap='gray', alpha=0.5)
h, w, _ = image.shape
ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis('off')

homography_path = os.path.join(os.path.dirname(__file__), '../assets/homographies', dataset + '.txt')
H = np.loadtxt(homography_path, dtype=float)

visualize_point(goal_pos, H, ax, color='black', marker='s', s=80, label=None, zorder=500)
goal_image_x, goal_image_y = to_image_frame(goal_pos[None, ...], H)
ax.text(goal_image_x, goal_image_y + 30, 'goal', fontsize=18)


colors = ['tab:blue', 'tab:brown', 'tab:purple']
zorders = [10, 5, 1]
labels = ['ECP-MPC', 'ACP-MPC', 'CC']

marked = False



def add_arrows(xs, ys, n_skips, ax, color):
    n_steps = xs.size
    for i in range(0, n_steps, n_skips):
        if i +1 < n_steps:
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            ax.arrow(x=xs[i], y=ys[i], dx=dx, dy=dy, color=color, head_width=8, head_length=8)

for controller, color, zorder, label in zip(['ecp-mpc', 'acp-mpc', 'cc'], colors, zorders, labels):
    trajectories = np.load(os.path.join(os.path.dirname(__file__), 'traj/{}_{}.npy'.format(dataset, controller)))

    for traj in trajectories:

        traj_image_x, traj_image_y = to_image_frame(traj, H)

        if not marked:
            pos_begin = traj[0]
            visualize_point(pos_begin, H, ax, color='black', marker='s', s=80, zorder=500, label=None)
            ax.text(traj_image_x[0], traj_image_y[0] + 30, 'start', fontsize=18)

            marked = True
        ax.plot(traj_image_x, traj_image_y, color=color, label=label, linewidth=2, zorder=zorder)

        add_arrows(traj_image_x, traj_image_y, n_skips=10, ax=ax, color=color)
        ax.scatter(traj_image_x[-1], traj_image_y[-1], marker='o', color=color, s=40)


        ax.legend(loc='upper right', fontsize=16)
        fig.tight_layout()
        fig.savefig(os.path.join('compare.pdf'), bbox_inches='tight', pad_inches=0)
        plt.close()
