import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

points = np.random.rand(30, 2)   # 30 random points in 2-D
hull = ConvexHull(points)

points_2 = np.random.rand(30, 2)   # 30 random points in 2-D
hull_2 = ConvexHull(points_2)

plt.plot(points[:,0], points[:,1], 'o')
cent = np.mean(points, axis=0)
pts = points[hull.vertices]
print(pts.shape)

k = 1.1
alpha = 0.5
# color = 'green'
color = 'lightcyan'
poly = Polygon(k*(pts - cent) + cent, closed=True,
               capstyle='round', facecolor=color, alpha=alpha)
plt.gca().add_patch(poly)
plt.savefig('./convex.png')

plt.plot(points_2[:,0], points_2[:,1], 'o')
cent_2 = np.mean(points_2, axis=0)
pts_2 = points_2[hull_2.vertices]

poly_2 = Polygon(k*(pts_2 - cent_2) + cent_2, closed=True,
                capstyle='round', facecolor=color, alpha=alpha)
plt.gca().add_patch(poly_2)
plt.savefig('./convex_2.png')
