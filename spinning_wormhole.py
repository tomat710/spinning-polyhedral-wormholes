
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import plasma

# Dodecahedron vertices (normalized to unit sphere)
phi = (1 + np.sqrt(5)) / 2
vertices = np.array([
    [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
    [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
    [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi],
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
    [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
])
norms = np.linalg.norm(vertices, axis=1)
vertices /= norms[:, np.newaxis]

# Extract edges from ConvexHull
hull = ConvexHull(vertices)
edges = set()
for simplex in hull.simplices:
    for i in range(3):
        edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
        edges.add(edge)
edges = list(edges)

# Casimir calculations
distances = [np.linalg.norm(vertices[e[0]] - vertices[e[1]]) for e in edges]
casimir_values = [-1 / d**4 for d in distances]

# Normalize for color
norm = Normalize(vmin=min(casimir_values), vmax=max(casimir_values))
colors = plasma(norm(casimir_values))
linewidths = [5 / d for d in distances]

# Frame-dragging simulation
def omega_func(r):
    return 0.1 / r

geodesic_points = np.array([[1.0, 0.0, 0.0]])
dt = 0.01
phi_deflection = 0.0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Spinning Polyhedral Wormhole with Frame-Dragging')

ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], color='r', s=50)
lines = [ax.plot([], [], [], color=colors[i], linewidth=linewidths[i])[0] for i in range(len(edges))]
geodesic_line, = ax.plot([], [], [], 'g--', linewidth=2, label='Dragged Geodesic')

def update(frame):
    global geodesic_points, phi_deflection
    angle = frame * (360 / 100)
    rot_matrix = np.array([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])
    rotated_vertices = vertices @ rot_matrix.T

    for i, e in enumerate(edges):
        p1, p2 = rotated_vertices[e[0]], rotated_vertices[e[1]]
        lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
        lines[i].set_3d_properties([p1[2], p2[2]])

    r_current = np.linalg.norm(geodesic_points[-1])
    dphi = omega_func(r_current) * dt * (frame / 10)
    phi_deflection += dphi
    new_point = geodesic_points[-1] @ rot_matrix.T + np.array([0, 0.01 * np.sin(phi_deflection), 0.01 * np.cos(phi_deflection)])
    new_point /= np.linalg.norm(new_point)
    geodesic_points = np.append(geodesic_points, [new_point], axis=0)

    geodesic_line.set_data(geodesic_points[:,0], geodesic_points[:,1])
    geodesic_line.set_3d_properties(geodesic_points[:,2])
    return lines + [geodesic_line]

ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
ani.save('dodecahedron_spinning_wormhole.gif', writer='pillow')
plt.legend()
plt.show()
