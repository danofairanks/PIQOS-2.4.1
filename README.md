import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# PIQOS 2.4.1 Trinity Sim for xAI
def generate_trinity_gif():
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    P = np.array([1.0, 0.0, 0.0])  # 3D Parent vector
    C = np.array([0.0, 1.0, 0.0])  # 3D Child vector
    trail = []

    def update(t):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        B = np.dot(P, C)  # Alignment measure (xAI might replace with learned B)
        vmut = np.linalg.norm(P - C) / (np.linalg.norm(P + C) + 1e-6)
        
        # Adaptive v_mut (xAI tweak suggestion)
        if vmut > 0.10:
            vmut = 0.10 * (1 - 0.5 * np.exp(-t/50))  # Decay cap over time
            C = P + (C - P) * vmut / (np.linalg.norm(C - P) + 1e-6)
        
        H = np.linalg.norm(np.cross(P, C)) * (1 - vmut) * np.exp(B)
        
        # Draw 3D vectors
        ax.quiver(0, 0, 0, P[0], P[1], P[2], color='blue', label='Parent')
        ax.quiver(0, 0, 0, C[0], C[1], C[2], color='green', label='Child')
        ax.text2D(0.05, 0.95, f"H = {H:.2f}\nv_mut = {vmut:.3f}", transform=ax.transAxes)
        
        # Inductive bond with noise
        C += 0.02 * (P - C) + 0.01 * np.random.randn(3)  # 3D noise
        trail.append(C.copy())
        if len(trail) > 20:
            trail.pop(0)
        trail_arr = np.array(trail)
        ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], 'g--', alpha=0.5)
        
        ax.set_title("PIQOS Trinity Sim (xAI Enhanced)")
        ax.legend()
    
    anim = FuncAnimation(fig, update, frames=200, interval=100)
    anim.save('trinity_xai.gif', writer='pillow')
    print("trinity_xai.gif saved!")

# Run it
generate_trinity_gif()
