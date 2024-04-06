import numpy as np

def calculate_ttc(ref_pos, ref_vel, other_pos, other_vel):
    """
    Calculates the time-to-collision (TTC) between the reference vehicle (with position ref_pos and velocity ref_vel)
    and the other vehicle (with position other_pos and velocity other_vel).
    """
    # Calculate relative position and velocity
    rel_pos = other_pos - ref_pos
    rel_vel = other_vel - ref_vel

    # Calculate the time to collision
    dist = np.linalg.norm(rel_pos)
    print(dist)
    relative_speed = np.dot(rel_vel, rel_pos) / dist
    ttc = dist / abs(relative_speed) if abs(relative_speed) > 1e-3 else float('inf')

    return ttc



# Positions and velocities of the three vehicles
ref_pos = np.array([0, 0])
ref_vel = np.array([10, 0])
other1_pos = np.array([30, 0])
other1_vel = np.array([-10, 0])
other2_pos = np.array([60, 0])
other2_vel = np.array([-5, 0])

# Calculate the TTCs between the reference vehicle and the other two vehicles
ttc1 = calculate_ttc(ref_pos, ref_vel, other1_pos, other1_vel)
ttc2 = calculate_ttc(ref_pos, ref_vel, other2_pos, other2_vel)

# Print the TTCs
print("TTC between reference vehicle and other vehicle 1: ", ttc1)
print("TTC between reference vehicle and other vehicle 2: ", ttc2)