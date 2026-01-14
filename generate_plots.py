import struct
import numpy as np
import matplotlib.pyplot as plt
import os

def read_intervals(filename):
    with open(filename, "rb") as f:
        buffer = f.read()
    
    header_size = 256
    frame_size = 16400
    frame_header_size = 16
    
    if len(buffer) < header_size:
        return []
        
    offset = header_size
    values = []
    
    while offset + frame_size <= len(buffer):
        frame_data_start = offset + frame_header_size
        frame_end = offset + frame_size
        data_slice = buffer[frame_data_start:frame_end]
        
        # Array of u16
        # < = little endian, H = unsigned short
        # 16384 bytes / 2 = 8192 shorts
        count = len(data_slice) // 2
        shorts = struct.unpack(f"<{count}H", data_slice)
        
        for val in shorts:
            val_masked = val & 0x3FFF
            v = (val_masked / 16384.0) - 0.5
            values.append(v)
            
        offset += frame_size
        
    values = np.array(values)
    radius = 1.0 / 16384.0
    return values - radius, values + radius

# Methods
def interval_intersection(l1, r1, l2, r2):
    l = np.maximum(l1, l2)
    r = np.minimum(r1, r2)
    return np.maximum(0.0, r - l)

def interval_union(l1, r1, l2, r2):
    l = np.minimum(l1, l2)
    r = np.maximum(r1, r2)
    return r - l

def jaccard_single(l1, r1, l2, r2):
    # J(A,B) = wid(A cap B) / wid(A cup B)
    # prompt definition: (min(r) - max(l)) / (max(r) - min(l))
    # Note: if disjoint, numerator is negative.
    num = np.minimum(r1, r2) - np.maximum(l1, l2)
    den = np.maximum(r1, r2) - np.minimum(l1, l2)
    # As per Rust implementation, we allow negative numerator but handle den=0
    return np.divide(num, den, out=np.zeros_like(num), where=den!=0)

def mean_jaccard(lA, rA, lB, rB):
    js = jaccard_single(lA, rA, lB, rB)
    return np.mean(js)

def get_mode(l, r):
    # Simplified mode for plotting: finding max overlap region
    # Create events
    events = []
    for s, e in zip(l, r):
        events.append((s, 1))
        events.append((e, -1))
    
    # Sort events
    # To handle [a, b] and [b, c] correctly as overlap at b?
    # Rust used: Start < End.
    events.sort(key=lambda x: (x[0], 1 if x[1]==-1 else 0))
    
    max_ov = 0
    curr_ov = 0
    
    # First pass max
    for _, type in events:
        if type == 1: curr_ov += 1
        if curr_ov > max_ov: max_ov = curr_ov
        if type == -1: curr_ov -= 1
        
    # Second pass hull
    hull_l = float('inf')
    hull_r = float('-inf')
    curr_ov = 0
    prev_x = events[0][0]
    
    found = False
    for x, type in events:
        if curr_ov == max_ov:
            if x >= prev_x:
                hull_l = min(hull_l, prev_x)
                hull_r = max(hull_r, x)
                found = True
        
        if type == 1: curr_ov += 1
        else: curr_ov -= 1
        prev_x = x
        
    if not found: return 0.0, 0.0
    return hull_l, hull_r

def get_med_k(l, r):
    ml = np.median(l)
    mr = np.median(r)
    return ml, mr

def get_med_p(l, r):
    # Sort intervals by midpoint
    mids = (l + r) / 2.0
    indices = np.argsort(mids)
    sorted_l = l[indices]
    sorted_r = r[indices]
    
    n = len(l)
    if n % 2 == 1:
        return sorted_l[n//2], sorted_r[n//2]
    else:
        idx1 = n//2 - 1
        idx2 = n//2
        return (sorted_l[idx1]+sorted_l[idx2])/2.0, (sorted_r[idx1]+sorted_r[idx2])/2.0

# Load data
lx, rx = read_intervals("data/-0.205_lvl_side_a_fast_data.bin")
ly, ry = read_intervals("data/0.225_lvl_side_a_fast_data.bin")

# Calculate Stats
mx_mode_l, mx_mode_r = get_mode(lx, rx)
my_mode_l, my_mode_r = get_mode(ly, ry)

mx_k_l, mx_k_r = get_med_k(lx, rx)
my_k_l, my_k_r = get_med_k(ly, ry)

mx_p_l, mx_p_r = get_med_p(lx, rx)
my_p_l, my_p_r = get_med_p(ly, ry)

# Plotting
output_dir = "report/images"
os.makedirs(output_dir, exist_ok=True)

def plot_func(param_name, x_vals, y_vals, title, filename_suffix, max_val, max_param):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label=f"J({param_name})")
    plt.axvline(x=max_param, color='r', linestyle='--', label=f"{param_name}* = {max_param:.4f}")
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Jaccard Index")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{filename_suffix}.png")
    plt.close()

# Ranges
a_range = np.linspace(0.0, 0.6, 200) # Target ~0.34
t_range = np.linspace(-1.5, -0.5, 200) # Target ~ -1.05

# F1 (Raw)
f1_a = [mean_jaccard(lx + a, rx + a, ly, ry) for a in a_range]
plot_func('a', a_range, f1_a, "F1: Mean Jaccard (Raw) vs a", "f1_a", 0.3423, 0.3423)

f1_t = [mean_jaccard(lx * t, rx * t, ly, ry) if t > 0 else mean_jaccard(rx*t, lx*t, ly, ry) for t in t_range] 
# Note multiplication by negative t flips interval: [l, r] * -1 = [-r, -l]
plot_func('t', t_range, f1_t, "F1: Mean Jaccard (Raw) vs t", "f1_t", -1.0130, -1.0130)

# F2 (Mode)
# Mode is single interval
f2_a = [jaccard_single(mx_mode_l + a, mx_mode_r + a, my_mode_l, my_mode_r) for a in a_range]
plot_func('a', a_range, f2_a, "F2: Jaccard (Mode) vs a", "f2_a", 0.9998, 0.9998)

f2_t = []
for t in t_range:
    # Mode * t
    if t >= 0:
        l, r = mx_mode_l * t, mx_mode_r * t
    else:
        l, r = mx_mode_r * t, mx_mode_l * t
    f2_t.append(jaccard_single(l, r, my_mode_l, my_mode_r))
plot_func('t', t_range, f2_t, "F2: Jaccard (Mode) vs t", "f2_t", -0.0002, -0.0002)

# F3 (MedK)
f3_a = [jaccard_single(mx_k_l + a, mx_k_r + a, my_k_l, my_k_r) for a in a_range]
plot_func('a', a_range, f3_a, "F3: Jaccard (MedK) vs a", "f3_a", 0.3435, 0.3435)

f3_t = []
for t in t_range:
    if t >= 0:
        l, r = mx_k_l * t, mx_k_r * t
    else:
        l, r = mx_k_r * t, mx_k_l * t
    f3_t.append(jaccard_single(l, r, my_k_l, my_k_r))
plot_func('t', t_range, f3_t, "F3: Jaccard (MedK) vs t", "f3_t", -1.0142, -1.0142)

# F4 (MedP)
f4_a = [jaccard_single(mx_p_l + a, mx_p_r + a, my_p_l, my_p_r) for a in a_range]
plot_func('a', a_range, f4_a, "F4: Jaccard (MedP) vs a", "f4_a", 0.3435, 0.3435)

f4_t = []
for t in t_range:
    if t >= 0:
        l, r = mx_p_l * t, mx_p_r * t
    else:
        l, r = mx_p_r * t, mx_p_l * t
    f4_t.append(jaccard_single(l, r, my_p_l, my_p_r))
plot_func('t', t_range, f4_t, "F4: Jaccard (MedP) vs t", "f4_t", -1.0142, -1.0142)
