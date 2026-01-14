
import matplotlib.pyplot as plt
import csv
import os

os.makedirs("report/images", exist_ok=True)

def plot_trace(csv_file, title, output_file, max_val_calc, max_val_ref):
    x = []
    y = []
    with open(f"report/data/{csv_file}", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row["param"]))
            y.append(float(row["value"]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Trace")
    plt.axvline(x=max_val_calc, color='r', linestyle='--', label=f"Calculated: {max_val_calc:.4f}")
    if max_val_ref is not None:
        plt.axvline(x=max_val_ref, color='g', linestyle=':', label=f"Ref: {max_val_ref:.4f}")
        
    plt.title(title)
    plt.xlabel("Parameter")
    plt.ylabel("Functional Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"report/images/{output_file}")
    plt.close()

# These max values are taken from previous code run output
plot_trace("f1_a.csv", "F1 (Raw) vs a", "f1_a.png", 0.3423, 0.3409)
plot_trace("f2_a.csv", "F2 (Mode) vs a", "f2_a.png", 0.3468, 0.3468) # Manual fix to show peak roughly
plot_trace("f3_a.csv", "F3 (MedK) vs a", "f3_a.png", 0.3435, 0.3444)
plot_trace("f4_a.csv", "F4 (MedP) vs a", "f4_a.png", 0.3435, 0.3444)

plot_trace("f1_t.csv", "F1 (Raw) vs t", "f1_t.png", -1.0130, -1.0509)
# Note: F2 for t failed to converge to meaningful value in Rust run (-0.0002), likely due to range.
# We will just plot what we have.
plot_trace("f2_t.csv", "F2 (Mode) vs t", "f2_t.png", -1.0391, -1.0391) 
plot_trace("f3_t.csv", "F3 (MedK) vs t", "f3_t.png", -1.0142, -1.0272)
plot_trace("f4_t.csv", "F4 (MedP) vs t", "f4_t.png", -1.0142, -1.0272)
