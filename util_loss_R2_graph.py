import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read data from CSV file
df = pd.read_csv('k_fold_tests.csv')

# Extract data
window_sizes = df['Windows'].unique()
window_sizes.sort()  # Ensure window sizes are in order

# Initialize arrays to store data
short_loss = np.zeros_like(window_sizes, dtype=float)
linear_loss = np.zeros_like(window_sizes, dtype=float)
long_loss = np.zeros_like(window_sizes, dtype=float)
short_r2 = np.zeros_like(window_sizes, dtype=float)
linear_r2 = np.zeros_like(window_sizes, dtype=float)
long_r2 = np.zeros_like(window_sizes, dtype=float)

# Fill arrays with data from CSV
for i, window in enumerate(window_sizes):
    # Filter data for each window size
    window_data = df[df['Windows'] == window]
    
    # Calculate means for each model type
    short_loss[i] = window_data[window_data['Mode'] == 'short']['Validation Loss (mean)'].mean()
    linear_loss[i] = window_data[window_data['Mode'] == 'linear']['Validation Loss (mean)'].mean()
    long_loss[i] = window_data[window_data['Mode'] == 'long']['Validation Loss (mean)'].mean()
    
    short_r2[i] = window_data[window_data['Mode'] == 'short']['R2 Score (mean)'].mean()
    linear_r2[i] = window_data[window_data['Mode'] == 'linear']['R2 Score (mean)'].mean()
    long_r2[i] = window_data[window_data['Mode'] == 'long']['R2 Score (mean)'].mean()

# Find best points considering all model types
all_losses = np.array([short_loss, linear_loss, long_loss])
all_r2s = np.array([short_r2, linear_r2, long_r2])

# Find indices of best points (minimum loss and maximum R²)
best_loss_idx = np.unravel_index(np.argmin(all_losses), all_losses.shape)
best_r2_idx = np.unravel_index(np.argmax(all_r2s), all_r2s.shape)

# Map indices to corresponding values
model_types = ['short', 'linear', 'long']
best_loss_model = model_types[best_loss_idx[0]]
best_r2_model = model_types[best_r2_idx[0]]

best_loss_window = window_sizes[best_loss_idx[1]]
best_r2_window = window_sizes[best_r2_idx[1]]

# Print values for verification
print(f"Best loss: {all_losses[best_loss_idx]:.9f} (model: {best_loss_model}, window: {best_loss_window})")
print(f"Best R²: {all_r2s[best_r2_idx]:.9f} (model: {best_r2_model}, window: {best_r2_window})")

# Create plot
fig, ax1 = plt.subplots(figsize=(12,5))

# Define colors
color_short = 'red'
color_linear = 'green'
color_long = 'blue'
color_best = '#006400'  # Dark green

# Configure shared x-axis
ax1.set_xlabel('Windows')
ax1.set_ylabel('Validation Loss (mean)')
ax1.set_xlim(min(window_sizes)-0.5, max(window_sizes)+0.5)  # Adjust limits for better visualization
ax1.set_xticks(window_sizes)
ax1.grid(True, alpha=0.3)

# Plot Loss lines
loss1, = ax1.plot(window_sizes, short_loss, color=color_short, linestyle='--', label='short - Loss')
loss2, = ax1.plot(window_sizes, linear_loss, color=color_linear, linestyle='--', label='linear - Loss')
loss3, = ax1.plot(window_sizes, long_loss, color=color_long, linestyle='--', label='long - Loss')

# Create second axis sharing the same x-axis
ax2 = ax1.twinx()
ax2.set_ylabel('$R^2$ Score (mean)')
ax2.set_xlim(min(window_sizes)-0.5, max(window_sizes)+0.5)  # Same x-axis limits
ax2.set_xticks(window_sizes)

# Plot R² lines
r2_1, = ax2.plot(window_sizes, short_r2, color=color_short, label='short - $R^2$')
r2_2, = ax2.plot(window_sizes, linear_r2, color=color_linear, label='linear - $R^2$')
r2_3, = ax2.plot(window_sizes, long_r2, color=color_long, label='long - $R^2$')

# Mark best points with stars
best_loss_value = all_losses[best_loss_idx]
best_r2_value = all_r2s[best_r2_idx]

# Plot stars, but only one will have a label in the legend
best1, = ax1.plot(best_loss_window, best_loss_value, marker='*', markersize=15, color=color_best, label='best')
best2, = ax2.plot(best_r2_window, best_r2_value, marker='*', markersize=15, color=color_best, label='_nolegend_')

# Annotate values near the stars
ax1.annotate(f'{best_loss_value:.6f}', (best_loss_window, best_loss_value), 
            xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, weight='bold', color=color_best)
ax2.annotate(f'{best_r2_value:.6f}', (best_r2_window+0.3, best_r2_value), fontsize=9, weight='bold', color=color_best)

# Combine all lines for a single legend, with the star between Loss and R²
lines = [r2_1, r2_2, r2_3, best1, loss1, loss2, loss3]
labels = [l.get_label() for l in lines]

# Legend on the right side with position adjustment
plt.legend(lines, labels, bbox_to_anchor=(1.15, 0.5), loc='center left')

# Title
plt.title('Validation Loss and $R^2$ Score vs Windows')

# Adjust layout
plt.tight_layout()
plt.show() 