import matplotlib.pyplot as plt
import numpy as np

# Buffer sizes in bytes and converted to MB
buffer_sizes = np.array([
    1024, 4096, 16384, 65536,
    262144, 1048576, 4194304, 16777216,
    67108864, 268435456
])
buffer_sizes_mb = buffer_sizes / (1024 ** 2)

# Average throughput (GB/s) from your output
avg_throughput_pageable = [
    0.0525, 0.0775, 0.2675, 0.675,
    1.1925, 1.295, 1.7025, 1.8375,
    0.96, 1.36
]

avg_throughput_pinned = [
    0.095, 0.0875, 0.3375, 1.1375,
    2.1425, 2.46, 2.62, 2.6325,
    2.6625, 2.6625
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(buffer_sizes_mb, avg_throughput_pageable, 'o-', label='Pageable Memory')
plt.plot(buffer_sizes_mb, avg_throughput_pinned, 's-', label='Pinned Memory')

# Log scale for x-axis (buffer size)
plt.xscale('log', base=2)
plt.xticks(buffer_sizes_mb, [f"{x:.2f}" for x in buffer_sizes_mb], rotation=45)
plt.xlabel('Buffer Size (MB)')
plt.ylabel('Average Throughput (GB/s)')
plt.title('CUDA Memory Transfer Throughput: Pinned vs Pageable Memory')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

