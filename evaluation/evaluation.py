import numpy as np
import matplotlib.pyplot as plt
import os

def create_cdf(data):
    sorted_data = np.sort(data)
    p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return p, sorted_data

def plot_cdf(p, sorted_data):
    output_dir = "outputs"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.plot(sorted_data, p)
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.show()
    plt.savefig(os.path.join(plots_dir, "CDF.png"))
    plt.close()