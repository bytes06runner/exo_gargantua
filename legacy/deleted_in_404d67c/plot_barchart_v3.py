#!/usr/bin/env python3
"""
Generate publication-ready 1x1 grouped bar chart for the Exo-Gargantua benchmark.
Updated with clean, deduplicated 109-target metrics, dropping the SNR panel.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Professional academic styling ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

# --- Data ---
rate_labels = ['Total\nRecovery', 'Shallow\nRecovery', 'FP\nRejection']
vanilla_rates = [55.0, 40.0, 100.0]
exo_rates     = [50.0, 35.0, 30.6]

# --- Colors ---
c_vanilla = '#4878A8'   # muted steel blue
c_exo     = '#D95F02'   # bold burnt orange

# --- Figure ---
fig, ax = plt.subplots(figsize=(6.0, 4.0))

# ========== SINGLE PANEL: Rates (%) ==========
x = np.arange(len(rate_labels))
w = 0.35

bars1 = ax.bar(x - w/2, vanilla_rates, w, color=c_vanilla, edgecolor='#2C4A6E',
                linewidth=0.8, label='Vanilla BLS', zorder=3)
bars2 = ax.bar(x + w/2, exo_rates, w, color=c_exo, edgecolor='#8B3A01',
                linewidth=0.8, label='Exo-Gargantua', zorder=3)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=9, color='#2C4A6E', fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=9, color='#8B3A01', fontweight='bold')

ax.set_ylabel('Rate (%)')
ax.set_ylim(0, 125)  # Leave room for legend
ax.set_xticks(x)
ax.set_xticklabels(rate_labels)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_benchmark_barchart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_benchmark_barchart.png")
