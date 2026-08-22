#!/usr/bin/env python3
"""
Generate publication-ready 1x2 grouped bar chart for the Exo-Gargantua benchmark.
Updated with clean, deduplicated 109-target metrics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Professional academic styling ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

# --- Data ---
rate_labels = ['Total\nRecovery', 'Shallow\nRecovery', 'FP\nRejection']
vanilla_rates = [55.0, 40.0, 100.0]
exo_rates     = [50.0, 35.0, 30.6]

snr_labels = ['Median SNR']
vanilla_snr = [0.001]  # effectively zero
exo_snr     = [296.2]

# --- Colors ---
c_vanilla = '#4878A8'   # muted steel blue
c_exo     = '#D95F02'   # bold burnt orange

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                gridspec_kw={'width_ratios': [3, 1.2], 'wspace': 0.35})

# ========== LEFT PANEL: Rates (%) ==========
x = np.arange(len(rate_labels))
w = 0.32

bars1 = ax1.bar(x - w/2, vanilla_rates, w, color=c_vanilla, edgecolor='#2C4A6E',
                linewidth=0.6, label='Vanilla BLS', zorder=3)
bars2 = ax1.bar(x + w/2, exo_rates, w, color=c_exo, edgecolor='#8B3A01',
                linewidth=0.6, label='Exo-Gargantua', zorder=3)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=7, color='#2C4A6E', fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.1f}%',
             ha='center', va='bottom', fontsize=7, color='#8B3A01', fontweight='bold')

ax1.set_ylabel('Rate (%)')
ax1.set_ylim(0, 120)
ax1.set_xticks(x)
ax1.set_xticklabels(rate_labels)
ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='#cccccc')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ========== RIGHT PANEL: Median SNR ==========
x2 = np.arange(len(snr_labels))

bars3 = ax2.bar(x2 - w/2, vanilla_snr, w, color=c_vanilla, edgecolor='#2C4A6E',
                linewidth=0.6, label='Vanilla BLS', zorder=3)
bars4 = ax2.bar(x2 + w/2, exo_snr, w, color=c_exo, edgecolor='#8B3A01',
                linewidth=0.6, label='Exo-Gargantua', zorder=3)

# Value labels
for bar in bars3:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 8, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7, color='#2C4A6E', fontweight='bold')
for bar in bars4:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 8, f'{h:.1f}',
             ha='center', va='bottom', fontsize=7, color='#8B3A01', fontweight='bold')

ax2.set_ylabel('Median SNR')
ax2.set_ylim(0, 360)
ax2.set_xticks(x2)
ax2.set_xticklabels(snr_labels)
ax2.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_benchmark_barchart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved fig_benchmark_barchart.png")
