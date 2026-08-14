import os
import json
import numpy as np
import matplotlib.pyplot as plt
import corner


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'mask'):  # handle masked arrays/elements
            return float(obj) if not np.ma.is_masked(obj) else None
        return super().default(obj)

def generate_summary_report(tic_id, bls_results, vetting_results, centroid_results, mcmc_posteriors, derived_params, output_dir="output"):
    """
    Exports a JSON summary of all pipeline outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        'target_id': tic_id,
        'bls_parameters': {
            'period_days': bls_results['period'].value if hasattr(bls_results['period'], 'value') else float(bls_results['period']),
            't0_btjd': bls_results['t0'].value if hasattr(bls_results['t0'], 'value') else float(bls_results['t0']),
            'depth': bls_results['depth'].value if hasattr(bls_results['depth'], 'value') else float(bls_results['depth']),
            'duration_days': bls_results['duration'].value if hasattr(bls_results['duration'], 'value') else float(bls_results['duration']),
            'snr': bls_results['snr'].value if hasattr(bls_results['snr'], 'value') else float(bls_results['snr'])
        },
        'vetting': {
            'odd_even_depth_diff': vetting_results.get('depth_diff'),
            'secondary_eclipse_depth': vetting_results.get('secondary_eclipse_depth'),
            'secondary_eclipse_sigma': vetting_results.get('secondary_eclipse_sigma')
        },
        'centroid': {
            'shift_pixels': centroid_results.get('centroid_shift'),
            'passed': centroid_results.get('centroid_vetting_passed')
        },
        'mcmc_posteriors': mcmc_posteriors,
        'derived_physical_parameters': derived_params
    }

    # Clean tic_id for filename
    clean_id = tic_id.replace("TIC", "").strip()
    out_path = os.path.join(output_dir, f"TIC_{clean_id}_summary.json")
    
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4, cls=NumpyEncoder)
    
    print(f"  Saved summary report to {out_path}")
    return out_path


def generate_publication_figure(tic_id, lc, mcmc_sampler, vetting_results, centroid_results, output_dir="output"):
    """
    Generates a 4-panel diagnostic plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    
    os.makedirs(output_dir, exist_ok=True)
    clean_id = tic_id.replace("TIC", "").strip()
    out_path = os.path.join(output_dir, f"TIC_{clean_id}_vetting_summary_panel.png")

    plt.figure(figsize=(16, 12))
    
    # 1. Phase-folded LC + Model (Top row)
    ax1 = plt.subplot(2, 2, 1)
    folded_lc = vetting_results['folded']
    binned_lc = vetting_results['binned']
    
    folded_lc.scatter(ax=ax1, c='gray', alpha=0.3, label='Raw Cadences')
    binned_lc.plot(ax=ax1, c='red', lw=2, label='Binned Average')
    ax1.set_title(f"Phase-Folded Light Curve: TIC {clean_id}")
    ax1.set_xlim(-0.1, 0.1)

    # 2. Corner Plot (Bottom Left)
    # Extract flattened samples for P, t0, rp_rs, a_rs
    if mcmc_sampler is not None:
        samples = mcmc_sampler.get_chain(discard=500, flat=True)
        # Select first 4 parameters for the corner plot (P, t0, rp, a)
        samples_corner = samples[:, :4]
        labels = ["Period (d)", "t0", "Rp/Rs", "a/Rs"]
        
        # Create corner plot in the designated axes
        # (Corner usually creates a new figure, so we need some hacking)
        fig_corner = corner.corner(samples_corner, labels=labels, show_titles=True, plot_datapoints=False)
        # Save corner plot separately because corner.py doesn't play well with subplots
        corner_path = os.path.join(output_dir, f"TIC_{clean_id}_corner.png")
        fig_corner.savefig(corner_path)
        plt.close(fig_corner)
        
        # Display the saved corner plot in ax2
        ax2 = plt.subplot(2, 2, 3)
        img = plt.imread(corner_path)
        ax2.imshow(img)
        ax2.axis('off')
        ax2.set_title("MCMC Posteriors")
    else:
        ax2 = plt.subplot(2, 2, 3)
        ax2.text(0.5, 0.5, "MCMC not run", ha='center', va='center')
        ax2.axis('off')

    # 3. Centroid Map (Top Right)
    ax3 = plt.subplot(2, 2, 2)
    # Check if we have centroid images
    if 'in_transit_img' in centroid_results and 'out_transit_img' in centroid_results:
        diff_img = centroid_results['out_transit_img'] - centroid_results['in_transit_img']
        im = ax3.imshow(diff_img, origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_title(f"Centroid Offset (Out - In)\nShift: {centroid_results.get('centroid_shift', 0):.4f} pix")
    else:
        ax3.text(0.5, 0.5, "No Centroid Images Available", ha='center', va='center')
        ax3.axis('off')

    # 4. Odd/Even + Secondary Eclipse (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    vetting_results['odd'].plot(ax=ax4, c='orange', label='Odd Transits', lw=2)
    vetting_results['even'].plot(ax=ax4, c='blue', label='Even Transits', lw=2)
    ax4.set_title(f"Odd/Even (Diff: {vetting_results.get('depth_diff', 0):.5f}) | Sec Eclipse: {vetting_results.get('secondary_eclipse_sigma', 0):.1f}σ")
    ax4.set_xlim(-0.2, 0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved vetting summary panel to {out_path}")
    return out_path
