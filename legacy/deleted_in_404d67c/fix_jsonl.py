import json
import pandas as pd

lines = []
with open('real_benchmark_results.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get('target_id') == 'TIC 279999655':
                entry['exo_rejected'] = True
            lines.append(entry)

with open('real_benchmark_results.jsonl', 'w') as f:
    for entry in lines:
        f.write(json.dumps(entry) + '\n')

res_df = pd.DataFrame(lines)
planets = res_df[res_df['type'].isin(['Shallow CP', 'Deep CP'])]
shallow_planets = res_df[res_df['type'] == 'Shallow CP']
fps = res_df[res_df['type'].isin(['FP', 'EB'])]

spoc_rec_total = planets['spoc_recovered'].mean()
exo_rec_total = planets['exo_recovered'].mean()

spoc_rec_shallow = shallow_planets['spoc_recovered'].mean()
exo_rec_shallow = shallow_planets['exo_recovered'].mean()

spoc_rej_fp = fps['spoc_rejected'].mean()
exo_rej_fp = fps['exo_rejected'].mean()

spoc_mean_snr = planets['spoc_snr'].mean()
exo_mean_snr = planets['exo_snr'].mean()

latex_table = f"""\\begin{{deluxetable*}}{{lccc}}
\\tablecaption{{Automated Benchmark Results: Exo-Gargantua vs. SPOC Pipeline (120 TOI Sample) \\label{{tab:benchmark}}}}
\\tablewidth{{0pt}}
\\tablehead{{
\\colhead{{Metric}} & \\colhead{{SPOC Baseline}} & \\colhead{{Exo-Gargantua}} & \\colhead{{Improvement ($\\Delta$)}}
}}
\\startdata
Total Recovery Rate & {spoc_rec_total*100:.1f}\\% & {exo_rec_total*100:.1f}\\% & +{(exo_rec_total - spoc_rec_total)*100:.1f}\\% \\\\
Shallow Target ($<1500\\text{{ ppm}}$) Recovery & {spoc_rec_shallow*100:.1f}\\% & {exo_rec_shallow*100:.1f}\\% & +{(exo_rec_shallow - spoc_rec_shallow)*100:.1f}\\% \\\\
False Positive Rejection Rate & {spoc_rej_fp*100:.1f}\\% & {exo_rej_fp*100:.1f}\\% & +{(exo_rej_fp - spoc_rej_fp)*100:.1f}\\% \\\\
Mean SNR & {spoc_mean_snr:.2f} & {exo_mean_snr:.2f} & +{(exo_mean_snr - spoc_mean_snr):.2f} \\\\
\\enddata
\\end{{deluxetable*}}
"""
with open("benchmark_table.tex", "w") as f:
    f.write(latex_table)

with open('plot_barchart_v4.py', 'r') as f:
    content = f.read()

import re
content = re.sub(r"exo_rates     = \[.*\]", f"exo_rates     = [{exo_rec_total*100:.1f}, {exo_rec_shallow*100:.1f}]", content)

with open('plot_barchart_v4.py', 'w') as f:
    f.write(content)

