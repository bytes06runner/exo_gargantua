import json

def generate_appendix():
    lines = []
    with open('real_benchmark_results.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
                
    with open('appendix_table.tex', 'w') as out:
        out.write("\\begin{deluxetable*}{llcccc}\n")
        out.write("\\tablecaption{Per-Target Automated Benchmark Results \\label{tab:per_target}}\n")
        out.write("\\tablewidth{0pt}\n")
        out.write("\\tablehead{\n")
        out.write("\\colhead{Target ID} & \\colhead{Target Type} & \\colhead{Catalog Period [d]} & \\colhead{Vanilla BLS Period [d]} & \\colhead{Exo-Gargantua Period [d]} & \\colhead{Exo Disposition}\n")
        out.write("}\n")
        out.write("\\startdata\n")
        
        for row in lines:
            target = row.get('target_id', '').replace('_', '\\_')
            t_type = row.get('type', '')
            cat_p = row.get('catalog_period', 0.0)
            spoc_p = row.get('spoc_period', 0.0)
            exo_p = row.get('exo_period', 0.0)
            exo_rej = row.get('exo_rejected', False)
            disp = "Rejected (FP)" if exo_rej else "Candidate"
            
            # Format periods nicely
            cat_p_str = f"{cat_p:.4f}" if cat_p else "N/A"
            spoc_p_str = f"{spoc_p:.4f}" if spoc_p else "N/A"
            exo_p_str = f"{exo_p:.4f}" if exo_p else "N/A"
            
            out.write(f"{target} & {t_type} & {cat_p_str} & {spoc_p_str} & {exo_p_str} & {disp} \\\\\n")
            
        out.write("\\enddata\n")
        out.write("\\end{deluxetable*}\n")

if __name__ == '__main__':
    generate_appendix()
