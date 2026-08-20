import json

with open('benchmark_targets.json', 'r') as f:
    targets = json.load(f)

# Create lookup for ALL 20 targets
cat_periods = {}
for t in targets:
    if t.get('pl_orbper') is not None:
        cat_periods[t['target_id']] = t.get('pl_orbper')

results = []
try:
    with open('benchmark_results.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
except FileNotFoundError:
    pass

print("| Target ID | Catalog Period (d) | BLS Period (d) | MCMC Period (d) | Error (%) | Match? (<1%) |")
print("| :--- | :---: | :---: | :---: | :---: | :---: |")

matches = 0
total_targets = 0

for res in results:
    tid = res['target_id']
    if tid not in cat_periods:
        continue
    
    cp = cat_periods[tid]
    total_targets += 1
    
    bls_p = res.get('bls_period', 0)
    mcmc_p = res.get('mcmc_period', 0)
    
    if mcmc_p is None:
        mcmc_p = bls_p
        
    if mcmc_p is None or cp is None:
        continue
        
    err = abs(mcmc_p - cp) / cp * 100
    is_match = err < 1.0
    
    if is_match:
        matches += 1
        
    match_str = "✅" if is_match else "❌"
    
    print(f"| {tid} | {cp:.5f} | {bls_p:.5f} | {mcmc_p:.5f} | {err:.2f}% | {match_str} |")

if total_targets > 0:
    print(f"\n**Overall Period Match Rate:** {matches}/{total_targets} ({matches/total_targets*100:.1f}%)")
else:
    print("\nNo targets processed yet.")
