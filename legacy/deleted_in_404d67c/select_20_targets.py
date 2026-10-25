import json
import random
import os
import shutil

# First make a backup of the original 1000 target json
if not os.path.exists('benchmark_targets_full.json'):
    shutil.copy('benchmark_targets.json', 'benchmark_targets_full.json')

with open('benchmark_targets_full.json', 'r') as f:
    targets = json.load(f)

# The 3 we definitely want
required_ids = ["TIC 180695581", "TIC 36724087", "TIC 142937186"]

cp_targets = [t for t in targets if t['ground_truth'] == 'PLANET' and t['target_id'] not in required_ids]
fp_targets = [t for t in targets if t['ground_truth'] == 'FALSE_POSITIVE' and t['target_id'] not in required_ids]

# We need 20 total. If we add 3 required, we need 17 more. 
# Let's say 8 CP and 9 FP to get ~10 of each
random.seed(42)
selected_cp = random.sample(cp_targets, min(8, len(cp_targets)))
selected_fp = random.sample(fp_targets, min(9, len(fp_targets)))

required_targets = [t for t in targets if t['target_id'] in required_ids]

final_20 = required_targets + selected_cp + selected_fp

with open('benchmark_targets.json', 'w') as f:
    json.dump(final_20, f, indent=2)

print(f"Created a 20-target benchmark_targets.json. Total targets: {len(final_20)}")
