from astroquery.mast import Observations
import lightkurve as lk
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Search for TESS observations of Eclipsing Binaries
    obs = Observations.query_criteria(
        obs_collection="TESS",
        project="TESS",
        target_name="*EB*"
    )
    print("Found observations:", len(obs))
    for row in obs[:20]:
        print(row['target_name'], row['project'], row['provenance_name'])
