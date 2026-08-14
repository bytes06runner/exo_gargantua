from astroquery.simbad import Simbad
import lightkurve as lk
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    custom_simbad = Simbad()
    custom_simbad.add_votable_fields('otype')
    
    # Query for known eclipsing binaries (EB*)
    result = custom_simbad.query_criteria('otype="EB*"', limit=20)
    
    found_targets = []
    if result is not None:
        for row in result:
            name = row['MAIN_ID']
            print(f"Checking {name}...")
            try:
                res = lk.search_lightcurve(name, author="SPOC", mission="TESS")
                if len(res) > 0:
                    print(f"  -> Found {len(res)} SPOC lightcurves!")
                    found_targets.append(name)
                    if len(found_targets) >= 2:
                        break
            except Exception:
                pass

    print("Use these targets for validation:")
    print(found_targets)
