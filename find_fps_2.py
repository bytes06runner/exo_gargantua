import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    import lightkurve as lk
    
    # Query all TOIs
    table = NasaExoplanetArchive.query_criteria(table="toi", select="ticid,tfopwg_disp")
    df = table.to_pandas()
    
    # Filter for False Positives
    fps = df[df['tfopwg_disp'] == 'FP']
    
    print(f"Found {len(fps)} false positives in TOI catalog.")
    
    found_targets = []
    for tic in fps['ticid'].unique()[:50]:
        target = f"TIC {tic}"
        print(f"Checking {target}...")
        try:
            res = lk.search_lightcurve(target, author="SPOC", mission="TESS")
            if len(res) > 0:
                print(f"  -> Found {len(res)} SPOC lightcurves!")
                found_targets.append(target)
                if len(found_targets) >= 2:
                    break
        except Exception as e:
            pass
            
    print("Use these targets for validation:")
    print(found_targets)
