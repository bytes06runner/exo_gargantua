import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import lightkurve as lk
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Query the TOI catalog from NASA Exoplanet Archive
    try:
        print("Querying TOI catalog...")
        toi_table = NasaExoplanetArchive.query_criteria(
            table="toi", select="tic_id,tfopwg_disp", 
            where="tfopwg_disp='FP'"
        )
        df = toi_table.to_pandas()
        
        fps_found = []
        for tic in df['tic_id'].unique()[:20]:
            tic_str = f"TIC {tic}"
            print(f"Checking {tic_str}...")
            res = lk.search_lightcurve(tic_str, author="SPOC", mission="TESS")
            if len(res) > 0:
                print(f"Found SPOC light curves for {tic_str}!")
                fps_found.append(tic_str)
                if len(fps_found) >= 2:
                    break
        
        print("Accessible FPs:", fps_found)
    except Exception as e:
        print(e)
