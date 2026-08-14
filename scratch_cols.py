from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# Print the columns of the TOI table
print(NasaExoplanetArchive.query_criteria(table="toi", select="*").colnames)
