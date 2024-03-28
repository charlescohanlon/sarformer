BEFORE_PREFIX = "/vsiaz/"
AFTER_PREFIX = "/vsicurl/https://ai4edataeuwest.blob.core.windows.net/"

with open("DEM_Paths_exp_1_30_24.csv", "r") as infile, open(
    "qgis_dem_paths.csv", "w"
) as outfile:
    for line in infile.read().split("\n"):
        try:
            outfile.write(
                AFTER_PREFIX
                + line[line.index(BEFORE_PREFIX) + len(BEFORE_PREFIX) :]
                + "\n"
            )
        except:
            print(line)
