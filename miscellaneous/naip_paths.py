import pandas as pd

df = pd.read_csv(
    "/Users/charlesohanlon/Development/dataset_gen/NAIP2021_TileIndex_Seamlines_UTMZone12.csv"
)

PATH = "C:/Users/Administrator/Documents/ArcGIS/Projects/AI_4_SAR/arcgis-for-mpc-main/arcgis-for-mpc-main/AMPC_Resources/ACS_Files/esrims_pc_naip.acs/v002"

with open("mt_paths_only.csv", "w") as output:
    for _, row in df.iterrows():
        if pd.isna(row["ST"]):
            continue
        state = "mt"
        year = "2021"
        coords = row["Folder"]
        file_name = row["FileName"]
        res = f"{state}_060cm_{year}"

        truncate_idx = file_name[::-1].index("_") + 1
        file_name = file_name[:-truncate_idx] + ".tif"

        output.write(f"{PATH}/{state}/{year}/{res}/{coords}/{file_name},\n")
