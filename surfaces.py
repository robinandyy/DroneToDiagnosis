# imports
import arcpy
from arcpy import env
from arcpy.sa import *
import numpy as np
import torch
import pandas as pd

# enable overwriting, because mistakes happen :)
env.overwriteOutput = True



# declare workspace
env.workspace = "c:/Users/Robin/Desktop/Vineyard_Dataset/0_V1"

# designate spatial reference
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32629)
sr = arcpy.SpatialReference(32629)

arcpy.management.CreateFileGDB(env.workspace, "final_outputs")

DSM = Raster("dsmclip")
DTM = Raster("dtmclip")

# create canopy height model by subtracting Digital Terrain Model from Digital Surface Model
CHM = DSM - DTM 
CHM.save("c:/Users/Robin/Desktop/Vineyard_Dataset/0_V1/CHM_preadjustments.tif")

# some small negative values (anomalies/errors) appeared, so change them to null
CHM_masked = SetNull("CHM.tif", "CHM.tif", "VALUE < 0")
CHM_masked.save("c:/Users/Robin/Desktop/Vineyard_Dataset/0_V1/CHM.tif")



# NDRE = (NIR – RedEdge)/(NIR + RedEdge)
# NIR is band 4, RedEdge is 5
# save NDRE raster
NDRE = RasterCalculator(["orthoclipc5.tif", "orthoclipc4.tif"],
                                       ["x", "y"], "(x-y)/(x+y)")
NDRE.save(env.workspace + "/NDRE.tif")


# remove edge cases/noise that is below -1 by setting values to null and saving raster
NDRE_masked = SetNull("NDRE.tif", "NDRE.tif", "VALUE < -1")
NDRE_masked.save(env.workspace + "/NDRE_min.tif")



# remove edge cases/noise that is above 1 and save
NDRE_masked2 = SetNull("NDRE_min.tif", "NDRE_min.tif", "VALUE > 1")
NDRE_masked2.save(env.workspace + "/NDRE_adjusted.tif")



# mask NDRE to 0.5 and greater to isolate grapevine NDRE values and ignore soil + weeds values
CHM_raster = "CHM.tif"
NDRE_raster = "NDRE_adjusted.tif"
NDRE_grapevine = SetNull(CHM_raster, NDRE_raster, "VALUE < 0.5")
NDRE_grapevine.save(env.workspace + "/NDRE_grapevines.tif")



# create an estimate of leaf area index by combining NDRE and CHM, then save
# LAI = NDRE x CHM
LAI = RasterCalculator(["NDRE_grapevines.tif", "CHM.tif"], 
                       ["a","b"], "a*b")
LAI.save(env.workspace + "/LAI.tif")

# mask LAI similarly to NDRE and save
LAI_masked = SetNull(CHM_raster, "LAI.tif", "VALUE < 0.5")
LAI_masked.save(env.workspace + "/LAI_masked.tif")


# get coordinates for the grid system
trunks = "C:/Users/Robin/Downloads/Trunk_locations/Trunks.shp"


    


        

        





# create fishnet

# input data is in NAD 1983 UTM Zone 11N coordinate system
input_features = trunks

# output data
output_feature_class = r"trunks_Project.shp"

# change the coordinate system of the plant locations to align with the project
arcpy.management.Project(input_features, output_feature_class, sr)


# find origin coordinates based on point with the lowest y value
with arcpy.da.SearchCursor(output_feature_class,["SHAPE@XY"]) as cursor:
    minimum = 100000000000000
    leftmost = 1000000000000000000000000
    for row in cursor:
        if row[0][0] < leftmost:
            print(f"minimum updated to: {row[0][1]}")
            minimum = row[0][1]
            leftmost = row[0][0]
            x_origin = row[0][0]
            x_origin -= 2
            y_origin = row[0][1]

# find y axis point based on highest point (top left corner is northmost point in the field)
with arcpy.da.SearchCursor(output_feature_class,["SHAPE@XY"]) as cursor:
    maximum = 0
    for row in cursor:
        if row[0][1] > maximum:
            maximum = row[0][1]
            x_y_axis_coord = row[0][0]
            y_y_axis_coord = row[0][1]
            y_y_axis_coord += 2


# find rightmost point for opposite corner (rightmost point is opposite corner in the field)
with arcpy.da.SearchCursor(output_feature_class,["SHAPE@XY"]) as cursor:
    rightmost = 0
    for row in cursor:
        if row[0][0] > rightmost:
            rightmost = row[0][0]
            x_opposite_corner = row[0][0]
            y_opposite_corner = row[0][1]


arcpy.management.CreateFishnet(out_feature_class="fishnet.shp", 
                               origin_coord= str(x_origin) + " " + str(y_origin),
                                y_axis_coord= str(x_y_axis_coord) + " " + str(y_y_axis_coord),
                                cell_width=3.25,
                                cell_height=2.5,
                                number_rows=51,
                                number_columns=3,
                                geometry_type="POLYGON") 


# create NDRE zonal statistics table
ZonalStatisticsAsTable(in_zone_data="fishnet.shp",
                       zone_field="FID",
                       in_value_raster="NDRE_grapevines.tif",
                       out_table="NDRE_mean_values.dbf",
                       statistics_type="MEAN")


# create LAI zonal statistics table
ZonalStatisticsAsTable(in_zone_data="fishnet.shp",
                       zone_field="FID",
                       in_value_raster="LAI_masked.tif",
                       out_table="LAI_mean_values.dbf",
                       statistics_type="MEAN")


# create CHM zonal statistics table
ZonalStatisticsAsTable(in_zone_data="fishnet.shp",
                       zone_field="FID",
                       in_value_raster=CHM_raster,
                       out_table="CHM_mean_values.dbf",
                       statistics_type="MEAN")


# create DTM zonal statistics table
ZonalStatisticsAsTable(in_zone_data="fishnet.shp",
                       zone_field="FID",
                       in_value_raster=DTM,
                       out_table="DTM_mean_values.dbf",
                       statistics_type="MEAN")


# add NDRE values to a set
NDRE_id = set()
count = 0


joins = ["NDRE_mean_values.dbf", "CHM_mean_values.dbf", "LAI_mean_values.dbf", "DTM_mean_values.dbf"]
words = ["NDRE", "CHM", "LAI", "DTM"]


NDRE_id = set()
count = 0

# add all NDRE FIDs to a set NDRE_id
with arcpy.da.SearchCursor("NDRE_mean_values.dbf","FID_") as cursor:
    for row in cursor:
        NDRE_id.add(row[0])
        count += 1

for i in range(len(joins)):
    
    # if CHM, LAI, or DTM has FIDs that the NDRE table does not have, remove them
    with arcpy.da.UpdateCursor(joins[i], "FID_") as cursor:
        print(f"**************moving on to DTM*********************")
        for row in cursor:
            if row[0] not in NDRE_id:
                cursor.deleteRow()
        
    out_table_name = words[i] + "Export"
    arcpy.conversion.ExportTable(in_table=joins[i], 
                                out_table=out_table_name + '.dbf')
    
    # list field names
    fields = [f.name for f in arcpy.ListFields(out_table_name + '.dbf')]
    print(f"*************************Fields in {out_table_name}: {fields}**********************")
    
    

    arcpy.management.JoinField(in_data="fishnet_label", in_field="FID", join_table=out_table_name, join_field="FID_", fields="MEAN")
    print(f"**************************join complete****************************************")



with arcpy.da.UpdateCursor("fishnet_label", "MEAN") as cursor:
    for row in cursor:
        if row[0] == 0:
            cursor.deleteRow()



arcpy.analysis.SpatialJoin(target_features="fishnet",
                        join_features="Botrytis_clusters", 
                        out_feature_class="spatial_join_fishnet", 
                        join_operation="JOIN_ONE_TO_MANY")




arcpy.management.AddField(in_table="fishnet_label",
                        field_name="BBR", 
                        field_type="SHORT")

# disease = set()
# with arcpy.da.SearchCursor("spatial_join_fishnet",["FID","Join_Count"]) as cursor:
#     for row in cursor:
#         if row[1] > 0:
#             disease.add(row[0])

# create fishnet labels (centroids) with join count indicating the number of diseases within each grid
arcpy.analysis.SpatialJoin(target_features="fishnet_label", 
                           join_features="spatial_join_fishnet", 
                           out_feature_class="fishnet_label_join_count", 
                           join_operation="JOIN_ONE_TO_ONE", 
                           match_option="COMPLETELY_WITHIN")


# if disease count per centroid is greater than 0, mark as 1 otherwise mark as 0
with arcpy.da.UpdateCursor("fishnet_label_join_count.shp", ["Join_Count", "BBR"]) as cursor:
    for row in cursor:
        if row[0] > 0:
            row[1] = 1
        else:
            row[1] = 0

        cursor.updateRow(row)


# clean up data by removing irrelevant fields
arcpy.management.DeleteField(in_table="fishnet_label_join_count.shp", drop_field=["Join_Count",
                                                                                "TARGET_FID", 
                                                                                "Id", 
                                                                                "Join_Cou_1", 
                                                                                "Target_F_1", 
                                                                                "Id_1", 
                                                                                "ID_12", 
                                                                                "ROW", 
                                                                                "LABEL",
                                                                                "JOIN_FID"])
# export clean data as csv
arcpy.conversion.ExportTable(in_table="fishnet_label_join_count.shp", 
                            out_table="BBR_info.csv")


X = np.genfromtxt(fname=env.workspace + "/BBR_info.csv",
                           dtype=np.float32, 
                           delimiter=',', 
                           skip_header=1,
                           usecols=(0,1,2,3))


y = np.genfromtxt(fname=env.workspace + "/BBR_info.csv",
                           dtype=np.float32, 
                           delimiter=',', 
                           skip_header=1,
                           usecols=(4))


X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

print(X.shape, y.shape, X.ndim, y.ndim)
print(X[:10], y[:10])
print(len(X), len(y))



# if X and y are torch tensors:
X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)

df_train = pd.DataFrame(
    X_np,
    columns=['NDRE','CHM','LAI','DTM']
)
df_train['Disease'] = y_np.ravel()





csv_file_path = "vineyard_data.csv"  # or any filename you like
df_train.to_csv(csv_file_path, index=False)

print(f"CSV file '{csv_file_path}' has been created successfully.")




print("succeeded!")