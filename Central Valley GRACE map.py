import numpy as np
import matplotlib.pyplot as plt
import tifffile
from osgeo import gdal
import pyproj
from numpy import ma


time_averaged_map=tifffile.imread('C:\\Users\\user\\Desktop\\Rundle Research\\InSAR Stuff\\GRACE Central Valley\\GIOVANNI-g4.timeAvgMap.GLDAS_CLSM025_DA1_D_2_2_GWS_tavg.20030201-20220630.121W_35N_118W_37N.tif')

height=np.shape(time_averaged_map)[0]
width=np.shape(time_averaged_map)[1]

lon=np.zeros((height,width))
lat=np.zeros((height,width))

dataset=gdal.Open('C:\\Users\\user\\Desktop\\Rundle Research\\InSAR Stuff\\GRACE Central Valley\\GIOVANNI-g4.timeAvgMap.GLDAS_CLSM025_DA1_D_2_2_GWS_tavg.20030201-20220630.121W_35N_118W_37N.tif')
dsr=dataset.GetSpatialRef()
gt=dataset.GetGeoTransform()
dstproj4=dsr.ExportToProj4()
ct2=pyproj.Proj(dstproj4)
for i in range (0, height):
    for j in range(0, width):
        x = gt[0] + j * gt[1] + i * gt[2]
        y = gt[3] + j * gt[4] + i * gt[5]
        lon[i][j], lat[i][j] = ct2(x, y, inverse=True)

time_averaged_map_copy=time_averaged_map
time_averaged_map=ma.masked_where(time_averaged_map==-9999,time_averaged_map)

smoothed=plt.contourf(lon,lat,time_averaged_map,cmap='Spectral',vmin=375.704711914,vmax=857.241699219)
plt.rcParams.update({'font.size': 10})
cbar=plt.colorbar()
cbar.set_label('Time Averaged Groundwater Storage (mm)',labelpad=20,rotation=270)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
plt.savefig('GRACE map.pdf')
plt.show()



# plt.imshow(time_averaged_map,cmap='Spectral',vmin=375.704711914,vmax=857.241699219,extent=[np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)])
# plt.colorbar()

