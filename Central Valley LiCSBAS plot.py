import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tifffile
from osgeo import gdal
import pyproj

#USE DPI COMMAND IN PLT.FIGURE TO MAKE FIGURES MORE DETAILED RATHER THAN
#MAKING THE WHOLE FIGURE BIGGER.
vel_filt_mskd=tifffile.imread('C:\\Users\\user\\Desktop\\Rundle Research\\Research Images\\Subsidence in central valley (potential paper)\\LiCSBAS results\\vel.filt.mskd.geo.tif')

height=np.shape(vel_filt_mskd)[0]
width=np.shape(vel_filt_mskd)[1]

###
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
###

fig=plt.figure(figsize=(15,17),dpi=100)
matplotlib.rcParams.update({'font.size': 23})
plt.imshow(vel_filt_mskd,cmap='winter',vmin=-53,vmax=10,extent=[-122.02211,-118.49111,35.66411,39.01611])
im_ratio = vel_filt_mskd.shape[0]/vel_filt_mskd.shape[1]
cbar=plt.colorbar(fraction=0.045*im_ratio)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
plt.grid(linewidth=0.25,color='gray')
plt.savefig('LiCSBAS velocity map.pdf')
plt.show()







