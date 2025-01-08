import matplotlib.pyplot as plt
import tifffile
import numpy as np
import skimage.measure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
from datetime import datetime
from osgeo import gdal
import pyproj

#Code to produce principal components out of LOS tif images
#Just indicate filepath where the images are

start=datetime.now()

list_of_images=[]
# list_of_images_trimmed=[]
list_of_images_coarse_grained=[]

filepaths=glob.glob('D:\\InSAR Stuff\\Subsidence in Central Valley\\*tif')
number_of_images=len(filepaths)

for i in range(0,number_of_images):
    list_of_images.append(tifffile.imread(filepaths[i]))

#Have to get min and max of lat and lon to plot
dataset=gdal.Open(filepaths[0])
height=np.shape(list_of_images[0])[0]
width=np.shape(list_of_images[0])[1]
lat=np.zeros((height,width))
lon=np.zeros((height,width))

print(height)
print(width)

dsr=dataset.GetSpatialRef()
gt=dataset.GetGeoTransform()
dstproj4=dsr.ExportToProj4()
ct2=pyproj.Proj(dstproj4)
for i in range (0, height):
    for j in range(0, width):
        x = gt[0] + j * gt[1] + i * gt[2]
        y = gt[3] + j * gt[4] + i * gt[5]
        lon[i][j], lat[i][j] = ct2(x, y, inverse=True)
        
print(datetime.now() - start)

#Now, I will coarse-grain the trimmed images.
#I will take the average of a bunch of local blocks in the image to get it down to a manageable size.
#local blocks have a size of 30x30

downsample=30
for k in range(0,number_of_images):
    list_of_images_coarse_grained.append(skimage.measure.block_reduce(list_of_images[k],(downsample,downsample),np.mean))
   

#Now, before starting PCA, the images have to be the same size exactly.
#The idea is to just trim the rows until the minimum row from the image set
#is reached, and then do the same for the columns.

row_list=[]
column_list=[]
for k in range(0,number_of_images):
    row_list.append(np.shape(list_of_images_coarse_grained[k])[0])
    column_list.append(np.shape(list_of_images_coarse_grained[k])[1])
min_row=np.min(row_list)
min_column=np.min(column_list)

for k in range(0,number_of_images):
    height = np.shape(list_of_images_coarse_grained[k])[0]
    width = np.shape(list_of_images_coarse_grained[k])[1]
    if height != min_row:
        list_of_images_coarse_grained[k]=np.delete(list_of_images_coarse_grained[k],np.arange(height-min_row),0)
    if width != min_column:
        list_of_images_coarse_grained[k]=np.delete(list_of_images_coarse_grained[k],np.arange(width-min_column),1)
        
height = min_row
width = min_column
number_of_pixels = height*width

#I will do PCA analysis now using Sklearn to find principal components

#Creating data matrix with images as row vectors
X=np.zeros((number_of_images,number_of_pixels))

for i in range(0,number_of_images):
    X[i,:]=np.ndarray.flatten(list_of_images_coarse_grained[i])

#I can always use scaler.inverse_transform to transform the scaled matrix back to the original scaling

pca=PCA(0.92)
pca.fit(X)
#pca.components_ are the eigenfaces and also the principal components!
principal_components=pca.components_
# for i in range(0,len(principal_components)):
#     plt.subplot(1,len(principal_components),i+1)
#     plt.imshow(np.reshape(principal_components[i],(height,width)),extent=[np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)])
#     cbar=plt.colorbar()
#     cbar.formatter.set_powerlimits((0,0))
#     cbar.set_label('Line of Sight Diplacement (m)',rotation=270,size=7)
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Principal Component %d' %(i+1))
#     plt.show()

image1=np.reshape(principal_components[0],(height,width))
image2=np.reshape(principal_components[1],(height,width))

vmin1=np.amin(image1)
vmin2=np.amin(image2)
vmin=np.amin([vmin1,vmin2])
vmax1=np.amax(image1)
vmax2=np.amax(image2)
vmax=np.amax([vmax2,vmax2])

fig=plt.figure(figsize=(9.5,3))
fig.add_subplot(1,2,1)
plt.imshow(image1,cmap='seismic',vmin=vmin,vmax=-vmin,extent=[np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)])
im_ratio = image1.shape[0]/image1.shape[1]
cbar1=plt.colorbar(fraction=0.0377*im_ratio)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Principal Component 1')

plt.subplot(1,2,2)
plt.imshow(image2,cmap='seismic',vmin=vmin,vmax=-vmin,extent=[np.amin(lon),np.amax(lon),np.amin(lat),np.amax(lat)])
im_ratio = image2.shape[0]/image2.shape[1]
cbar2=plt.colorbar(fraction=0.0377*im_ratio)
#cbar.formatter.set_powerlimits((0,0))
#cbar.set_label('Line of Sight Displacement (m)',rotation=270,size=7)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Principal Component 2')

plt.subplots_adjust(wspace=0.4)

plt.savefig('principal_components.pdf')
plt.show()



print(pca.explained_variance_ratio_)

X_dim_reduced=pca.transform(X) #This projects the data onto the principal components
print(np.shape(X_dim_reduced)) #It will have shape (number of images,number of principal components)
#So, I can now use X_scaled_dim_reduced in a ML algorithm, and it will be very efficient
#since we reduced the dimensionality of the dataset from 10179 to just 2 (if i take PCA(0.8))!!!

#Now, I want to plot a histogram of explained variance ratios.
# plt.figure(figsize=(10,7))
# plt.bar(np.arange(1,len(principal_components)+1),pca.explained_variance_ratio_*100,color='red')
# plt.title('Explained Variance vs Principal Components',fontsize=20)
# plt.xlabel('Principal Components',fontsize=15)
# plt.ylabel('Percentage of explained variance',fontsize=15)
# plt.yticks(fontsize=10)
# #making sure I see all the numbers on the x-axis
# plt.xticks(np.arange(1,len(principal_components)+1),np.arange(1,len(principal_components)+1),fontsize=10)
# plt.savefig('explained_variance.pdf')
# plt.show()

print(datetime.now()-start)





