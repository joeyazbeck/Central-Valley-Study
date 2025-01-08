import netCDF4 as nc
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

filepath='C:\\Users\\user\\Desktop\\Rundle Research\\InSAR Stuff\\GRACE Central Valley\\g4.areaAvgTimeSeries.GLDAS_CLSM025_DA1_D_2_2_GWS_tavg.20030201-20220630.121W_35N_118W_37N.nc'
ds=nc.Dataset(filepath)

#print(ds)

gws_data=ds['GLDAS_CLSM025_DA1_D_2_2_GWS_tavg'][:]
time_data=ds['dataday'][:]


petunia=str(time_data[0])
petunia=datetime.strptime(petunia,'%Y%j')

time_data_datetime=[]

for i in range(0,len(time_data)):
    petunia=str(time_data[i])
    time_data_datetime.append(datetime.strptime(petunia,'%Y%j'))

plt.figure(figsize=(9.5,5))
plt.rcParams.update({'font.size': 15})
plt.plot(time_data_datetime,gws_data,color='blue')
plt.xticks(rotation='45')

X=np.linspace(1,len(gws_data),len(gws_data))
p=np.polyfit(X,gws_data,1)
first_value=p[0]*X[0]+p[1]
last_value=p[0]*X[len(X)-1]+p[1]
line_fit_values=[]
line_fit_values.append(first_value)
line_fit_values.append(last_value)
line_fit_times=[]
line_fit_times.append(time_data_datetime[0])
line_fit_times.append(time_data_datetime[len(time_data_datetime)-1])

plt.plot(line_fit_times,line_fit_values,color='red',label='y = %.3f t + %.3f' %(p[0],p[1]))
plt.legend(loc='upper right')
plt.ylabel('Groundwater Storage (mm)')

plt.savefig('groundwater.pdf')
plt.show()






