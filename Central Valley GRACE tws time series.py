import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

df=pd.read_csv('C:\\Users\\user\\Desktop\\Rundle Research\\InSAR Stuff\\GRACE Central Valley\\timeSeries.water_equivalent_thickness_–_land_(grace,_grace-fo_jpl).200204-202207.35.3602N_121.6828W_37.5539N_121.6828W.csv')
times=[]
tws_data=[]

number_of_rows=df.shape[0]

for i in range(0,number_of_rows):
    times.append(df['DateTime'][i])
    tws_data.append(df['Water Equivalent Thickness – Land (GRACE, GRACE-FO JPL)'][i])
    
datetimes=[]
for i in range(0,number_of_rows):
    datetimes.append(datetime.datetime.strptime(times[i], "%Y-%m-%d %H:%M:%S").date())
    
    
plt.figure(figsize=(9.5,5))
plt.rcParams.update({'font.size': 15})
plt.plot(datetimes,tws_data,color='blue')
plt.xticks(rotation=45)

X=np.linspace(1,len(tws_data),len(tws_data))
p=np.polyfit(X,tws_data,1)
first_value=p[0]*X[0]+p[1]
last_value=p[0]*X[len(X)-1]+p[1]

line_fit_values=[]
line_fit_values.append(first_value)
line_fit_values.append(last_value)
line_fit_times=[]
line_fit_times.append(datetimes[0])
line_fit_times.append(datetimes[len(datetimes)-1])

plt.plot(line_fit_times,line_fit_values,color='red',label='y = %.3f t + %.3f' %(p[0],p[1]))
plt.legend(loc='upper right')
plt.ylabel('Water Equivalent Thickness (cm)')

#plt.savefig('terrestrial water.pdf')
plt.show()







