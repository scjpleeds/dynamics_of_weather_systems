"""
reated on Sun Feb 17 17:50:59 2019
Script to load in ERA-interim data. Calculate windspeed, Coriolis force and Relative vorticity.
Plot horizontal windspeed and vertical cross-sections of windspeed
Plot relative vorticity
Save all figures created for all days, months, years and times specified
@author: Sam Clarke
"""


#This is just to load in modules required in order to run the script


import cartopy.feature as cfeature ## for lakes/borders et
countries_50m = cfeature.NaturalEarthFeature('cultural','admin_0_countries','50m',edgecolor='k',facecolor='none')
import cartopy.crs as ccrs
import numpy as np
from netCDF4 import Dataset  
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as mticker
import matplotlib as mpl
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
plt.switch_backend('agg')




# In[]: Specify the path out for all figures that will be saved i.e. the path to your directory on linux or the directory you wish to 
# save the figurepathout ="/nfs/see-fs-02_users/username/SOEEmodule"
pathout ="/nfs/see-fs-01_users/scjp/winter_storm_friedhelm/plots"

# In[]: Specify the year, month, hours and days for which you wish to plot data for
year = "2011"

mon= "12"

# if you want to output files for all times and all days then take the # away from the below two lines. 
#hours = ["00","06","12","18"] 
#days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

# Just want to output for one time and day...then use e.g.:
hours = ["00","06","12","18"]
days = ["06","07","08"]

# In[]: Loop over all years, months, days and hours specified above to load in the data and then plot horizontal windspeed
# vertical cross-sections of windspeed, calculate relative vorticity and plot relative vorticity. 
for d in days:
    for time in hours: 
        erai_data = Dataset('/nfs/a321/datasets/ERA-interim/'+str(year)+'/ap/ggap' + str(year)+''+str(mon)+''+str(d)+''+str(time)+'00.nc','r')
        erai_data_surface = Dataset('/nfs/a321/datasets/ERA-interim/'+str(year)+'/as/ggas' + str(year)+''+str(mon)+''+str(d)+''+str(time)+'00.nc','r')
 
        
        
        # In[]: Extracting the relevant data from the ERA-interim files loaded

        
        lat = erai_data.variables['latitude'][:] #latitudes
        
        lon = erai_data.variables['longitude'][:] #longitudes
        
        
        p   = erai_data.variables['p'][:] #pressure
        t   = erai_data.variables['t'][:] #time


        MSLP = erai_data_surface.variables['MSL'][:] #mean sea level pressure
        u   = erai_data.variables['U'][:] #u winds
        v   = erai_data.variables['V'][:] #v winds
        w   = erai_data.variables['W'][:] #w winds
        Z   = erai_data.variables['Z'][:] #geopotential height 
        T   = erai_data.variables['T'][:] #tempertature
        D   = erai_data.variables['D'][:] #divergence
        PV  = erai_data.variables['PV'][:] #potential vorticity 
        Q   = erai_data.variables['Q'][:] #specific humidity 
        CLWC= erai_data.variables['CLWC'][:] #cloud liquid water content 
        CIWC= erai_data.variables['CIWC'][:] #cloud ice water content
        erai_data.close()
        erai_data_surface.close()
        
        
        # In[]: Longitude values are loaded in as 0 to 360 degrees. Modify so 
        # that they go -180 to 180 degrees as easier to find subsections this way. 
        lon = ((lon - 180) % 360) - 180
        indcs = np.argsort(lon)
        
        #Make the lon, u and v arrays know that they too need to have their longitudes between -180 to 180 degrees 
        # instread of 0 to 360 degrees. We don't need to do this for p as it is just a 1D array
        u = u[:,:,:,indcs]
        v = v[:,:,:,indcs]
        w = w[:,:,:,indcs]
        Z = Z[:,:,:,indcs] 
        T = T[:,:,:,indcs]
        D = D[:,:,:,indcs]
        PV = PV[:,:,:,indcs]
        Q = Q[:,:,:,indcs]
        CLWC=CLWC[:,:,:,indcs]
        CIWC=CIWC[:,:,:,indcs]
        MSLP = MSLP[:,:,:,indcs]
        lon = lon[indcs]
      
 
        
        # In[]: 
        #define the pressure level you wish to use to plot the horizontal windspeed and relative vorticity plots
        p_lev = 850 # this indicates 500hPa level
        p_top= 250
        p_level = '850' # this is used to label the figures which are saved at the end
        
        # Find at which index in the array for pressure the p_lev specified is located
        pressure = np.where(p==p_lev) 
        top = np.where(p==p_top)
        # Take time/pressure slice
        u_slice = u[0,pressure,:,:]
        v_slice = v[0,pressure,:,:]
        w_slice = w[0,pressure,:,:]
        Z_slice = Z[0,pressure,:,:]
        T_slice = T[0,pressure,:,:]
        D_slice = D[0,pressure,:,:]
        PV_slice = PV[0,pressure,:,:]
        PV_top = PV[0,top,:,:]
        Q_slice = Q[0,pressure,:,:]
        CLWC_slice = CLWC[0,pressure,:,:]
        CIWC_slice = CIWC[0,pressure,:,:]
        MSLP_slice = MSLP[0,0,:,:]
        #Make arrays 2D so now just an array of u and v winds for pressure level p_lev at all latitude and longitudes
        u_slice = u_slice[0,0,:,:]
        v_slice=v_slice[0,0,:,:]
        w_slice = w_slice[0,0,:,:]
        Z_slice = Z_slice[0,0,:,:]
        T_slice = T_slice[0,0,:,:]
        D_slice = D_slice[0,0,:,:]
        PV_slice = PV_slice[0,0,:,:]
        Q_slice = Q_slice[0,0,:,:]
        CLWC_slice = CLWC_slice[0,0,:,:]
        CIWC_slice = CIWC_slice[0,0,:,:]
        PV_top = PV_top[0,0,:,:]

        
       
        # In[]: Calculate windspeed for one pressure level specified above
        wspeed=np.sqrt((u_slice**2)+(v_slice**2))

        #Calculate windspeed for all pressure levels
        
        wspeed_all = np.sqrt((u**2)+(v**2))
        wspeed_all= wspeed_all[0,:,:,:]
         
        # In[]: Coriolis force calculation
        
        earthrot = 7.29*10**-5
        pi       = 3.14159
    

        Coriolis = 2*earthrot*np.sin(lat*pi/180)
             
        print(Coriolis.shape) 
        
        # In[42]: Calculate relative vorticity
       
        # Define haversine formula to convert lon/lat (degrees) to metres
        
        
        
        def haversine( lon1, lat1, lon2, lat2):
            R = 6371e3 # metres
            phi1 = math.radians(lat1)
            phi2 = math.radians(lat2)
            dphi = phi2-phi1
            
            lambda1 = math.radians(lon1)
            lambda2 = math.radians(lon2)
            dlambda = lambda2-lambda1
        
            a = math.sin(dphi/2) * math.sin(dphi/2) + math.cos(phi1) * math.cos(phi1) * math.sin(dlambda/2) * math.sin(dlambda/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
            d = R * c
            
            return d
        
     
        
        
        # ### Calculate dx,dy across points
        # 
        
        # In[118]: Create empty arrays for dx and dy 
        
        
        dx_array = np.empty( u_slice.shape)# 
        dy_array = np.empty( u_slice.shape)# 
        
        
        # In[119]:
        
             
        for i in range( 0, lat.size):
            for j in range( 0, lon.size):
                
                # Find index +/- 1, remembering that we need to wrap around the earth at
                # the extent of the coordinates
                
                low_index = j-1
                high_index = j+1
                
                if high_index == lon.size:
                    high_index = -1
                
                dx_array[i,j] = haversine( lon[low_index], lat[i], lon[high_index], lat[i]) 
                
        
        
        # In[120]:
        
        
                
        for j in range( 0, lon.size):
            for i in range( 0, lat.size):
                
                # Find index +/- 1, remembering that we need to wrap around the earth at
                # the extent of the coordinates
                
                low_index = i-1
                high_index = i+1
                
                if high_index == lat.size:
                    high_index -= 1
                    dy_array[i,j] = haversine( lon[j], lat[low_index], lon[j], lat[high_index]) 
                elif low_index == -1:
                    low_index=0
                    dy_array[i,j] = haversine( lon[j], lat[low_index], lon[j], lat[high_index]) 
                    
                else:
                    dy_array[i,j] = haversine( lon[j], lat[low_index], lon[j], lat[high_index])   
        
        # Find du and dv
        # As above, we need to find du and dv using centred difference
        
        # In[123]: Define an empty array to put the calculations of du an dv into
        
        
        du_array = np.empty( u_slice.shape)
        dv_array = np.empty( u_slice.shape)
        dZu_array = np.empty(Z_slice.shape) 
        dZv_array = np.empty(Z_slice.shape)
        dTu_array = np.empty(T_slice.shape)
        dTv_array = np.empty(T_slice.shape) 
        
        # In[124]: Check the shape of u_slice to check it worked
         
        
        # In[126]: Calculate the central finite difference for dv
        #(U(i+1) + u(i-1))/2*dx  central differening
        
        for i in range( 0, lat.size):
             for j in range( 0, lon.size):
                
                # Find index +/- 1, remembering that we need to wrap around the earth at
                # the extent of the coordinates
                
                low_index = j-1
                high_index = j+1
                
                if high_index == lon.size:
                    high_index = -1
                
                dv_array[i,j] = (v_slice[ i, high_index] - v_slice[ i, low_index])
                dZu_array[i,j] = (Z_slice[i,high_index]-Z_slice[i,low_index])
                dTu_array[i,j] = (T_slice[i,high_index]-T_slice[i,low_index])
        # In[128]: Calculate the central finite difference for du
        
        for j in range( 0, lon.size):
            for i in range( 0, lat.size):
            
            # Find index +/- 1, remembering that we need to wrap around the earth at
            # the extent of the coordinates
            
                low_index = i-1
                high_index = i+1
                
                if high_index == lat.size:
                    high_index -= 1
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                elif low_index==-1:
                    low_index=0
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                else:
                    du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                    dZv_array[i,j] = (Z_slice[high_index,j]-Z_slice[low_index,j])
                    dTv_array[i,j] = (T_slice[high_index,j]-T_slice[low_index,j])
       
        # In[]: Calculate relative vorticity
        
        #The the haversine formula is always a positive distance so we need to do dv/dx + du/dy 
        #to calculate relative vorticity instead of dv/dx - du/dy as is the usual equation
        
        vort = dv_array / dx_array + du_array / dy_array   
        
        #geostrophic wind
        ug = (1/Coriolis)*(dZv_array.T/dy_array.T)
        vg = (1/Coriolis)*(dZu_array.T/dx_array.T)
        
        ug = ug.T
        vg = vg.T

        #ageostrophic wind
        uag = u_slice - ug
        vag = v_slice - vg
        
        #potential temperature
        p0 = 1000
        theta = T_slice*(p0/p_lev)**(0.286)


        #Q vector
        dug_array = np.empty(ug.shape)
        dvg_array = np.empty(vg.shape)
        for j in range( 0, lon.size):
            for i in range( 0, lat.size):
            
            # Find index +/- 1, remembering that we need to wrap around the earth at
            # the extent of the coordinates
            
                low_index = i-1
                high_index = i+1
                
                if high_index == lat.size:
                    high_index -= 1
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                elif low_index==-1:
                    low_index=0
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                else:
                    dvg_array[i,j] = (vg[ high_index, j] - vg[ low_index, j]) 

        for i in range( 0, lat.size):
             for j in range( 0, lon.size):
                
                # Find index +/- 1, remembering that we need to wrap around the earth at
                # the extent of the coordinates
                
                low_index = j-1
                high_index = j+1
                
                if high_index == lon.size:
                    high_index = -1
                
                dug_array[i,j] = (ug[ i, high_index] - ug[ i, low_index])
        
        Rgas = 287
        Qx = (dug_array/dx_array)*(dTu_array/dx_array) + (dvg_array/dx_array)*(dTv_array/dy_array)
        Qy = (dug_array/dy_array)*(dTu_array/dx_array) + (dvg_array/dy_array)*(dTv_array/dy_array)

        Qx = (-Rgas/p_lev)*Qx
        Qy = (-Rgas/p_lev)*Qy
        
        dQx_array = np.empty(Qx.shape)
        dQy_array = np.empty(Qy.shape)
        for j in range( 0, lon.size):
            for i in range( 0, lat.size):
            
            # Find index +/- 1, remembering that we need to wrap around the earth at
            # the extent of the coordinates
            
                low_index = i-1
                high_index = i+1
                
                if high_index == lat.size:
                    high_index -= 1
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                elif low_index==-1:
                    low_index=0
                    #du_array[i,j] = (u_slice[ high_index, j] - u_slice[ low_index, j])
                else:
                    dQy_array[i,j] = (Qy[ high_index, j] - Qy[ low_index, j]) 

        for i in range( 0, lat.size):
             for j in range( 0, lon.size):
                
                # Find index +/- 1, remembering that we need to wrap around the earth at
                # the extent of the coordinates
                
                low_index = j-1
                high_index = j+1
                
                if high_index == lon.size:
                    high_index = -1
                
                dQx_array[i,j] = (Qx[ i, high_index] - Qx[ i, low_index])

        divQ = dQx_array/dx_array + dQy_array/dy_array
     
        # Change the units of vorticity 
        vorticity = vort *100000 
        
        
        print "now onto producing the plots!!!!!"
        
        # In[] : Specify the subsection of the Global domain which you wish to plot: latitude and longitude required.
        # If you wish to plot for the whole Globe then comment this section out using #
        
        #Specify min and max values of latitude and longitude 
        lat_min = 30
        lat_max =80
        lon_min = -40
        lon_max = 30
        
        # Find where in the longitude array the values are between lon_min and lon_max 
        lons = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        
        # Gives an array which is the subdomain of the longitude values 
        lon_subset = lon[lons]
        
        #Find the values in the longitude array at which this subsection range of longitudes lies between
        lon_min_index = np.min(lons)  
        lon_max_index = np.max(lons) + 1
        #Repeat for latitude
        lats = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lat_subset = lat[lats]
        lat_min_index =np.min(lats) 
        lat_max_index = np.max(lats) + 1
        
        # Find the values of u and v for just the subdomain (and still only for one pressure level)
        u_subset=  u_slice[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        v_subset = v_slice[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        w_subset = w_slice[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        
        
        #In[]: Plot the windspeed with windbarbs to indicate wind direction

        #Specify manually the contour levels that will be plotted
        wlevels = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75]
        

        #Plot the windspeed
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.contourf(lon_subset, lat_subset, wspeed[lat_min_index:lat_max_index,lon_min_index:lon_max_index], levels=wlevels, transform=ccrs.PlateCarree(),cmap=plt.cm.rainbow)
        #Produce wind barbs
        widths = np.linspace(0, 2, lats.size)
        crs = ccrs.PlateCarree()
        transform = crs
        ax.quiver(lon_subset[::3], lat_subset[::3], u_subset[::3,::3], v_subset[::3,::3], transform=crs, pivot='middle', linewidths=widths)
        # Plot a colourbar and label 
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Wind speed (ms$^{-1}$)')
        #Add lines for countries 
        ax.add_feature(countries_50m, linewidth=1)
        #Add latitude and longitude grid
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        # This specifies that you only want to label the lat and long values at the bottom and to the left of the figure
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        # Manually specify where you want the tick marks and thus labels for latitude and longitude to be: is the same as the subdomain specified above
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
     
        plt.title("Windspeed")
        #Save the figures for all dates and times input at the start
        plt.savefig(pathout+'/ERA_interim_Windspeed_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        
        print 'plotted windspeed now onto cross-section'
        
        # In[]: Find the longitude value through which you wish to plot a vertical cross-section#
        #Function the nearest longitude from the ERA data to the longitude that you want to plot
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        #Specify the longitude value you wish to plot a cross-section through
        lon_cross_section = -20
        #Find the nearest longitude value to lon_cross_section value that is contained in the ERA-interim longitude values
        nearest = find_nearest(lon_subset,lon_cross_section)
        print nearest , 'nearest value'
        #Find the number within the longitude array at which the longitude you wish to plot a cross-section through is located
        lon_slice_index = np.where(lon_subset == nearest)[0]
        print lon_slice_index, 'lon_slice_index'
        
        
        #Create an array of windspeeds for all pressure levels, through the sub-section range of latitudes and 
        #through the longitude value just determined above
        windspeed_vert = wspeed_all[:,lat_min_index:lat_max_index,lon_slice_index]
        CLWC_vert = CLWC[0,:,lat_min_index:lat_max_index,lon_slice_index]
        CIWC_vert = CIWC[0,:,lat_min_index:lat_max_index,lon_slice_index]
        T_vert = T[0,:,lat_min_index:lat_max_index,lon_slice_index]
        Z_vert = Z[0,:,lat_min_index:lat_max_index,lon_slice_index]
        PV_vert = PV[0,:,lat_min_index:lat_max_index,lon_slice_index]
        Q_vert = Q[0,:,:lat_min_index:lat_max_index,lon_slice_index]
        D_vert = D[0,:,:lat_min_index:lat_max_index,lon_slice_index]
        
        print D_vert.shape
        
        # In[]: Plot vertical cross-section for windspeed
        w_vert = w[0,:,lat_min_index:lat_max_index,lon_slice_index]      
        #ug_vert = ug[0,:lat_min_index:lat_max_index,lon_slice_index]
        #vg_vert = vg[0,:lat_min_index:lat_max_index,lon_slice_index]
        ug_slice = ug[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        vg_slice = vg[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        uag_slice = uag[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        vag_slice = vag[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        #clevels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
        plt.contour(lon_subset, lat_subset,theta[lat_min_index:lat_max_index, lon_min_index:lon_max_index],10, transform=ccrs.PlateCarree(),colors='black')
        plt.quiver(lon_subset[::3],lat_subset[::3], uag_slice[::3,::3],vag_slice[::3,::3],transform=ccrs.PlateCarree(),linewidths=widths,color='blue',pivot='middle')
        #plt.quiver(lon_subset[::3],lat_subset[::3], uag_slice[::3,::3],vag_slice[::3,::3],transform=ccrs.PlateCarree(),color='red',pivot='middle')

        #cbar = plt.colorbar(orientation='horizontal')
        #cbar.outline.set_linewidth(0.5)
        #cbar.ax.tick_params(labelsize=10)
        #cbar.set_label('$m^2 s^{-2}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Geopotential Height Contours and ageostrophic Wind Vectors")
        plt.savefig(pathout+'/ERA_interim_theta_ageostrophic_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        #clevels = np.linspace(180,300,lats.size)
        fig = plt.figure(figsize=(10,10))
        clevels=np.linspace(0,1.5,lats.size)
        plt.gca().invert_yaxis()
        plt.contourf(lat_subset, p, PV_vert[0,:,:]*10**6,levels=clevels,cmap='Reds',extend='max')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$PVU$')
        plt.title("PV")
        #save the figures for all dates and times 
        plt.savefig(pathout+'/ERA_interim_PV_Cross_Section'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_Longitude_'+str(lon_cross_section)+'.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        #clevels=np.linspace(0,1.5,lats.size)
        plt.gca().invert_yaxis()
        plt.contourf(lat_subset, p, Q_vert[0,:,:]*10**6,cmap='Reds',extend='max')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$PVU$')
        plt.title("PV")
        #save the figures for all dates and times 
        plt.savefig(pathout+'/ERA_interim_Q_Cross_Section'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_Longitude_'+str(lon_cross_section)+'.png',format ='png', dpi=150, bbox_inches='tight')


        
        
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(0,1.5,lats.size)
        plt.contour(lon_subset, lat_subset,theta[lat_min_index:lat_max_index, lon_min_index:lon_max_index],10, transform=ccrs.PlateCarree(),colors='black')
        plt.contourf(lon_subset, lat_subset,PV_top[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**6, transform=ccrs.PlateCarree(),levels=clevels,cmap='Reds',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$10^{-6}$ PVU')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Potential Vorticity")
        plt.savefig(pathout+'/ERA_interim_potential_vorticity_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        Qx_subset = Qx[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        Qy_subset = Qy[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        ug_subset = ug[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        vg_subset = vg[lat_min_index:lat_max_index,lon_min_index:lon_max_index]



        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels=np.linspace(-15,15,lats.size)*10**-14
        widths = np.linspace(-10,10,lats.size)*10**5
        templevels=np.linspace(230,290,35)
        cont = plt.contourf(lon_subset, lat_subset,-2*divQ[lat_min_index:lat_max_index, lon_min_index:lon_max_index], transform=ccrs.PlateCarree(),levels=clevels,cmap='seismic',extend='both')
        #plt.quiver(lon_subset[::3],lat_subset[::3],ug_subset[::3,::3],vg_subset[::3,::3],color='black',transform=ccrs.PlateCarree(),linewidth=widths)
        ptemp = plt.contour(lon_subset,lat_subset,theta[lat_min_index:lat_max_index,lon_min_index:lon_max_index],transform=ccrs.PlateCarree(),levels=templevels,colors='black')
        plt.clabel(ptemp,inline=1,fontsize=10) 
        cbar = plt.colorbar(cont,orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$mkg^{-1}s^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("")
        plt.savefig(pathout+'/ERA_interim_divQ_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight') 

        
        
        #Plot the vertical cross-section
'''
        clevels = np.linspace(-1.5,1.5,50)
        fig = plt.figure(figsize=(10,10))
        plt.gca().invert_yaxis()
        plt.contourf(lat_subset, p, w_vert[0,:,:],levels=clevels,cmap='seismic',extend='max')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$ms^{-1}$')
        plt.title("Vertical Velocity")
        #save the figures for all dates and times 
        plt.savefig(pathout+'/ERA_interim_w_Cross_Section'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_Longitude_'+str(lon_cross_section)+'.png',format ='png', dpi=150, bbox_inches='tight')

        clevels=np.linspace(0,3,41)*10**-4
        
        #Plot the vertical cross-section
        fig = plt.figure(figsize=(10,10))
        plt.gca().invert_yaxis()
        plt.contourf(lat_subset, p, CLWC_vert[0,:,:],levels=clevels,cmap=plt.cm.rainbow,extend='max')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('')
        plt.title("Cloud Liquid Water Content")
        #save the figures for all dates and times 
        plt.savefig(pathout+'/ERA_interim_CLWC_Cross_Section'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_Longitude_'+str(lon_cross_section)+'.png',format ='png', dpi=150, bbox_inches='tight')

        clevels=np.linspace(0,2,41)*10**-4
        
        #Plot the vertical cross-section
        fig = plt.figure(figsize=(10,10))
        plt.gca().invert_yaxis()
        plt.contourf(lat_subset, p, CIWC_vert[0,:,:],levels=clevels,cmap=plt.cm.rainbow,extend='max')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('')
        plt.title("Cloud Ice Water Content")
        #save the figures for all dates and times 
        plt.savefig(pathout+'/ERA_interim_CIWC_Cross_Section'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_Longitude_'+str(lon_cross_section)+'.png',format ='png', dpi=150, bbox_inches='tight')


        print 'plotted  windspeed cross section now to plot relative vorticity'
        #print MSLP_slice.shape 
        # In[]: Plot the relative vorticity

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
        plt.contourf(lon_subset, lat_subset,vorticity[lat_min_index:lat_max_index, lon_min_index:lon_max_index], transform=ccrs.PlateCarree(),levels=clevels,cmap='seismic',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$s^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Relative Vorticity")
        plt.savefig(pathout+'/ERA_interim_Relative_Vorticity_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(950,1100,lats.size)
        plt.contourf(lon_subset, lat_subset,MSLP_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]/10**2,levels=clevels,transform=ccrs.PlateCarree(),cmap='jet',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$hPa$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Mean Sea Level Pressure")
        plt.savefig(pathout+'/ERA_interim_MSLP_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00.png',format ='png', dpi=150, bbox_inches='tight')
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(-2,2,lats.size)
        plt.contourf(lon_subset, lat_subset,w_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index],levels=clevels,transform=ccrs.PlateCarree(),cmap='seismic',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$ms^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Vertical Velocity")
        plt.savefig(pathout+'/ERA_interim_Vertical_velocity_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        #clevels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
        clevels = np.linspace(-1,1,lats.size)
        plt.contourf(lon_subset, lat_subset,D_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**4,levels=clevels,transform=ccrs.PlateCarree(),cmap='seismic',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$ 10^{-4} s^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Divergence")
        plt.savefig(pathout+'/ERA_interim_Divergence_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(0,2,lats.size)
        plt.contourf(lon_subset, lat_subset,PV_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**6, transform=ccrs.PlateCarree(),levels=clevels,cmap='Reds',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$10^{-6}$ PVU')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Potential Vorticity")
        plt.savefig(pathout+'/ERA_interim_potential_vorticity_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(230,330,lats.size)
        plt.contourf(lon_subset, lat_subset,T_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index],levels=clevels, transform=ccrs.PlateCarree(),cmap='jet',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$K$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Temperature")
        plt.savefig(pathout+'/ERA_interim_Temperature_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(230,290,lats.size)
        cont= plt.contourf(lon_subset, lat_subset,theta[lat_min_index:lat_max_index, lon_min_index:lon_max_index],levels=clevels, transform=ccrs.PlateCarree(),cmap='jet',extend='both')
        plt.contour(lon_subset,lat_subset,MSLP_slice[lat_min_index:lat_max_index,lon_min_index:lon_max_index],15,colors='black',transform=ccrs.PlateCarree())
        cbar = plt.colorbar(cont,orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$K$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Potential Temperature")
        plt.savefig(pathout+'/ERA_interim_potential_temperature_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')
   
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(0,8,lats.size)
        plt.contourf(lon_subset, lat_subset,Q_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**3,levels=clevels, transform=ccrs.PlateCarree(),cmap='Reds',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$10^{-3}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Specific Humidity")
        plt.savefig(pathout+'/ERA_interim_specific_humidity'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')
       
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels = np.linspace(0,1,lats.size)
        plt.contourf(lon_subset, lat_subset,CLWC_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**3,levels=clevels, transform=ccrs.PlateCarree(),cmap='Reds',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$10^{-3}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Cloud Liquid Water Content")
        plt.savefig(pathout+'/ERA_interim_CLWC_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')
 
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        clevels = np.linspace(0,1,lats.size)
        plt.contourf(lon_subset, lat_subset,CIWC_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index]*10**4,levels=clevels, transform=ccrs.PlateCarree(),cmap='Reds',extend='both')
        cbar = plt.colorbar(orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$10^{-4}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Cloud Ice Water Content")
        plt.savefig(pathout+'/ERA_interim_CLIC_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        ug_slice = ug[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        vg_slice = vg[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        uag_slice = uag[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        vag_slice = vag[lat_min_index:lat_max_index,lon_min_index:lon_max_index]

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        #clevels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
        plt.contour(lon_subset, lat_subset,Z_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index],15, transform=ccrs.PlateCarree(),colors='black')
        plt.quiver(lon_subset[::3],lat_subset[::3], ug_slice[::3,::3],vg_slice[::3,::3],transform=ccrs.PlateCarree(),linewidths=widths,color='blue',pivot='middle')
        #plt.quiver(lon_subset[::3],lat_subset[::3], uag_slice[::3,::3],vag_slice[::3,::3],transform=ccrs.PlateCarree(),color='red',pivot='middle')

        #cbar = plt.colorbar(orientation='horizontal')
        #cbar.outline.set_linewidth(0.5)
        #cbar.ax.tick_params(labelsize=10)
        #cbar.set_label('$m^2 s^{-2}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Geopotential Height Contours and Geostrophic Wind Vectors")
        plt.savefig(pathout+'/ERA_interim_Z_geostrophic_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')
    
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        #clevels = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
        plt.contour(lon_subset, lat_subset,Z_slice[lat_min_index:lat_max_index, lon_min_index:lon_max_index],25, transform=ccrs.PlateCarree(),colors='black')
        #plt.quiver(lon_subset[::3],lat_subset[::3], ug_slice[::3,::3],vg_slice[::3,::3],transform=ccrs.PlateCarree(),color='blue',pivot='middle')
        plt.quiver(lon_subset[::3],lat_subset[::3], uag_slice[::3,::3],vag_slice[::3,::3],transform=ccrs.PlateCarree(),linewidths=widths,color='red',pivot='middle')
        #cbar = plt.colorbar(orientation='horizontal')
        #cbar.outline.set_linewidth(0.5)
        #cbar.ax.tick_params(labelsize=10)
        #cbar.set_label('$m^2 s^{-2}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("Geopotential Height Contours and Ageostrophic Wind Vectors")
        plt.savefig(pathout+'/ERA_interim_Z_ageostrophic_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')    

        Qx_subset = Qx[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        Qy_subset = Qy[lat_min_index:lat_max_index,lon_min_index:lon_max_index]
        #N = np.sqrt(Qx_subset**2 + Qy_subset**2) 
        #Qx_subset = Qx_subset/N
        #Qy_subset = Qy_subset/N 
        #print Qx_subset.shape
        #print Qy_subset.shape
        #print divQ[lat_min_index:lat_max_index,lon_min_index:lon_max_index].shape

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        #clevels=np.linspace(-8,8,lats.size)*10**-13
        #plt.contourf(lon_subset, lat_subset,-2*divQ[lat_min_index:lat_max_index, lon_min_index:lon_max_index], transform=ccrs.PlateCarree(),levels=clevels,cmap='seismic',extend='both')
        plt.quiver(lon_subset[::3],lat_subset[::3],Qx_subset[::3,::3],Qy_subset[::3,::3],color='black',transform=ccrs.PlateCarree())
        #cbar = plt.colorbar(orientation='horizontal')
        #cbar.outline.set_linewidth(0.5)
        #cbar.ax.tick_params(labelsize=10)
        #cbar.set_label('$mkg^{-1}s^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("")
        plt.savefig(pathout+'/ERA_interim_Q_vector_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight')

        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        clevels=np.linspace(-3.5,3.5,lats.size)*10**-14
        widths = np.linspace(0,2,lats.size)*10**-16
        templevels=np.linspace(230,290,35)
        cont = plt.contourf(lon_subset, lat_subset,-2*divQ[lat_min_index:lat_max_index, lon_min_index:lon_max_index], transform=ccrs.PlateCarree(),levels=clevels,cmap='seismic',extend='both')
        #plt.quiver(lon_subset[::5],lat_subset[::5],Qx_subset[::5,::5],Qy_subset[::5,::5],color='black',transform=ccrs.PlateCarree(),linewidth=widths)
        ptemp = plt.contour(lon_subset,lat_subset,theta[lat_min_index:lat_max_index,lon_min_index:lon_max_index],transform=ccrs.PlateCarree(),levels=templevels,colors='black')
        plt.clabel(ptemp,inline=1,fontsize=10) 
        cbar = plt.colorbar(cont,orientation='horizontal')
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('$mkg^{-1}s^{-1}$')
        ax.add_feature(countries_50m, linewidth=1)
        gl = ax.gridlines(color="black", linestyle="dotted",draw_labels='True')
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlines = True
        gl.ylines = True
        gl.xlocator = mticker.FixedLocator([-40,-35,-30,-25,-20,-15,-10,-5, 0,5,10,15,20,25,30])
        gl.ylocator = mticker.FixedLocator([30,35,40,45,50,55,60,65,70,75,80])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #Save and show the figure
        plt.title("")
        plt.savefig(pathout+'/ERA_interim_divQ_'+str(year)+''+str(mon)+''+str(d)+'_'+str(time)+'00_'+str(p_level)+'hPa.png',format ='png', dpi=150, bbox_inches='tight') 
'''
