import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np

start_time=datetime.now()

def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)


fig=plt.figure(figsize=(9,5))
tiler=cimgt.Stamen('terrain-background')
mercator=tiler.crs
ax=plt.axes(projection=mercator)
#extent=[153,153.2,-26.6,-26.4]
extent=[-116,-124,34,39]
ax.set_extent(extent)

zoom = 9
ax.add_image(tiler, zoom)

#ax.add_image(google_terrain,6)

ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.STATES)

text_kwargs = dict(size='x-large')
scale_bar(ax, (0.85, 0.04), 5_0,text_kwargs=text_kwargs)


#PCA
lons=[-121.6,-118.75]
lats=[37.05,37.5]
ax.plot(lons,lats,color='red',label='PCA',transform=ccrs.Geodetic())
lons=[-121.6,-121.25]
lats=[37.05,35.5]
ax.plot(lons,lats,color='red',transform=ccrs.Geodetic())
lons=[-121.25,-118.4]
lats=[35.5,35.9]
ax.plot(lons,lats,color='red',transform=ccrs.Geodetic())
lons=[-118.4,-118.75]
lats=[35.9,37.5]
ax.plot(lons,lats,color='red',transform=ccrs.Geodetic())
#GRACE
lons=[-121.6828,-118.3219]
lats=[37.5539,37.55539]
ax.plot(lons,lats,color='green',label='GRACE/GLDAS',transform=ccrs.Geodetic())
lons=[-118.3219,-118.3219]
lats=[37.5539,35.3602]
ax.plot(lons,lats,color='green',transform=ccrs.Geodetic())
lons=[-118.3219,-121.6828]
lats=[35.3602,35.3602]
ax.plot(lons,lats,color='green',transform=ccrs.Geodetic())
lons=[-121.6828,-121.6828]
lats=[35.3602,37.5539]
ax.plot(lons,lats,color='green',transform=ccrs.Geodetic())
#LiCSBAS + maybe LSTM?
#Got the values from the LiCSBAS plot
lons=[-121.90311,-119.15]
lats=[38.33011,38.79111]
ax.plot(lons,lats,color='blue',label='LiCSBAS',transform=ccrs.Geodetic())
lons=[-119.15,-118.53811]
lats=[38.79111,36.06411]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
lons=[-118.53811,-121.30011]
lats=[36.06411,35.71611]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
lons=[-121.30011,-121.90311]
lats=[35.71611,38.33011]
ax.plot(lons,lats,color='blue',transform=ccrs.Geodetic())
# #LSTM coordinates too small (consider separate map?)
# lons=[-120.07611,-120.06611]
# lats=[36.96211,36.96211]
# ax.plot(lons,lats,color='black',label='LSTM',transform=ccrs.Geodetic())
# lons=[-120.06611,-120.06611]
# lats=[36.96211,36.95211]
# ax.plot(lons,lats,color='black',transform=ccrs.Geodetic())
# lons=[-120.06611,-120.07611]
# lats=[36.95211,36.95211]
# ax.plot(lons,lats,color='black',transform=ccrs.Geodetic())
# lons=[-120.07611,-120.07611]
# lats=[36.95211,36.96211]
# ax.plot(lons,lats,color='black',transform=ccrs.Geodetic())
#MADERA
ax.plot(-120.0607,36.9613,marker='*',color='yellow',markersize=15,linestyle='None',label='Madera',transform=ccrs.Geodetic())

ax.text(-123.5,34.5,'Pacific Ocean',rotation=-45,fontsize=12,transform=ccrs.Geodetic())
ax.text(-117.65,36.25,'California',rotation=-44,fontsize=12,transform=ccrs.Geodetic())
ax.text(-117.3,36.5,'Nevada',rotation=-44,fontsize=12,transform=ccrs.Geodetic())
#North Arrow
x_pos=-116.5
start=35
finish=35.5
arrow_length=0.1
lons=[x_pos,x_pos]
lats=[start,finish-0.05]
ax.plot(lons,lats,color='black',transform=ccrs.Geodetic())
# lons=[x_pos,x_pos+arrow_length]
# lats=[finish,finish-arrow_length]
# ax.plot(lons,lats,color='black',linewidth=2,transform=ccrs.Geodetic())
# lons=[x_pos,x_pos-arrow_length]
# lats=[finish,finish-arrow_length]
# ax.plot(lons,lats,color='black',linewidth=2,transform=ccrs.Geodetic())

ax.text(-116.58,35.6,'N',fontsize=10,transform=ccrs.Geodetic())

trianglex=[x_pos,x_pos+arrow_length,x_pos-arrow_length,x_pos]
triangley=[finish,finish-arrow_length,finish-arrow_length,finish]
ax.plot(trianglex,triangley,'black',transform=ccrs.Geodetic())
ax.fill(trianglex,triangley,'black',transform=ccrs.Geodetic())


gl = ax.gridlines(draw_labels=True,
                  linewidth=1, color='grey', alpha=0.2)
gl.top_labels = True
gl.left_labels = True
gl.xlines = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 15, 'color': 'black'}
gl.ylabel_style = {'size': 15, 'color': 'black'}

ax.legend(prop={'size':9})

# plt.savefig('Central Valley map.pdf')
plt.show()

print(datetime.now()-start_time)












