import os
import pathlib
import math
from multiprocessing import Pool
import psutil
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import haversine as hs

def analyze_events(event_masks_xarray, class_masks_xarray, results_dir):
    """Analzse event masks of ARs and TCs

    Produces PNGs of
        - histograms of event lifetimes, speeds, and travel_distances
        - Frequency plots of genesus, termination, and global occurence
    
    Keyword arguements:
    class_masks_xarray -- the class masks as xarray, 0==Background, 1==TC, 2 ==AR
    event_masks_xarray -- the event masks as xarray with IDs as elements 
    results_dir -- the directory where the PNGs get saved to
    """
    # create results_dir if it doesn't exist
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True) 

    class_masks = class_masks_xarray.values
    event_masks = event_masks_xarray.values

    print('calculating centroids..', flush=True)

    def pixel_to_degree(pos):
        """Returns the (lat,long) position of a pixel coordinate"""
        return(pos[0] * 180.0 / event_masks.shape[1] - 90,
            pos[1] * 360 / event_masks.shape[2] + 180)

    def average_location(coordinates_pixel):
        """Returns the average geolocation in pixel space

        Based on https://stackoverflow.com/questions/37885798/how-to-calculate-the-midpoint-of-several-geolocations-in-python
        """
        coordinates_degree = [pixel_to_degree(cord) for cord in coordinates_pixel]    
        
        x = 0.0
        y = 0.0
        z = 0.0

        for lat_deg, lon_deg in coordinates_degree:
            latitude = math.radians(lat_deg)
            longitude = math.radians(lon_deg)

            x += math.cos(latitude) * math.cos(longitude)
            y += math.cos(latitude) * math.sin(longitude)
            z += math.sin(latitude)

        total = len(coordinates_degree)

        x = x / total
        y = y / total
        z = z / total

        central_longitude = math.atan2(y, x)
        central_square_root = math.sqrt(x * x + y * y)
        central_latitude = math.atan2(z, central_square_root)

        average_degree = math.degrees(central_latitude), math.degrees(central_longitude)

        return (event_masks.shape[1] * (average_degree[0] + 90) / 180,
                event_masks.shape[2] * (average_degree[1] + 180) / 360)

    global centroids # make function visible to pool
    def centroids(event_mask):
        """Returns a dict mapping from the IDs in event_mask to their centroids"""
        coordinates_per_id = {}
        
        for row in range(np.shape(event_mask)[0]):
            for col in range(np.shape(event_mask)[1]):
                this_id = event_mask[row][col]
                if this_id == 0: # don't consider background as event
                    continue
                coordinates_per_id.setdefault(this_id, []).append((row, col))
        
        centroid_per_id = {}
        for this_id in coordinates_per_id:
            centroid_per_id[this_id] = average_location(coordinates_per_id[this_id])
        
        return centroid_per_id

    pool = Pool(psutil.cpu_count(logical=False))
    centroid_per_id_per_time = pool.map(centroids, event_masks)




    # %%
    print('extracting event types..', flush=True)
    global event_type_of_mask # make function visible to pool
    def event_type_of_mask(event_mask, class_mask):
        """Returns a dict mapping from the IDs in event_mask to their type ('tc' or 'ar")"""
        print('computing event types')
        event_type = {} # event type as tring 'ar' or 'tc' per event ID
        for row in range(np.shape(event_mask)[0]):
            for col in range(np.shape(event_mask)[1]):
                this_id = event_mask[row][col]
                this_class = class_mask[row][col]
                if this_id == 0:
                    continue
                elif this_class == 1:
                    event_type[this_id] = 'tc'
                else:
                    event_type[this_id] = 'ar'
        return event_type

    pool = Pool(psutil.cpu_count(logical=False))
    pool_result = pool.starmap(event_type_of_mask, zip(event_masks, class_masks))
    event_type = dict(i for dct in pool_result for i in dct.items())


    # %%
    print('calculating genesis and termination frequencies..', flush=True)
    genesis_time_per_id = {}
    termination_time_per_id = {}

    previous_ids = set()
    for time in range(len(event_masks)):
        for this_id in centroid_per_id_per_time[time].keys():
            if this_id not in previous_ids:
                genesis_time_per_id[this_id] = time
                previous_ids.add(this_id)
            termination_time_per_id[this_id] = time
            
    genesis_ids_per_time = {}
    termination_ids_per_time = {}
    for this_id, time in genesis_time_per_id.items():
        genesis_ids_per_time.setdefault(time, []).append(this_id)
    for this_id, time in termination_time_per_id.items():
        termination_ids_per_time.setdefault(time, []).append(this_id)
        
    genesis_count_ar = np.zeros(event_masks.shape[1:3]) # sum over all AR genesis events
    genesis_count_tc = np.zeros(event_masks.shape[1:3])
    termination_count_ar = np.zeros(event_masks.shape[1:3])
    termination_count_tc = np.zeros(event_masks.shape[1:3])
    for time in range(event_masks.shape[0]):
        genesis_events = np.isin(event_masks[time], genesis_ids_per_time.get(time, []))
        termination_events = np.isin(event_masks[time], termination_ids_per_time.get(time, []))
        genesis_count_tc += (class_masks[time] == 1) * genesis_events
        genesis_count_ar += (class_masks[time] == 2) * genesis_events
        termination_count_tc += (class_masks[time] == 1) * termination_events
        termination_count_ar += (class_masks[time] == 2) * termination_events

    genesis_frequency_ar = genesis_count_ar / (5 * 12)
    genesis_frequency_tc = genesis_count_tc / (5 * 12)
    termination_frequency_ar = termination_count_ar / (5 * 12)
    termination_frequency_tc = termination_count_tc / (5 * 12)


    # %%
    print('generating histograms..', flush=True)
    event_ids = set(genesis_time_per_id.keys()).union(set(termination_time_per_id.keys()))

    for event_class in ['tc', 'ar']:
        this_class_ids = set()
        for event_id in event_ids:
            if event_type[event_id] == event_class:
                this_class_ids.add(event_id)
        
        # lifetime calculation
        termination_times = np.array([termination_time_per_id[event_id] for event_id in this_class_ids])
        genesis_times = np.array([genesis_time_per_id[event_id] for event_id in this_class_ids])
        lifetimes = termination_times - genesis_times
        
        # lifetime histogram
        plt.figure(dpi=100)
        plt.hist(3 * lifetimes, bins = np.arange(0, 264, 12),
                cumulative=0, rwidth=0.85, color='#607c8e') # multiplied by 3 to get result in hours
        plt.title(f"Lifetime histogram of {event_class.upper():s}s", fontdict={'fontsize': 16})  
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        plt.xlabel("Lifetime in hours")
        plt.xticks(np.arange(12, 264, 48))
        plt.xlim(12, 252)
        plt.ylabel("Count")
        # plt.show()
        plt.savefig(results_dir + f"histogram_lifetime_{event_class:s}")
        
        # travel distance calculation
        termination_centroids = []
        genesis_centroids = []
        for i in range(len(this_class_ids)):
            termination_centroids.append(centroid_per_id_per_time[termination_times[i]][list(this_class_ids)[i]])
            genesis_centroids.append(centroid_per_id_per_time[genesis_times[i]][list(this_class_ids)[i]]) 
        
        distances = np.array([hs.haversine(pixel_to_degree(pos1), pixel_to_degree(pos2))
                            for pos1, pos2 in zip(termination_centroids, genesis_centroids)])
        
        # travel distance histogram
        plt.figure(dpi=100)
        plt.hist(distances, bins = np.arange(0, 10000, 500),
                rwidth=0.85, color='#607c8e')
        plt.title(f"Travel distance histogram of {event_class.upper():s}s", fontdict={'fontsize': 16})  
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        plt.xlabel("distance in km")
        plt.xticks(np.arange(0, 10001, 2500))
        plt.xlim(0, 10000)
        plt.ylabel("Count")
        plt.savefig(results_dir + f"histogram_travel_distance_{event_class:s}")
        
        # speed histogram
        plt.figure(dpi=100)
        plt.hist(distances / (3 * lifetimes), bins = np.arange(0, 100, 5),
                rwidth=0.85, color='#607c8e') # multiplied by 3 to get result in km/h)
        plt.title(f"Speed histogram of {event_class.upper():s}s", fontdict={'fontsize': 16})  
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        plt.xlabel("speed in km/h")
        plt.xticks(np.arange(0, 101, 25))
        plt.xlim(0, 100)
        plt.ylabel("Count")
        plt.savefig(results_dir + f"histogram_speed_{event_class:s}")
        
    # set cartopy background dir to include blue marble
    os.environ['CARTOPY_USER_BACKGROUNDS'] = str(os.getcwd() + '/climatenet/bluemarble')

    def map_instance(title):
        """Returns a matplotlib instance with bluemarble background"""
        plt.figure(figsize=(100,20),dpi=100)
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        mymap = plt.subplot(111,projection=ccrs.PlateCarree())
        mymap.set_global()
        mymap.background_img(name='BM')
        mymap.coastlines()
        mymap.gridlines(crs=ccrs.PlateCarree(),linewidth=2, color='k', alpha=0.5, linestyle='--')
        mymap.set_xticks([-180,-120,-60,0,60,120,180])
        mymap.set_yticks([-90,-60,-30,0,30,60,90])
        plt.title(title, fontdict={'fontsize': 44})
        return mymap

    # Sth in this function fails with <urlopen error [Errno 111] Connection refused>
    def visualize_frequency_map(frequency_map, title, colorbar_text, filepath): 
        """Save a PNG of frequency_map with title and colorbar_text at filepath"""
        print('visualizing frequency map :', title)
        
        # initialize
        print('initializing..', title, flush=True)
        mymap = map_instance(title)
        lon = np.linspace(0,360,frequency_map.shape[1])
        lat = np.linspace(-90,90,frequency_map.shape[0])
        
        # draw frequencies
        print('drawing frequencies..', title, flush=True)
        contourf = mymap.contourf(lon, lat, 
                                np.ma.masked_array(frequency_map, mask=(frequency_map==0)), 
                                levels=np.linspace(0.0, frequency_map.max(), 11),
                                alpha=0.7)

        #colorbar and legend
        print('drawing colorbar..', title, flush=True)
        cbar = mymap.get_figure().colorbar(contourf,orientation='vertical',
                                        ticks=np.linspace(0,frequency_map.max(),3))
        cbar.ax.set_ylabel(colorbar_text,size=32)
        
        #save
        mymap.get_figure().savefig(filepath, bbox_inches="tight", facecolor='w')


    print('generating frequency maps..', flush=True)
    visualize_frequency_map(genesis_frequency_tc, "Genesis frequency map of TCs",
                            "Frequency in events per month", results_dir + "genesis_frequency_tc")
    visualize_frequency_map(genesis_frequency_ar, "Genesis frequency map of ARs",
                            "Frequency in events per month", results_dir + "genesis_frequency_ar")
    visualize_frequency_map(termination_frequency_tc, "Termination frequency map of TCs",
                            "Frequency in events per month", results_dir + "termination_frequency_tc")
    visualize_frequency_map(termination_frequency_ar, "Termination frequency map of ARs",
                            "Frequency in events per month", results_dir + "termination_frequency_ar")

    visualize_frequency_map(100 * ((class_masks == 1) * (event_masks != 0)).sum(axis=0) / event_masks.shape[0],
                            "Global frequency map of TCs", "Frequency in % of time steps",
                            results_dir + "global_frequency_tc");
    visualize_frequency_map(100 * ((class_masks == 2) * (event_masks != 0)).sum(axis=0) / event_masks.shape[0],
                            "Global frequency map of ARs", "Frequency in % of time steps",
                            results_dir + "global_frequency_ar");
