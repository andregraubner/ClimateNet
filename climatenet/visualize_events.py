import pathlib
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
import psutil
from climatenet.utils.data import ClimateDataset

def visualize_events(event_masks_xarray, input_data : ClimateDataset, png_dir):
    """Generates PNGs of event masks with TMQ background

    The PNGs can later be stitched together to a video with a tool such as ffmpeg. E.g. use
    ffmpeg -r 5 -pattern_type glob -i 'png_dir/*.png' -c:v libx264 -pix_fmt yuv420p 
           -vf scale=1920:1080,fps=29 -crf 29 -preset veryslow video.mp4

    Keyword arguments:
    input_data -- a ClimateDataset containt TMQ
    event_masks_xarray -- the event masks as xarray with IDs as elements 
    png_dir -- the directory where the PNGs get saved to
    """

    # create png_dir if it doesn't exist
    pathlib.Path(png_dir).mkdir(parents=True, exist_ok=True) 

    event_masks = event_masks_xarray.values

    # latitude and longitude grid
    lat = event_masks_xarray.lat
    lon = event_masks_xarray.lon

    def map_instance():
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
        plt.title("AR and TC Event Tracking", fontdict={'fontsize': 44})
        
        return mymap

    def generatePNG(filepath, tmq_data, event_mask):
        """Save a PNG of tmq_data and event_mask filepath"""

        mymap = map_instance()

        # contour data
        colors_1 = [(252-32*i,252-32*i,252-32*i,i*1/16) for i in np.linspace(0, 1, 32)]
        colors_2 = [(220-60*i,220-60*i,220,i*1/16+1/16) for i in np.linspace(0, 1, 32)]
        colors_3 = [(160-20*i,160+30*i,220,i*3/8+1/8) for i in np.linspace(0, 1, 96)]
        colors_4 = [(140+80*i,190+60*i,220+30*i,i*4/8+4/8) for i in np.linspace(0, 1, 96)]
        colors = colors_1 + colors_2 + colors_3 + colors_4
        colors = list(map(lambda c: (c[0]/256,c[1]/256,c[2]/256,c[3]), colors))
        data_cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=64) 

        data_contour = mymap.contourf(lon,lat,tmq_data,128,vmin=0,vmax=89, cmap=data_cmap,
                                      levels=np.arange(0,89,2),transform=ccrs.PlateCarree())
        
        # contour events   
        ls = np.linspace(0,1,2000)
        np.random.shuffle(ls)
        event_cmap = ListedColormap([1,1,1,0.3] * np.concatenate((np.zeros((1,4)), plt.cm.hsv(ls))))
        event_contourf = mymap.contourf(lon,lat,event_mask,cmap=event_cmap,
                                    vmin=0,vmax=len(event_cmap.colors)-1,levels=np.arange(len(event_cmap.colors)),
                                    norm=mpl.colors.Normalize(vmin=0, vmax=len(event_cmap.colors)-1))
        event_contour = mymap.contour(lon,lat,event_mask,colors=['#000000ff'])
        
        #colorbar and legend
        cbar = mymap.get_figure().colorbar(data_contour,ticks=np.arange(0,89,11),orientation='vertical')
        cbar.ax.set_ylabel('Integrated Water Vapor kg $m^{-2}$',size=32)

        #savefig
        mymap.get_figure().savefig(filepath,bbox_inches="tight",facecolor='w')

    print('generating images..', flush=True)

    global spawn # make function visible to pool
    def spawn(i):
        filename = png_dir + f"{i:04d}.png"
        generatePNG(filename, input_data[int(i/8)].sel(variable="TMQ")[i%8],
                    event_masks[i])

    pool = Pool(psutil.cpu_count(logical=False))
    pool.map(spawn, range(len(event_masks)))