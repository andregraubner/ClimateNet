import psutil
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import xarray as xr

def track_events(class_masks_xarray, minimum_time_length=5,
                 tc_drop_threshold=250, ar_drop_threshold=250,
                 future_lookup_range=1):
    """track AR and TC events across time

    Keyword arguments:
    class_masks_xarray -- the class masks as xarray, 0==Background, 1==TC, 2 ==AR
    minimum_time_length -- the minimum number of time stamps an event ought to persist to be considered
    tc_dop_threshold -- the pixel threshold below which TCs are droped
    ar_dop_threshold -- the pixel threshold below which ARs are droped
    future_lookup_range -- across how many time stamps events get stitched together
    """

    class_masks = class_masks_xarray.values

    # per timestamp, assign ids to connected components
    global identify_components # make function visible to pool
    def identify_components(time):
        """Returns an event mask with ids assigned to the connected components at time"""
        class_mask = class_masks[time] # class masks of assigned time stamp
        # data structure for ids of connected components
        event_mask = np.zeros(np.shape(class_mask)).astype(np.int)
        next_id = 1
        for i in range(np.shape(class_mask)[0]):
            for j in range(np.shape(class_mask)[1]):
                class_type = class_mask[i][j]
                if class_type != 0 and event_mask[i][j] == 0: # label connected component with new id with BFS
                    frontier = [(i, j)]
                    event_mask[i, j] = next_id
                    while len(frontier) > 0:
                        element = frontier.pop(0)
                        row = element[0]
                        col = element[1]
                        for neighbor_row in range(row-1, row+2):
                            neighbor_row = neighbor_row % event_mask.shape[0]
                            for neighbor_col in range(col-1, col+2):
                                neighbor_col = neighbor_col % event_mask.shape[1]
                                if class_mask[neighbor_row][neighbor_col] != class_type: # don't propagate to other type
                                    continue
                                if event_mask[neighbor_row][neighbor_col] == 0: # not yet considered
                                    event_mask[neighbor_row][neighbor_col] = next_id
                                    frontier.append((neighbor_row, neighbor_col))
                    next_id = next_id + 1
        return event_mask

    # paralelize call to identify_components
    print('identifying connected components..', flush=True)
    pool = Pool(psutil.cpu_count(logical=False))
    event_masks = np.array(pool.map(
                        identify_components, 
                        range(len(class_masks)))).astype(np.int)

    def size_is_smaller_threshold(mask, i, j, threshold):
        """Returns True iff the size of the connected component that (i, j) is part of is smaller threshold"""
        visited = np.full(mask.shape, False)
        component_class = mask[i][j]
        frontier = [(i, j)]
        visited[i][j] = True
        count = 1
        while len(frontier) > 0:
            element = frontier.pop(0)
            row = element[0]
            col = element[1]
            for neighbor_row in range(row-1, row+2):
                neighbor_row = neighbor_row % mask.shape[0]
                for neighbor_col in range(col-1, col+2):
                    neighbor_col = neighbor_col % mask.shape[1]
                    if visited[neighbor_row][neighbor_col] == True:
                        continue
                    if mask[neighbor_row][neighbor_col] == component_class:
                        visited[neighbor_row][neighbor_col] = True
                        frontier.append((neighbor_row, neighbor_col))
                        count = count + 1
                        if count >= threshold:
                            return False
        return True

    def drop_threshold(class_mask, i, j):
        """Returns the minimal size a connected component containting [i, j] must have to not get removed"""
        if class_mask[i][j] == 1: #TC
            return tc_drop_threshold
        if class_mask[i][j] == 2: #AR
            return ar_drop_threshold    

    def label_component(mask, i, j, new_label, threshold=None):
        """Labels a connected component
        
        Labels the connected component at pixel [i, j] in mask as part of a component with new_label
        If a threshold is given: If the size of the connected component <= threshold: set the component to background (0) 
        Return True if the componend was set to new_label, False if it was set to background"""

        # apply thresholding
        if threshold != None and size_is_smaller_threshold(mask, i, j, threshold):
            label_component(mask, i, j, 0) # set component to background
            return False
        
        old_label = mask[i][j]
        if old_label == 0:
            return False

        frontier = [(i, j)]
        mask[i, j] = new_label
        while len(frontier) > 0:
            element = frontier.pop(0)
            row = element[0]
            col = element[1]
            for neighbor_row in range(row-1, row+2):
                neighbor_row = neighbor_row % mask.shape[0]
                for neighbor_col in range(col-1, col+2):
                    neighbor_col = neighbor_col % mask.shape[1]
                    if mask[neighbor_row][neighbor_col] == old_label:
                        mask[neighbor_row][neighbor_col] = new_label
                        frontier.append((neighbor_row, neighbor_col))
        return True

    print('tracking components across time..', flush=True)
    event_ids_per_time = [set() for x in range(len(event_masks))]
    # per time stamp and per id, have a pixel pointing to each connected component
    pixels_per_event_id_per_time = [{} for x in range(len(event_masks))] # one pixel per event Id
    new_event_index = 1000

    for time in tqdm(range(len(event_masks))):
        class_mask = class_masks[time]
        event_mask = event_masks[time]
        for i in range(np.shape(event_mask)[0]):
            for j in range(np.shape(event_mask)[1]):
                id = event_mask[i][j]
                # ignore background
                if id == 0:
                    continue
                # label new components
                if id < 1000:
                    above_threshold = label_component(event_mask, i, j, new_event_index,
                                                    drop_threshold(class_mask, i, j))
                    if above_threshold:
                        id = new_event_index
                        new_event_index = new_event_index + 1                    
                        event_ids_per_time[time].add(id)
                        pixels_per_event_id_per_time[time][id] = [(i, j)]
                    else:
                        label_component(class_mask, i, j, 0) # set component class to background
                        continue # component removed, don't propagate across time
                # label components in next time stamp(s) that are already present in this time stamp
                for future_time in range(time+1, min(time+1+future_lookup_range, len(event_masks))):
                    future_event_mask = event_masks[future_time]
                    future_class_mask = class_masks[future_time]
                    
                    if class_mask[i][j] == future_class_mask[i][j] \
                        and future_event_mask[i][j] < 100 \
                        and future_event_mask[i][j] != 0: # not removed yet

                        above_threshold = label_component(future_event_mask, i, j, id, drop_threshold(class_mask, i, j))
                        if above_threshold:
                            event_ids_per_time[future_time].add(id)
                            if id in pixels_per_event_id_per_time[future_time].keys():
                                # id was already in other connected component for this time stamp
                                # add another pixel so that all connected components with this id are ponited to
                                pixels_per_event_id_per_time[future_time][id].append((i, j))
                            else:
                                pixels_per_event_id_per_time[future_time][id] = [(i, j)]
                        else: # [i,j] removed from event_masks, also remove from class_masks
                            label_component(future_class_mask, i, j, 0)

    # removing connected components appearing less than minimum_time_length
    # finding time stamps per id
    times_per_event_id = {}
    for time in range(len(event_ids_per_time)):
        for id in event_ids_per_time[time]:
            if id in times_per_event_id:
                times_per_event_id[id].append(time)
            else:
                times_per_event_id[id] = [time]

    # finding ids_for_removal i.e. ids occuring shorter than minimum_time_length
    ids_for_removal = set()
    for id in times_per_event_id.keys():
        times = times_per_event_id[id]
        if len(times) < minimum_time_length:
            ids_for_removal.add(id)

    # removing ids_for_removal  
    for id in ids_for_removal:
        for time in times_per_event_id[id]:
            for pixel in pixels_per_event_id_per_time[time][id]:
                label_component(event_masks[time], pixel[0], pixel[1], 0) 
                label_component(class_masks[time], pixel[0], pixel[1], 0)                           
                event_ids_per_time[time].discard(id)
            del pixels_per_event_id_per_time[time][id]
        del times_per_event_id[id]

    # counting ARs and TCs
    num_tcs = 0
    num_ars = 0
    for id in times_per_event_id.keys():
        time = times_per_event_id[id][0]
        pixel = pixels_per_event_id_per_time[time][id][0]
        if class_masks[time][pixel[0]][pixel[1]] == 1:
            num_tcs = num_tcs + 1
        else:
            num_ars = num_ars + 1
    print('num TCs: ' + str(num_tcs))
    print('num ARs: ' + str(num_ars))

    event_masks_xarray = xr.DataArray(event_masks, 
                                      coords=class_masks_xarray.coords, 
                                      attrs=class_masks_xarray.attrs)
    
    return event_masks_xarray