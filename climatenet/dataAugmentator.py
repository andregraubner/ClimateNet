import netCDF4 as nc
import os
import random

data_path = '/Users/lucashendren/workspace/cs230/Data-4/Train'
dims_to_change = ["lon"]

for filename in os.listdir(data_path):
    f = os.path.join(data_path, filename)
    newF = os.path.join(data_path,"augmented-"+filename)

    # checking if it is a file
    if filename == ".DS_Store":
        continue
    if os.path.isfile(f):
        ds = nc.Dataset(f)
        print(ds)

        newDs = nc.Dataset(newF, "a", format="NETCDF4")
        newDs.setncatts(ds.__dict__)

        # copy dimensions
        for name, dimension in ds.dimensions.items():
            newDs.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
       # copy all file data modif dims to change
        for name, variable in ds.variables.items():
            x = newDs.createVariable(name, variable.datatype, variable.dimensions)
            values = ds[name][:]
            if name in dims_to_change:
                for i,val in enumerate(values):
                    toAdd = random.randint(-180, 180)
                    newVal=val+toAdd
                    if newVal < -180:
                        newVal= 360+newVal
                    if newVal > 180:
                        newVal= newVal-360
                    values[i]=newVal

            newDs[name][:] = values
            if name in dims_to_change:
                print("old",ds.variables[name][0])
                print("new",newDs.variables[name][0])
        print(ds)
        print(newDs)

        newDs.close()
