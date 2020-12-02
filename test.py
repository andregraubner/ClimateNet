from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
import json

config = json.load(open('config.json'))

cgnet = CGNet(config)
train = ClimateDatasetLabeled('/home/lukasks/neurips/expert_data/', cgnet.fields)
allhist = ClimateDataset('/home/lukasks/neurips/input_data/ALLHIST/', cgnet.fields)

#cgnet.train(dataset, loss='jaccard', epochs=15)

predictions = cgnet.predict(allhist)

#cgnet.evaluate()
#cgnet.save_weights('PATH-TO-SAVE')