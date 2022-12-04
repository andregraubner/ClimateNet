from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config

#from climatenet.track_events import track_events
#from climatenet.analyze_events import analyze_events
#from climatenet.visualize_events import visualize_events

from os import path

config = Config('models/TMQ-WS850-VRT850-PSL-.001-wce/config.json')
cgnet = CGNet(config)

train_path = '../Data/ClimateNet_Engineered/'
#inference_path = '../Data/ClimateNet_Engineered/'

train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
#inference = ClimateDataset(inference_path, config)

cgnet.train(train)
cgnet.evaluate(test)

cgnet.save_model('trained_cgnet')
# use a saved model with
# cgnet.load_model('trained_cgnet')

#class_masks = cgnet.predict(inference) # masks with 1==TC, 2==AR
#event_masks = track_events(class_masks) # masks with event IDs

#analyze_events(event_masks, class_masks, 'results/')
#visualize_events(event_masks, inference, 'pngs/')
