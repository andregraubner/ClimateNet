from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config

#from climatenet.track_events import track_events
#from climatenet.analyze_events import analyze_events
#from climatenet.visualize_events import visualize_events

from os import path

config = Config('config-ablation-V850.json')
cgnet = CGNet(config)

train_path = 'Data/engineered'
inference_path = 'Data'


train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
#inference = ClimateDataset(inference_path, config)

cgnet.train(train)
cgnet.evaluate(test)

model_path = "ablation-V850-.001-jaccard"
cgnet.save_model(path.join('models', model_path))
# use a saved model with
# cgnet.load_model('trained_cgnet')

#class_masks = cgnet.predict(inference) # masks with 1==TC, 2==AR
#event_masks = track_events(class_masks) # masks with event IDs

#analyze_events(event_masks, class_masks, 'results/')
#visualize_events(event_masks, inference, 'pngs/')
