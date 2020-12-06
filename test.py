from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

config = Config('config.json')
cgnet = CGNet(config)

train = ClimateDatasetLabeled('/cm/shared/pool/climatenet/train', cgnet.config)
test = ClimateDatasetLabeled('/cm/shared/pool/climatenet/test', cgnet.config)
#allhist = ClimateDataset('/home/lukasks/neurips/input_data/ALLHIST/', cgnet.config)
allhist = ClimateDataset('/home/lukasks/neurips/input_data/ALLHIST_small/', cgnet.config)

#cgnet.train(train, epochs=5)

#cgnet.save_model('trained_cgnet')
cgnet.load_model('trained_cgnet')

#cgnet.evaluate(test)

class_masks = cgnet.predict(allhist) # masks with 1==TC, 2==AR
event_masks = track_events(class_masks) # masks with event IDs

analyze_events(event_masks, class_masks, 'results/')
visualize_events(event_masks, allhist, 'pngs/')