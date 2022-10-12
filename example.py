from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

from os import path

def run(checkpoint_path=None,data_dir=None):
    config = Config('config.json')
    cgnet = CGNet(config)
    train_path = data_dir + '/train'
    inference_path = data_dir + '/test' # I don't get what is the inference dataset

    print("Train path : ", path.join(train_path, 'train'))
    print("Test path  : ", path.join(train_path, 'test' ))
    train = ClimateDatasetLabeled(path.join(train_path, 'train'), config)
    test = ClimateDatasetLabeled(path.join(train_path, 'test'), config)
    inference = ClimateDataset(inference_path, config)

    # cgnet.train(train)
    # cgnet.evaluate(test)
    # cgnet.save_model('trained_cgnet_2')
    # use a saved model with
    cgnet.load_model('trained_cgnet')

    class_masks = cgnet.predict(inference) # masks with 1==TC, 2==AR
    event_masks = track_events(class_masks) # masks with event IDs

    try :
        analyze_events(event_masks, class_masks, 'results/')
    except Exception as e:
        print("Error when analyzing events : ", e)

    try : 
        visualize_events(event_masks, inference, 'pngs/')
    except Exception as e:
        print("Error when visualizing events : ", e)
