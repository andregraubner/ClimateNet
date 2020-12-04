from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config

config = Config('config.json')
cgnet_new = CGNet(config)

cgnet_loaded = CGNet.load_model(model_path='model')

train = ClimateDatasetLabeled('/cm/shared/pool/climatenet/', cgnet_loaded.config)
allhist = ClimateDataset('/home/lukasks/neurips/input_data/ALLHIST/', cgnet_loaded.config)

cgnet_loaded.train(train)
predictions = cgnet_loaded.predict(allhist)