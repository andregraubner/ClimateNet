from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config

config = Config('config.json')
cgnet = CGNet(config)

train = ClimateDatasetLabeled('/cm/shared/pool/climatenet/train', cgnet.config)
test = ClimateDatasetLabeled('/cm/shared/pool/climatenet/test', cgnet.config)
allhist = ClimateDataset('/home/lukasks/neurips/input_data/ALLHIST/', cgnet.config)

cgnet.train(train, epochs=5)
cgnet.save_model('trained_cgnet')
cgnet.evaluate(test)
#predictions = cgnet.predict(allhist)