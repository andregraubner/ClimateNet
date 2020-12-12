# ClimateNet

ClimateNet is a Python library for deep learning-based Climate Science. It provides tools for quick detection and tracking of extreme weather events. We also expose models, data sets and metrics to jump-start your research.

## Usage

Install the conda environment using `conda env create -f conda_env.yml`.

You can find the data and a pre-trained model at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/).
Download the train and test data and the trained model, and you're good-to-go.

The high-level API makes it easy to train a model from scratch or to use our models to run inference on your own climate data. Just download the model config (or write your own) and train the model using:

```python
config = Config('PATH_TO_CONFIG')
model = CGNet(config)

training_set = ClimateDatasetLabeled('PATH_TO_TRAINING_SET', model.config)
inference_set = ClimateDataset('PATH_TO_INFERENCE_SET', model.config)

model.train(training_set)
model.save_model('PATH_TO_SAVE')

predictions = model.predict(inference_set)
```

You can find an example of how to load our trained model for inference in example.py.

If you are familiar with PyTorch and want a higher degree of control over training procedures, data flow or other aspects of your project, we suggest you use our lower-level modules.
The CGNetModule and Dataset classes conform to what you would expect from standard PyTorch, which means that you can take whatever parts you need and swap out the others for your own building blocks. A quick example of this:

```python
training_data = ... # Plug in your own Dataloader and data handling
cgnet = CGNetModule(classes=3, channels=4)
optimizer = Adam(cgnet.parameters(), ...)      
for features, labels in epoch_loader:
    outputs = softmax(cgnet(features), 1)

    loss = jaccard_loss(outputs, labels) # Or plug in your own loss...
    loss.backward()
    optimizer.step()
    optimizer.zero_grad() 
```

## Data

Climate data can be complex. In order to avoid hard-to-debug issues when reading and interpreting the data, we require the data to adhere to a strict interface when using the high-level abstractions. We're working on conforming to the NetCDF Climate and Forecast Metadata Conventions in order to provide maximal flexibility while still making sure that your data gets interpreted the right way by the models.

## Configurations

When creating a (high-level) model, you need to specify a configuration - on one hand this encourages reproducibility and makes it easy to track and share experiments, on the other hand it helps you avoid issues like running a model that was trained on one variable on an unrelated variable or using the wrong normalisation statistics.
See config.json for an example configuration file.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
