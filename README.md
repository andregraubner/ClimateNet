# ClimateNet

ClimateNet is a Python library for deep learning-based Climate Science. It provides command line tools for quick detection and tracking of extreme weather events. We also expose models, data sets and metrics to jump-start your research.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install ClimateNet.

```bash
pip install ClimateNet
```

## Usage
Running inference using our provided models is simple.

```console
foo@bar:~$ climatenet track input_path output_path
```

The corresponding python package also seamlessly integrates into your PyTorch workflow.

```python
from climatenet import models, data, metrics

model = models.cgnet(load_weights='cgnet_base')
loader = data.loader(dir='path_to_data')

for sample in loader:
    pred = model(sample)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
