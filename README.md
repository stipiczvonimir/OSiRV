# OSiRV Detekcija objekta iz predloška putem normalizirane unakrsne korelacije

### Installation
Requires python>= 3.6
```bash
pip install -r requirements.txt
```

### Generate dataset 
```bash
python generate_data.py
```

You can also change a bunch of settings by writing:
```bash
python generate_dataset.py -h
```
### Create templates
```bash
python templates.py
```

### Detect digits
```bash
python normalized_cross_correlation.py data/mnist_detection/test/ 
```
For a different number of displayed images use:
```bash
python normalized_cross_correlation.py data/mnist_detection/test/  --num_images
```