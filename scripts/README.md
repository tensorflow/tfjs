# Prepare Tensorflow model

**convert.py** is an python script that convert the Tensorflow SavedModel for web consumption.

## Usage

`python convert.py --saved_model_dir=/tmp/mobilenet/ --output_node_names='MobilenetV1/Predictions/Reshape_1' --output_graph=/tmp/mobilenet/optimized_model.pb --saved_model_tags=SERVE`

| Options         | Description                                                      | Default value |
|---|---|---|
|saved_model_dir  | The saved model directory                                        | |
|output_node_names| The names of the output nodes, comma separated                   | |
|output_graph     |The name of the output graph file                                 | |
|saved_model_tags |Tags of the MetaGraphDef to load, in comma separated string format| serve |

## Outputs

This script would generate two files, one is model topology, one is the weight collection.
For the above command example, generated files are:
1. optimized_model.pb (model)
2. otpimized_model.pb.weight (weight)

## Using the generated files
You need to make both model and weight files accessible through an url and feed them to the Model class.

```typescript
import {Model} from 'tfjs-converter';

const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/learnjs-data/tf_model_zoo/mobilenet_v1_1.0_224/';
const MODEL_FILE_URL = 'optimized_model.pb';
const WEIGHT_FILE_URL = 'optimized_model.pb.weight';

const model = new Model(MODEL_FILE_URL, WEIGHT_FILE_URL);
model.predict({...}) // run the inference on your model.
```
