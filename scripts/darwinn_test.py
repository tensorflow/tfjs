import sys
import os
import json

models = []
with open('/tmp/darwinn/metadata.json') as json_data:
   models = json.load(json_data)

for model in models:
    print(model)
    path = '/tmp/darwinn/' + model['model_name'] + '/'
    outputs = ','.join([x['name'] for x in model['outputs']])
    os.system('python3 scripts/convert.py --saved_model_dir=' + path
            + '  --output_graph=' + path + 'optimized_graph.pb' +
            ' --saved_model_tags=serve --output_node_names=' + outputs)
