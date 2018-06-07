/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// tslint:disable:max-line-length
import {io, ones, randomNormal, Scalar, scalar, serialization, sum, Tensor, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';

import {Model} from './engine/training';
import * as tfl from './index';
import {Reshape} from './layers/core';
import {deserialize} from './layers/serialization';
import {loadModelInternal, ModelAndWeightsConfig, modelFromJSON} from './models';
import {JsonDict} from './types';
import {convertPythonicToTs} from './utils/serialization_utils';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from './utils/test_utils';
import {version as layersVersion} from './version';

// tslint:enable:max-line-length

describeMathCPU('Nested model topology', () => {
  it('Nested Sequential model: Sequential as first layer', done => {
    const modelLevel1 = tfl.sequential(
        {layers: [tfl.layers.dense({units: 2, inputShape: [3]})]});
    const x = ones([1, 3]);
    const y = modelLevel1.predict(x) as Tensor;

    const modelLevel2 = tfl.sequential();
    modelLevel2.add(modelLevel1);
    expectTensorsClose(modelLevel2.predict(x) as Tensor, y);

    const modelLevel3 = tfl.sequential();
    modelLevel3.add(modelLevel2);
    expectTensorsClose(modelLevel3.predict(x) as Tensor, y);

    const xs = ones([8, 3]);
    const ys = zeros([8, 2]);
    modelLevel3.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    modelLevel3.fit(xs, ys)
        .then(history => {
          const newY = modelLevel1.predict(x) as Tensor;
          expectTensorsClose(modelLevel2.predict(x) as Tensor, newY);
          expectTensorsClose(modelLevel3.predict(x) as Tensor, newY);
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('Nested Sequential model: Functional model as first layer', done => {
    const input = tfl.input({shape: [3]});
    const output =
        tfl.layers.dense({units: 2}).apply(input) as tfl.SymbolicTensor;
    const modelLevel1 = tfl.model({inputs: input, outputs: output});
    const x = ones([1, 3]);
    const y = modelLevel1.predict(x) as Tensor;

    const modelLevel2 = tfl.sequential();
    modelLevel2.add(modelLevel1);
    expectTensorsClose(modelLevel2.predict(x) as Tensor, y);

    const modelLevel3 = tfl.sequential();
    modelLevel3.add(modelLevel2);
    expectTensorsClose(modelLevel3.predict(x) as Tensor, y);

    const xs = ones([8, 3]);
    const ys = zeros([8, 2]);
    modelLevel3.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    modelLevel3.fit(xs, ys)
        .then(history => {
          const newY = modelLevel1.predict(x) as Tensor;
          expectTensorsClose(modelLevel2.predict(x) as Tensor, newY);
          expectTensorsClose(modelLevel3.predict(x) as Tensor, newY);
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('Nested Sequential model: Sequential as second layer', done => {
    const innerModel = tfl.sequential(
        {layers: [tfl.layers.dense({units: 2, inputShape: [4]})]});
    const x = ones([1, 4]);
    const y = innerModel.predict(x) as Tensor;

    const x2By2 = ones([1, 2, 2]);

    const outerModel = tfl.sequential(
        {layers: [tfl.layers.reshape({targetShape: [4], inputShape: [2, 2]})]});
    outerModel.add(innerModel);
    expectTensorsClose(outerModel.predict(x2By2) as Tensor, y);

    const xs = ones([8, 2, 2]);
    const ys = zeros([8, 2]);
    outerModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    outerModel.fit(xs, ys)
        .then(history => {
          const newY = innerModel.predict(x) as Tensor;
          expectTensorsClose(outerModel.predict(x2By2) as Tensor, newY);
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('Nested Sequential model: Sequential as middle layer', done => {
    const innerModel = tfl.sequential(
        {layers: [tfl.layers.dense({units: 4, inputShape: [4]})]});
    const x = ones([1, 4]);
    const y = innerModel.predict(x) as Tensor;

    const x2By2 = ones([1, 2, 2]);

    const outerModel = tfl.sequential({
      layers: [
        tfl.layers.reshape({targetShape: [4], inputShape: [2, 2]}), innerModel,
        tfl.layers.reshape({targetShape: [2, 2]})
      ]
    });
    expectTensorsClose(
        outerModel.predict(x2By2) as Tensor, y.reshape([1, 2, 2]));

    const xs = ones([8, 2, 2]);
    const ys = zeros([8, 2, 2]);
    outerModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    outerModel.fit(xs, ys)
        .then(history => {
          const newY = innerModel.predict(x) as Tensor;
          expectTensorsClose(
              outerModel.predict(x2By2) as Tensor, newY.reshape([1, 2, 2]));
          done();
        })
        .catch(err => done.fail(err.stack));
  });

  it('getLayer() works for nested sequential model', () => {
    const innerModel = tfl.sequential({
      layers: [
        tfl.layers.dense({
          units: 4,
          inputShape: [4],
          activation: 'relu',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        }),
        tfl.layers.dense({
          units: 2,
          activation: 'tanh',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        })
      ]
    });
    const outerModel = tfl.sequential({
      layers: [
        innerModel,
        tfl.layers.dense(
            {units: 1, kernelInitializer: 'ones', biasInitializer: 'ones'})
      ]
    });
    expect(outerModel.getLayer(null, 0) instanceof tfl.Sequential)
        .toEqual(true);
    // Expect all-one values based on the kernel and bias initializers specified
    // above.
    expectTensorsClose(
        outerModel.getLayer(null, 1).getWeights()[0], ones([2, 1]));
    expectTensorsClose(outerModel.getLayer(null, 1).getWeights()[1], ones([1]));
    // Expect there to be only two layers.
    expect(() => outerModel.getLayer(null, 2)).toThrow();
  });

  it('getWeights() works for nested sequential model', () => {
    const innerModel = tfl.sequential({
      layers: [
        tfl.layers.dense({
          units: 4,
          inputShape: [4],
          activation: 'relu',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        }),
        tfl.layers.dense({
          units: 2,
          activation: 'tanh',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        })
      ]
    });
    const outerModel =
        tfl.sequential({layers: [innerModel, tfl.layers.dense({units: 1})]});
    const weights = outerModel.getWeights();
    expect(weights.length).toEqual(6);
    expect(weights[0].shape).toEqual([4, 4]);
    expect(weights[1].shape).toEqual([4]);
    expect(weights[2].shape).toEqual([4, 2]);
    expect(weights[3].shape).toEqual([2]);
    expect(weights[4].shape).toEqual([2, 1]);
    expect(weights[5].shape).toEqual([1]);
  });

  it('setWeights() works for nested sequential model', () => {
    const innerModel = tfl.sequential({
      layers: [
        tfl.layers.dense({
          units: 2,
          inputShape: [3],
          activation: 'relu',
          kernelInitializer: 'zeros',
          useBias: false
        }),
      ]
    });
    const outerModel = tfl.sequential({
      layers: [
        innerModel,
        tfl.layers.dense(
            {units: 1, kernelInitializer: 'zeros', useBias: false})
      ]
    });

    // The kernel start off as zeros. Set them all to ones.
    outerModel.setWeights([ones([3, 2]), ones([2, 1])]);
    expectTensorsClose(outerModel.getWeights()[0], ones([3, 2]));
    expectTensorsClose(outerModel.getWeights()[1], ones([2, 1]));
    expectTensorsClose(
        outerModel.predict(ones([1, 3])) as Tensor, tensor2d([[6]]));
  });

  it('Sequential as layer: save-load round trip', () => {
    const innerModel = tfl.sequential({
      layers: [
        tfl.layers.dense({
          units: 4,
          inputShape: [4],
          activation: 'relu',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        }),
        tfl.layers.dense({
          units: 4,
          activation: 'tanh',
          kernelInitializer: 'ones',
          biasInitializer: 'ones'
        })
      ]
    });
    const outerModel = tfl.sequential({
      layers: [
        tfl.layers.reshape({targetShape: [4], inputShape: [2, 2]}), innerModel,
        tfl.layers.dense(
            {units: 1, kernelInitializer: 'ones', biasInitializer: 'ones'})
      ]
    });
    const x = randomNormal([1, 2, 2]);
    const y = outerModel.predict(x) as Tensor;

    const unusedArg: {} = null;
    const returnString = false;
    const outerModelJSON = outerModel.toJSON(unusedArg, returnString);
    const reconstructedModel =
        deserialize(convertPythonicToTs(outerModelJSON) as JsonDict) as
        tfl.Sequential;
    expect(reconstructedModel.toJSON(unusedArg, returnString))
        .toEqual(outerModelJSON);
    expectTensorsClose(reconstructedModel.predict(x) as Tensor, y);
  });

  it('Functional model as layer: save-load round trip', () => {
    const input = tfl.input({shape: [4]});
    const layer1 = tfl.layers.dense({
      units: 4,
      activation: 'relu',
      kernelInitializer: 'ones',
      biasInitializer: 'ones'
    });
    const layer2 = tfl.layers.dense({
      units: 4,
      activation: 'tanh',
      kernelInitializer: 'ones',
      biasInitializer: 'ones'
    });
    const output = layer2.apply(layer1.apply(input)) as tfl.SymbolicTensor;
    const innerModel = tfl.model({inputs: input, outputs: output});
    const outerModel = tfl.sequential({
      layers: [
        tfl.layers.reshape({targetShape: [4], inputShape: [2, 2]}), innerModel,
        tfl.layers.reshape({targetShape: [2, 2]})
      ]
    });
    const x = randomNormal([1, 2, 2]);
    const y = outerModel.predict(x) as Tensor;

    const unusedArg: {} = null;
    const returnString = false;
    const outerModelJSON = outerModel.toJSON(unusedArg, returnString);
    const reconstructedModel =
        deserialize(convertPythonicToTs(outerModelJSON) as JsonDict) as
        tfl.Sequential;
    expect(reconstructedModel.toJSON(unusedArg, returnString))
        .toEqual(outerModelJSON);
    expectTensorsClose(reconstructedModel.predict(x) as Tensor, y);
  });

  it('Attempt to nest two-input functional model fails', () => {
    const input1 = tfl.input({shape: [4]}) as tfl.SymbolicTensor;
    const input2 = tfl.input({shape: [5]}) as tfl.SymbolicTensor;
    const output =
        tfl.layers.concatenate().apply([input1, input2]) as tfl.SymbolicTensor;
    const innerModel = tfl.model({inputs: [input1, input2], outputs: output});

    const outerModel = tfl.sequential();

    // Adding the two-input model as the first layer of the new model should
    // fail.
    expect(() => outerModel.add(innerModel))
        .toThrowError(/should have a single input tensor/);

    // Adding the two-input model as the second layer of the new model should
    // fail.
    const outerModel2 = tfl.sequential(
        {layers: [tfl.layers.dense({units: 4, inputShape: [8]})]});
    expect(() => outerModel2.add(innerModel))
        .toThrowError(/should have a single input tensor/);

    // Creating a sequential model consisting of only the two-input model as
    // a layer should fail.
    expect(() => tfl.sequential({
      layers: [innerModel]
    })).toThrowError(/should have a single input tensor/);
  });

  it('Attempt to nest two-output functional model fails', () => {
    const input = tfl.input({shape: [12]}) as tfl.SymbolicTensor;
    const output1 = tfl.layers.reshape({targetShape: [2, 6]}).apply(input) as
        tfl.SymbolicTensor;
    const output2 = tfl.layers.reshape({targetShape: [3, 4]}).apply(input) as
        tfl.SymbolicTensor;
    const innerModel = tfl.model({inputs: input, outputs: [output1, output2]});

    // Adding the two-output model as the first layer of the new model should
    // fail.
    const outerModel = tfl.sequential();
    expect(() => outerModel.add(innerModel))
        .toThrowError(/should have a single output tensor/);

    // Adding the two-output model as the second layer of the new model should
    // fail.
    const outerModel2 = tfl.sequential(
        {layers: [tfl.layers.dense({units: 4, inputShape: [8]})]});
    expect(() => outerModel2.add(innerModel))
        .toThrowError(/should have a single output tensor/);

    // Creating a sequential model consisting of only the two-output model as
    // a layer should fail.
    expect(() => tfl.sequential({
      layers: [innerModel]
    })).toThrowError(/should have a single output tensor/);
  });
});

describeMathCPU('model_from_json', () => {
  it('reconstitutes pythonic json string', done => {
    /* python generating code
        a=Input(shape=(32,))
        b=Dense(32)(a)
        model = Model(inputs=a, outputs=b, name="test")
        model.to_json())
    */
    modelFromJSON(fakeSequentialModel)
        .then(model => {
          expect(model.name).toEqual('test');
          const allZeros = zeros([1, 32]);
          expectTensorsClose(model.apply(allZeros) as Tensor, allZeros);
          done();
        })
        .catch(done.fail);
  });
  // TODO(bileschi): Uncomment once we are able to load the full RNN model.
  /*
  it('reconstitutes rnn model', done => {
    modelFromJSON(fakeRNNModel)
        .then(model => {
          console.log(model);
          done();
        })
        .catch(done.fail);
  });*/

  it('reconstitutes mnist non-sequential mode.', async done => {
    /*
    input_shape = (28,28,1)
    num_classes=10

    input_layer= Input(shape=input_shape)
    layer1 = Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape)
    layer1_result = layer1(input_layer)
    layer2= Conv2D(64, (3,3), activation='relu')
    layer2_result = layer2(layer1_result)
    layer3 = MaxPooling2D(pool_size=(2, 2))
    layer3_result = layer3(layer2_result)
    layer4 = Dropout(0.25)
    layer4_result = layer4(layer3_result)
    layer5 = Flatten()
    layer5_result = layer5(layer4_result)
    layer6 = Dense(128, activation='relu')
    layer6_result = layer6(layer5_result)
    layer7=Dropout(0.5)
    layer7_result = layer7(layer6_result)
    layer8=Dense(num_classes, activation='softmax')
    layer8_result = layer8(layer7_result)
    model = Model(inputs=input_layer, outputs=layer8_result, name='mnist')
    model.to_json()
      */
    modelFromJSON(fakeNonSequentialModel)
        .then(model => {
          expect(model.name).toEqual('mnist');
          expect(model.layers.length).toEqual(9);
          const prediction = model.predict(zeros([1, 28, 28, 1])) as Tensor;
          expect(prediction.shape).toEqual([1, 10]);
          expect(sum(prediction).dataSync()).toBeCloseTo(1);
          done();
        })
        .catch(done.fail);
  });
  it('reconstitutes mnist sequential mode.', async done => {
    /*
    input_shape = (28,28,1)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.to_json())
    */

    modelFromJSON(fakeMnistModel).then(async model => {
      expect(model.layers.length).toEqual(8);
      const prediction = model.predict(zeros([1, 28, 28, 1])) as Tensor;
      expect(prediction.shape).toEqual([1, 10]);
      expect(sum(prediction).dataSync()).toBeCloseTo(1);
      done();
    });
  });

  it('Serialization round-tripping', done => {
    modelFromJSON(fakeRoundtripModel)
        .then(model => {
          const serializedModel = model.toJSON() as string;
          // toJSON() returns a string by default.
          expect(typeof serializedModel).toEqual('string');
          const reparsedJson = JSON.parse(serializedModel);
          expect(reparsedJson['class_name'])
              .toEqual(fakeRoundtripModel.modelTopology['class_name']);
          // Intentionally skipping backend and keras_version fields.
          expect(reparsedJson['config'])
              .toEqual(fakeRoundtripModel.modelTopology['config']);
        })
        .then(done)
        .catch(done.fail);
  });

  it('toJSON with returnString = false', done => {
    modelFromJSON(fakeRoundtripModel)
        .then(model => {
          const serializedModel = model.toJSON(null, false) as JsonDict;
          expect(serializedModel['class_name'])
              .toEqual(fakeRoundtripModel.modelTopology['class_name']);
          expect(serializedModel['config'])
              .toEqual(fakeRoundtripModel.modelTopology['config']);
        })
        .then(done)
        .catch(done.fail);
  });

  it('toJSON return value includes correct versions', async () => {
    const model = await modelFromJSON(fakeRoundtripModel);
    const serializedModel = model.toJSON(null, false) as JsonDict;
    expect(serializedModel['keras_version'])
        .toEqual(`tfjs-layers ${layersVersion}`);
  });
});

describeMathCPU('loadModel from URL', () => {
  const setupFakeWeightFiles =
      (fileBufferMap:
           {[filename: string]: Float32Array|Int32Array|ArrayBuffer}) => {
        spyOn(window, 'fetch').and.callFake((path: string) => {
          return new Response(fileBufferMap[path]);
        });
      };

  const isModelConfigNestedValues = [false, true];
  const pathPrefixes = ['.', './', './model-home', './model-home/'];
  for (const isModelConfigNested of isModelConfigNestedValues) {
    for (const pathPrefix of pathPrefixes) {
      it(`pathPrefix=${pathPrefix}`, done => {
        const path0 = pathPrefix.endsWith('/') ? `${pathPrefix}weight_0` :
                                                 `${pathPrefix}/weight_0`;
        const path1 = pathPrefix.endsWith('/') ? `${pathPrefix}weight_1` :
                                                 `${pathPrefix}/weight_1`;
        const fileBufferMap:
            {[filename: string]: Float32Array|Int32Array|ArrayBuffer} = {};
        fileBufferMap[path0] =
            ones([32, 32], 'float32').dataSync() as Float32Array;
        fileBufferMap[path1] = ones([32], 'float32').dataSync() as Float32Array;
        setupFakeWeightFiles(fileBufferMap);
        // Use a randomly generated layer name to prevent interaction with
        // other unit tests that load the same sample JSON.
        const denseLayerName = 'dense_' + Math.floor(Math.random() * 1e9);
        const weightsManifest: io.WeightsManifestConfig = [
          {
            'paths': ['weight_0'],
            'weights': [{
              'name': `${denseLayerName}/kernel`,
              'dtype': 'float32',
              'shape': [32, 32]
            }],
          },
          {
            'paths': ['weight_1'],
            'weights': [{
              'name': `${denseLayerName}/bias`,
              'dtype': 'float32',
              'shape': [32]
            }],
          }
        ];
        // JSON.parse and stringify to deep copy fakeSequentialModel.
        let modelTopology =
            JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
        modelTopology['config']['layers'][1]['config']['name'] = denseLayerName;
        if (isModelConfigNested) {
          // Simulate case in which `configJson` contains not only
          // `model_config`, but also other data, such as training.
          modelTopology = {'model_config': modelTopology};
        }
        modelFromJSON({modelTopology, weightsManifest, pathPrefix})
            .then(model => {
              expectTensorsClose(
                  model.weights[0].read(), ones([32, 32], 'float32'));
              expectTensorsClose(
                  model.weights[1].read(), ones([32], 'float32'));
            })
            .then(done)
            .catch(done.fail);
      });
    }
  }

  it('load topology and weights from implicit relative http path',
     async done => {
       const modelTopology =
           JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
       const weightsManifest: io.WeightsManifestConfig = [
         {
           'paths': ['weight_0'],
           'weights': [
             {'name': `dense_6/kernel`, 'dtype': 'float32', 'shape': [32, 32]}
           ],
         },
         {
           'paths': ['weight_1'],
           'weights':
               [{'name': `dense_6/bias`, 'dtype': 'float32', 'shape': [32]}],
         }
       ];
       spyOn(window, 'fetch').and.callFake((path: string) => {
         if (path === 'model/model.json') {
           return new Response(JSON.stringify({
             modelTopology,
             weightsManifest,
           }));
         } else if (path === 'model/weight_0') {
           return new Response(
               ones([32, 32], 'float32').dataSync() as Float32Array);
         } else if (path === 'model/weight_1') {
           return new Response(
               zeros([32], 'float32').dataSync() as Float32Array);
         } else {
           throw new Error(`Invalid path: ${path}`);
         }
       });

       loadModelInternal('model/model.json')
           .then(model => {
             expect(model.layers.length).toEqual(2);
             expect(model.inputs.length).toEqual(1);
             expect(model.inputs[0].shape).toEqual([null, 32]);
             expect(model.outputs.length).toEqual(1);
             expect(model.outputs[0].shape).toEqual([null, 32]);
             const weightValues = model.getWeights();
             expect(weightValues.length).toEqual(2);
             expectTensorsClose(weightValues[0], ones([32, 32]));
             expectTensorsClose(weightValues[1], zeros([32]));
             done();
           })
           .catch(err => {
             done.fail(err.stack);
           });
     });

  it('load topology and weights from implicit relative http path: HDF5 format',
     async done => {
       const modelTopology =
           JSON.parse(JSON.stringify(fakeSequentialModelFromHDF5))
               .modelTopology;
       const weightsManifest: io.WeightsManifestConfig = [
         {
           'paths': ['weight_0'],
           'weights': [
             {'name': `dense_1/kernel`, 'dtype': 'float32', 'shape': [10, 2]}
           ],
         },
         {
           'paths': ['weight_1'],
           'weights':
               [{'name': `dense_1/bias`, 'dtype': 'float32', 'shape': [2]}],
         },
         {
           'paths': ['weight_2'],
           'weights': [
             {'name': `dense_2/kernel`, 'dtype': 'float32', 'shape': [2, 1]}
           ],
         },
         {
           'paths': ['weight_3'],
           'weights':
               [{'name': `dense_2/bias`, 'dtype': 'float32', 'shape': [1]}],
         }
       ];
       spyOn(window, 'fetch').and.callFake((path: string) => {
         if (path === 'model/model.json') {
           return new Response(JSON.stringify({
             modelTopology,
             weightsManifest,
           }));
         } else if (path === 'model/weight_0') {
           return new Response(
               ones([10, 2], 'float32').dataSync() as Float32Array);
         } else if (path === 'model/weight_1') {
           return new Response(
               zeros([2], 'float32').dataSync() as Float32Array);
         } else if (path === 'model/weight_2') {
           return new Response(
               zeros([2, 1], 'float32').dataSync() as Float32Array);
         } else if (path === 'model/weight_3') {
           return new Response(ones([1], 'float32').dataSync() as Float32Array);
         } else {
           throw new Error(`Invalid path: ${path}`);
         }
       });

       loadModelInternal('model/model.json')
           .then(model => {
             expect(model.layers.length).toEqual(2);
             expect(model.inputs.length).toEqual(1);
             expect(model.inputs[0].shape).toEqual([null, 10]);
             expect(model.outputs.length).toEqual(1);
             expect(model.outputs[0].shape).toEqual([null, 1]);
             const weightValues = model.getWeights();
             expect(weightValues.length).toEqual(4);
             expectTensorsClose(weightValues[0], ones([10, 2]));
             expectTensorsClose(weightValues[1], zeros([2]));
             expectTensorsClose(weightValues[2], zeros([2, 1]));
             expectTensorsClose(weightValues[3], ones([1]));
             done();
           })
           .catch(err => {
             done.fail(err.stack);
           });
     });


  it('load topology and weights with browserHTTPRequest with requestInit',
     async done => {
       const modelTopology =
           JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
       const weightsManifest: io.WeightsManifestConfig = [
         {
           'paths': ['weight_0'],
           'weights': [
             {'name': `dense_6/kernel`, 'dtype': 'float32', 'shape': [32, 32]}
           ],
         },
         {
           'paths': ['weight_1'],
           'weights':
               [{'name': `dense_6/bias`, 'dtype': 'float32', 'shape': [32]}],
         }
       ];

       const requestHeaders: Array<{}> = [];
       const requestCredentials: string[] = [];
       spyOn(window, 'fetch')
           .and.callFake((path: string, requestInit?: RequestInit) => {
             if (requestInit != null) {
               requestHeaders.push(requestInit.headers);
               requestCredentials.push(requestInit.credentials);
             }
             if (path === 'model/model.json') {
               return new Response(JSON.stringify({
                 modelTopology,
                 weightsManifest,
               }));
             } else if (path === 'model/weight_0') {
               return new Response(
                   ones([32, 32], 'float32').dataSync() as Float32Array);
             } else if (path === 'model/weight_1') {
               return new Response(
                   zeros([32], 'float32').dataSync() as Float32Array);
             } else {
               throw new Error(`Invalid path: ${path}`);
             }
           });

       loadModelInternal(io.browserHTTPRequest('model/model.json', {
         headers: {'header_key_1': 'header_value_1'},
         credentials: 'include',
       }))
           .then(model => {
             expect(model.layers.length).toEqual(2);
             expect(model.inputs.length).toEqual(1);
             expect(model.inputs[0].shape).toEqual([null, 32]);
             expect(model.outputs.length).toEqual(1);
             expect(model.outputs[0].shape).toEqual([null, 32]);
             const weightValues = model.getWeights();
             expect(weightValues.length).toEqual(2);
             expectTensorsClose(weightValues[0], ones([32, 32]));
             expectTensorsClose(weightValues[1], zeros([32]));

             // Verify that the headers and credentials are sent via
             // `fetch` properly.
             expect(requestHeaders).toEqual([
               {'header_key_1': 'header_value_1'},
               {'header_key_1': 'header_value_1'},
               {'header_key_1': 'header_value_1'}
             ]);
             expect(requestCredentials).toEqual([
               'include', 'include', 'include'
             ]);

             done();
           })
           .catch(err => {
             done.fail(err.stack);
           });
     });

  const httpProtocols = ['http://', 'https://'];
  for (const protocol of httpProtocols) {
    it(`load topology and weights: explicit relative ${protocol} path`,
       async done => {
         const modelTopology =
             JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
         const weightsManifest: io.WeightsManifestConfig = [
           {
             'paths': ['weight_0'],
             'weights': [{
               'name': `dense_6/kernel`,
               'dtype': 'float32',
               'shape': [32, 32]
             }],
           },
           {
             'paths': ['weight_1'],
             'weights':
                 [{'name': `dense_6/bias`, 'dtype': 'float32', 'shape': [32]}],
           }
         ];
         spyOn(window, 'fetch').and.callFake((path: string) => {
           if (path === `${protocol}localhost:8888/models/model.json`) {
             return new Response(JSON.stringify({
               modelTopology,
               weightsManifest,
             }));
           } else if (path === `${protocol}localhost:8888/models/weight_0`) {
             return new Response(
                 ones([32, 32], 'float32').dataSync() as Float32Array);
           } else if (path === `${protocol}localhost:8888/models/weight_1`) {
             return new Response(
                 zeros([32], 'float32').dataSync() as Float32Array);
           } else {
             throw new Error(`Invalid path: ${path}`);
           }
         });

         loadModelInternal(`${protocol}localhost:8888/models/model.json`)
             .then(model => {
               expect(model.layers.length).toEqual(2);
               expect(model.inputs.length).toEqual(1);
               expect(model.inputs[0].shape).toEqual([null, 32]);
               expect(model.outputs.length).toEqual(1);
               expect(model.outputs[0].shape).toEqual([null, 32]);
               const weightValues = model.getWeights();
               expect(weightValues.length).toEqual(2);
               expectTensorsClose(weightValues[0], ones([32, 32]));
               expectTensorsClose(weightValues[1], zeros([32]));
               done();
             })
             .catch(err => {
               done.fail(err.stack);
             });
       });
  }

  it('Missing weight in manifest leads to error', done => {
    setupFakeWeightFiles({
      './weight_0': ones([32, 32], 'float32').dataSync() as Float32Array,
      './weight_1': ones([32], 'float32').dataSync() as Float32Array,
    });
    // Use a randomly generated layer name to prevent interaction with other
    // unit tests that load the same sample JSON.
    const denseLayerName = 'dense_' + Math.floor(Math.random() * 1e9);
    const weightsManifest: io.WeightsManifestConfig = [
      {
        'paths': ['weight_0'],
        'weights': [{
          'name': `${denseLayerName}/kernel`,
          'dtype': 'float32',
          'shape': [32, 32]
        }],
      },
    ];  // Missing bias.
    // JSON.parse and stringify to deep copy fakeSequentialModel.
    const configJson =
        JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
    configJson['config']['layers'][1]['config']['name'] = denseLayerName;
    modelFromJSON({modelTopology: configJson, weightsManifest, pathPrefix: '.'})
        .then(() => done.fail())
        .catch(() => done());
  });

  it('Loads weights despite uniqueified tensor names', async done => {
    try {
      setupFakeWeightFiles({
        './weight_0': ones([32, 32], 'float32').dataSync() as Float32Array,
        './weight_1': ones([32], 'float32').dataSync() as Float32Array,
      });
      const denseLayerName = 'dense_uniqueify';
      const weightsManifest: io.WeightsManifestConfig = [
        {
          'paths': ['weight_0'],
          'weights': [{
            'name': `${denseLayerName}/kernel`,
            'dtype': 'float32',
            'shape': [32, 32]
          }],
        },
        {
          'paths': ['weight_1'],
          'weights': [{
            'name': `${denseLayerName}/bias`,
            'dtype': 'float32',
            'shape': [32]
          }],
        }
      ];
      // JSON.parse and stringify to deep copy fakeSequentialModel.
      const configJson =
          JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
      configJson['config']['layers'][1]['config']['name'] = denseLayerName;
      const model1 = await modelFromJSON(
          {modelTopology: configJson, weightsManifest, pathPrefix: '.'});
      expect(model1.weights[0].name).toEqual('dense_uniqueify/kernel');
      expect(model1.weights[0].originalName).toEqual('dense_uniqueify/kernel');
      expect(model1.weights[1].name).toEqual('dense_uniqueify/bias');
      expect(model1.weights[1].originalName).toEqual('dense_uniqueify/bias');
      expectTensorsClose(model1.weights[0].read(), ones([32, 32], 'float32'));
      expectTensorsClose(model1.weights[1].read(), ones([32], 'float32'));

      // On the second load, the variable names will be uniqueified. This
      // test succeeds only because we maintain the name mapping, so we
      // can load weights--keyed by non-unique names in the weight
      // manifest--into variables with newly uniqueified names.
      const model2 = await modelFromJSON(
          {modelTopology: configJson, weightsManifest, pathPrefix: '.'});
      // note unique suffix
      expect(model2.weights[0].name).toEqual('dense_uniqueify/kernel_1');
      expect(model2.weights[0].originalName).toEqual('dense_uniqueify/kernel');
      // note unique suffix
      expect(model2.weights[1].name).toEqual('dense_uniqueify/bias_1');
      expect(model2.weights[1].originalName).toEqual('dense_uniqueify/bias');
      expectTensorsClose(model2.weights[0].read(), ones([32, 32], 'float32'));
      expectTensorsClose(model2.weights[1].read(), ones([32], 'float32'));
      done();
    } catch (e) {
      done.fail(e.stack);
    }
  });

  it('Repeated saving and loading of Model works', () => {
    const model1 = tfl.sequential();
    model1.add(
        tfl.layers.dense({units: 3, inputShape: [4], activation: 'relu'}));
    model1.add(
        tfl.layers.dense({units: 1, inputShape: [4], activation: 'sigmoid'}));
    const json1 = model1.toJSON(null, false);
    const model2 =
        deserialize(convertPythonicToTs(json1) as serialization.ConfigDict) as
        Model;
    const json2 = model2.toJSON(null, false);
    expect(json2).toEqual(json1);
  });
});

describeMathCPU('loadModel from IOHandler', () => {
  // The model topology JSON can be obtained with the following Python Keras
  // code:
  //
  // ```python
  // import keras
  // model = keras.Sequential([
  //     keras.layers.Dense(1, input_shape=[4], activation='sigmoid')
  // ])
  // print(model.to_json())
  // ```
  const modelTopology: {} = {
    'class_name': 'Sequential',
    'keras_version': '2.1.4',
    'config': [{
      'class_name': 'Dense',
      'config': {
        'kernel_initializer': {
          'class_name': 'VarianceScaling',
          'config': {
            'distribution': 'uniform',
            'scale': 1.0,
            'seed': null,
            'mode': 'fan_avg'
          }
        },
        'name': 'dense_1',
        'kernel_constraint': null,
        'bias_regularizer': null,
        'bias_constraint': null,
        'dtype': 'float32',
        'activation': 'sigmoid',
        'trainable': true,
        'kernel_regularizer': null,
        'bias_initializer': {'class_name': 'Zeros', 'config': {}},
        'units': 1,
        'batch_input_shape': [null, 4],
        'use_bias': true,
        'activity_regularizer': null
      }
    }],
    'backend': 'tensorflow'
  };
  const weightSpecs: io.WeightsManifestEntry[] = [
    {
      name: 'dense_1/kernel',
      shape: [4, 1],
      dtype: 'float32',
    },
    {
      name: 'dense_1/bias',
      shape: [1],
      dtype: 'float32',
    }
  ];
  const weightData = new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5]).buffer;

  // A dummy IOHandler that returns hard-coded model artifacts when its
  // `load` method is called.
  class IOHandlerForTest implements io.IOHandler {
    private readonly includeWeights: boolean;

    constructor(includeWeights = true) {
      this.includeWeights = includeWeights;
    }

    async load(): Promise<io.ModelArtifacts> {
      return this.includeWeights ? {modelTopology, weightSpecs, weightData} :
                                   {modelTopology};
    }
  }

  // A dummy IOHandler that doesn't have the `load` method implemented and
  // is expected to cause `loadModel` or `loadModelInternal` to fail.
  class IOHandlerWithoutLoad implements io.IOHandler {
    constructor() {}
  }

  it('load topology and weights', async done => {
    loadModelInternal(new IOHandlerForTest(true))
        .then(model => {
          expect(model.layers.length).toEqual(1);
          expect(model.inputs.length).toEqual(1);
          expect(model.inputs[0].shape).toEqual([null, 4]);
          expect(model.outputs.length).toEqual(1);
          expect(model.outputs[0].shape).toEqual([null, 1]);
          const weightValues = model.getWeights();
          expect(weightValues.length).toEqual(2);
          expectTensorsClose(
              weightValues[0], tensor2d([1.1, 2.2, 3.3, 4.4], [4, 1]));
          expectTensorsClose(weightValues[1], tensor1d([5.5]));
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('load topology only', async done => {
    loadModelInternal(new IOHandlerForTest(false))
        .then(model => {
          expect(model.layers.length).toEqual(1);
          expect(model.inputs.length).toEqual(1);
          expect(model.inputs[0].shape).toEqual([null, 4]);
          expect(model.outputs.length).toEqual(1);
          expect(model.outputs[0].shape).toEqual([null, 1]);
          done();
        })
        .catch(err => {
          done.fail(err.stack);
        });
  });

  it('IOHandler without load method causes error', async done => {
    loadModelInternal(new IOHandlerWithoutLoad())
        .then(model => {
          done.fail(
              'Loading with an IOHandler without load method succeeded ' +
              'unexpectedly.');
        })
        .catch(err => {
          expect(err.message).toMatch(/does not have .*load.* method/);
          done();
        });
  });
});

describeMathCPUAndGPU('Sequential', () => {
  const inputShape = [1, 6];
  const batchInputShape = [1].concat(inputShape);
  const firstReshape = [2, 3];
  const secondReshape = [3, 2];
  const layers = [
    new Reshape({targetShape: firstReshape, batchInputShape, name: 'layer1'}),
    new Reshape({targetShape: secondReshape, name: 'layer2'})
  ];

  function getInputs(): Tensor {
    return ones(batchInputShape);
  }

  function getExpectedOutputs(): Tensor {
    return ones([1].concat(secondReshape));
  }

  it('throws an exception if the first layer is not an input layer', () => {
    const layer = new Reshape({targetShape: firstReshape});
    expect(() => tfl.sequential({layers: [layer]}))
        .toThrowError(
            /The first layer in a Sequential model must get an `inputShape`/);
  });

  it('can accept a list of layers in constructor', () => {
    const model = tfl.sequential({layers});
    expect(model.layers).toEqual(layers);
  });

  it('can add layers', () => {
    const model = tfl.sequential();
    for (const layer of layers) {
      model.add(layer);
    }
    expect(model.layers).toEqual(layers);
  });

  it('can pop layers', () => {
    const model = tfl.sequential({layers});
    model.pop();
    expect(model.layers).toEqual(layers.slice(0, 1));
  });

  it('throws error if try to pop too many layers', () => {
    const model = tfl.sequential();
    expect(() => model.pop()).toThrowError(/There are no layers in the model/);
  });

  it('apply() threads data through the model.', () => {
    const model = tfl.sequential({layers});
    expectTensorsClose(
        model.apply(getInputs()) as Tensor, getExpectedOutputs());
  });

  it('predict() threads data through the model.', () => {
    const model = tfl.sequential({layers});
    expectTensorsClose(
        model.predict(getInputs()) as Tensor, getExpectedOutputs());
  });

  it('predictOnBatch() threads data through the model.', () => {
    const batchSize = 10;
    const inputShape = [1, 6];
    const batchInputShape = [batchSize].concat(inputShape);
    const firstReshape = [2, 3];
    const secondReshape = [3, 2];
    const layers = [
      new Reshape({targetShape: firstReshape, batchInputShape, name: 'layer1'}),
      new Reshape({targetShape: secondReshape, name: 'layer2'})
    ];
    const inputBatch = ones([batchSize].concat(inputShape));
    const expectedOutput = ones([batchSize].concat(secondReshape));
    const model = tfl.sequential({layers});
    expectTensorsClose(
        model.predictOnBatch(inputBatch) as Tensor, expectedOutput);
  });

  it('compile() and fit()', async () => {
    const batchSize = 5;
    const inputSize = 4;
    const xs = ones([batchSize, inputSize]);
    const ys = ones([batchSize, 1]);
    const denseLayer1 = tfl.layers.dense({
      units: 3,
      useBias: false,
      kernelInitializer: 'ones',
      inputShape: [inputSize]
    });
    const denseLayer2 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const model = tfl.sequential({layers: [denseLayer1, denseLayer2]});
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const history = await model.fit(xs, ys, {batchSize, epochs: 2});
    expect(history.history['loss'][0]).toBe(121);
    expect(history.history['loss'][1]).toBeCloseTo(0.015178224071860313);
  });

  it('Calling evaluate before compile leads to error', () => {
    const batchSize = 5;
    const inputSize = 4;
    const denseLayer1 = tfl.layers.dense({units: 3, inputShape: [inputSize]});
    const model = tfl.sequential({layers: [denseLayer1]});
    const xs = ones([batchSize, inputSize]);
    const ys = ones([batchSize, 1]);
    expect(() => model.evaluate(xs, ys))
        .toThrowError(/needs to be compiled before/);
  });

  it('compile() and evaluate()', () => {
    const batchSize = 5;
    const inputSize = 4;
    const xs = ones([batchSize, inputSize]);
    const ys = ones([batchSize, 1]);
    const denseLayer1 = tfl.layers.dense({
      units: 3,
      useBias: false,
      kernelInitializer: 'ones',
      inputShape: [inputSize]
    });
    const denseLayer2 =
        tfl.layers.dense({units: 1, useBias: false, kernelInitializer: 'ones'});
    const model = tfl.sequential({layers: [denseLayer1, denseLayer2]});
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
    const losses = model.evaluate(xs, ys, {batchSize}) as Scalar;
    expectTensorsClose(losses, scalar(121));
  });

  it('getConfig returns an Array', () => {
    const model = tfl.sequential({layers});
    const config = model.getConfig();
    expect(Array.isArray(config)).toEqual(true);
    expect(config.length).toEqual(layers.length);
  });
});

// Fake models.

const fakeSequentialModel: ModelAndWeightsConfig = {
  modelTopology: {
    'class_name': 'Model',
    'keras_version': '2.0.7',
    'config': {
      'layers': [
        {
          'class_name': 'InputLayer',
          'config': {
            'dtype': 'float32',
            'batch_input_shape': [null, 32],
            'name': 'input_6',
            'sparse': false
          },
          'inbound_nodes': [],
          'name': 'input_6'
        },
        {
          'class_name': 'Dense',
          'config': {
            'units': 32,
            'bias_constraint': null,
            'use_bias': true,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'activation': 'linear',
            'bias_regularizer': null,
            'activity_regularizer': null,
            'trainable': true,
            'kernel_constraint': null,
            'kernel_regularizer': null,
            'name': 'dense_6',
            'bias_initializer': {'class_name': 'Zeros', 'config': {}}
          },
          'inbound_nodes': [[['input_6', 0, 0, {}]]],
          'name': 'dense_6'
        }
      ],
      'input_layers': [['input_6', 0, 0]],
      'output_layers': [['dense_6', 0, 0]],
      'name': 'test'
    },
    'backend': 'tensorflow'
  }
};

const fakeSequentialModelFromHDF5: ModelAndWeightsConfig = {
  modelTopology: {
    'backend': 'tensorflow',
    'keras_version': '2.1.4',
    'model_config': {
      'class_name': 'Sequential',
      'config': [
        {
          'class_name': 'Dense',
          'config': {
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'name': 'dense_1',
            'kernel_constraint': null,
            'bias_regularizer': null,
            'bias_constraint': null,
            'dtype': 'float32',
            'activation': 'relu',
            'trainable': true,
            'kernel_regularizer': null,
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'units': 2,
            'batch_input_shape': [null, 10],
            'use_bias': true,
            'activity_regularizer': null
          }
        },
        {
          'class_name': 'Dense',
          'config': {
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'seed': null,
                'mode': 'fan_avg'
              }
            },
            'name': 'dense_2',
            'kernel_constraint': null,
            'bias_regularizer': null,
            'bias_constraint': null,
            'activation': 'sigmoid',
            'trainable': true,
            'kernel_regularizer': null,
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'units': 1,
            'use_bias': true,
            'activity_regularizer': null
          }
        }
      ]
    },
  }
};

// TODO(bileschi): Uncomment once we are able to load a model of this size.
/*
const fakeRNNModel: ModelAndWeightsConfig = {
  modelTopology: {
    'keras_version': '2.1.5',
    'training_config': {
      'optimizer_config': {
        'config': {
          'rho': 0.8999999761581421,
          'epsilon': 1e-07,
          'lr': 0.009999999776482582,
          'decay': 0.0
        },
        'class_name': 'RMSprop'
      },
      'loss': 'categorical_crossentropy',
      'metrics': ['acc'],
      'sample_weight_mode': null,
      'loss_weights': null
    },
    'backend': 'tensorflow',
    'model_config': {
      'config': [
        {
          'config': {
            'unroll': false,
            'name': 'lstm_9',
            'trainable': true,
            'implementation': 1,
            'recurrent_constraint': null,
            'stateful': false,
            'return_sequences': false,
            'recurrent_initializer': {
              'config': {'gain': 1.0, 'seed': null},
              'class_name': 'Orthogonal'
            },
            'bias_regularizer': null,
            'kernel_constraint': null,
            'dtype': 'float32',
            'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
            'kernel_regularizer': null,
            'go_backwards': false,
            'units': 256,
            'recurrent_regularizer': null,
            'kernel_initializer': {
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              },
              'class_name': 'VarianceScaling'
            },
            'use_bias': true,
            'bias_constraint': null,
            'activation': 'tanh',
            'recurrent_activation': 'hard_sigmoid',
            'recurrent_dropout': 0.0,
            'return_state': false,
            'dropout': 0.0,
            'batch_input_shape': [null, 10, 393],
            'activity_regularizer': null,
            'unit_forget_bias': true
          },
          'class_name': 'LSTM'
        },
        {
          'config': {
            'rate': 0.2,
            'name': 'dropout_4',
            'noise_shape': null,
            'seed': null,
            'trainable': true
          },
          'class_name': 'Dropout'
        },
        {
          'config': {
            'name': 'dense_4',
            'trainable': true,
            'use_bias': true,
            'bias_constraint': null,
            'units': 393,
            'kernel_initializer': {
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              },
              'class_name': 'VarianceScaling'
            },
            'activation': 'linear',
            'bias_regularizer': null,
            'kernel_constraint': null,
            'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
            'activity_regularizer': null,
            'kernel_regularizer': null
          },
          'class_name': 'Dense'
        },
        {
          'config': {
            'name': 'activation_4',
            'trainable': true,
            'activation': 'softmax'
          },
          'class_name': 'Activation'
        }
      ],
      'class_name': 'Sequential'
    }
  }
} */


const fakeNonSequentialModel: ModelAndWeightsConfig = {
  modelTopology: {
    'backend': 'tensorflow',
    'class_name': 'Model',
    'keras_version': '2.1.1',
    'config': {
      'name': 'mnist',
      'output_layers': [['dense_16', 0, 0]],
      'layers': [
        {
          'class_name': 'InputLayer',
          'name': 'input_6',
          'inbound_nodes': [],
          'config': {
            'batch_input_shape': [null, 28, 28, 1],
            'sparse': false,
            'name': 'input_6',
            'dtype': 'float32'
          }
        },
        {
          'class_name': 'Conv2D',
          'name': 'conv2d_15',
          'inbound_nodes': [[['input_6', 0, 0, {}]]],
          'config': {
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'padding': 'valid',
            'use_bias': true,
            'strides': [1, 1],
            'bias_regularizer': null,
            'activity_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'data_format': 'channels_last',
            'dilation_rate': [1, 1],
            'kernel_constraint': null,
            'kernel_regularizer': null,
            'kernel_size': [3, 3],
            'activation': 'relu',
            'name': 'conv2d_15',
            'filters': 32,
            'trainable': true,
            'bias_constraint': null
          }
        },
        {
          'class_name': 'Conv2D',
          'name': 'conv2d_16',
          'inbound_nodes': [[['conv2d_15', 0, 0, {}]]],
          'config': {
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'padding': 'valid',
            'use_bias': true,
            'strides': [1, 1],
            'bias_regularizer': null,
            'activity_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'data_format': 'channels_last',
            'dilation_rate': [1, 1],
            'kernel_constraint': null,
            'kernel_regularizer': null,
            'kernel_size': [3, 3],
            'activation': 'relu',
            'name': 'conv2d_16',
            'filters': 64,
            'trainable': true,
            'bias_constraint': null
          }
        },
        {
          'class_name': 'MaxPooling2D',
          'name': 'max_pooling2d_8',
          'inbound_nodes': [[['conv2d_16', 0, 0, {}]]],
          'config': {
            'padding': 'valid',
            'strides': [2, 2],
            'pool_size': [2, 2],
            'data_format': 'channels_last',
            'name': 'max_pooling2d_8',
            'trainable': true
          }
        },
        {
          'class_name': 'Dropout',
          'name': 'dropout_15',
          'inbound_nodes': [[['max_pooling2d_8', 0, 0, {}]]],
          'config': {
            'rate': 0.25,
            'noise_shape': null,
            'name': 'dropout_15',
            'trainable': true,
            'seed': null
          }
        },
        {
          'class_name': 'Flatten',
          'name': 'flatten_8',
          'inbound_nodes': [[['dropout_15', 0, 0, {}]]],
          'config': {'name': 'flatten_8', 'trainable': true}
        },
        {
          'class_name': 'Dense',
          'name': 'dense_15',
          'inbound_nodes': [[['flatten_8', 0, 0, {}]]],
          'config': {
            'use_bias': true,
            'bias_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'kernel_constraint': null,
            'bias_constraint': null,
            'kernel_regularizer': null,
            'activation': 'relu',
            'name': 'dense_15',
            'activity_regularizer': null,
            'trainable': true,
            'units': 128
          }
        },
        {
          'class_name': 'Dropout',
          'name': 'dropout_16',
          'inbound_nodes': [[['dense_15', 0, 0, {}]]],
          'config': {
            'rate': 0.5,
            'noise_shape': null,
            'name': 'dropout_16',
            'trainable': true,
            'seed': null
          }
        },
        {
          'class_name': 'Dense',
          'name': 'dense_16',
          'inbound_nodes': [[['dropout_16', 0, 0, {}]]],
          'config': {
            'use_bias': true,
            'bias_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'kernel_constraint': null,
            'bias_constraint': null,
            'kernel_regularizer': null,
            'activation': 'softmax',
            'name': 'dense_16',
            'activity_regularizer': null,
            'trainable': true,
            'units': 10
          }
        }
      ],
      'input_layers': [['input_6', 0, 0]]
    }
  }
};

const fakeMnistModel: ModelAndWeightsConfig = {
  modelTopology: {
    'backend': 'tensorflow',
    'config': [
      {
        'config': {
          'kernel_size': [3, 3],
          'use_bias': true,
          'batch_input_shape': [null, 28, 28, 1],
          'filters': 32,
          'kernel_regularizer': null,
          'dilation_rate': [1, 1],
          'strides': [1, 1],
          'padding': 'valid',
          'bias_constraint': null,
          'kernel_constraint': null,
          'data_format': 'channels_last',
          'trainable': true,
          'activation': 'relu',
          'dtype': 'float32',
          'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
          'bias_regularizer': null,
          'name': 'conv2d_1',
          'kernel_initializer': {
            'config': {
              'scale': 1.0,
              'mode': 'fan_avg',
              'seed': null,
              'distribution': 'uniform'
            },
            'class_name': 'VarianceScaling'
          },
          'activity_regularizer': null
        },
        'class_name': 'Conv2D'
      },
      {
        'config': {
          'kernel_size': [3, 3],
          'use_bias': true,
          'filters': 64,
          'kernel_regularizer': null,
          'dilation_rate': [1, 1],
          'strides': [1, 1],
          'padding': 'valid',
          'bias_constraint': null,
          'data_format': 'channels_last',
          'trainable': true,
          'activation': 'relu',
          'kernel_constraint': null,
          'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
          'bias_regularizer': null,
          'name': 'conv2d_2',
          'kernel_initializer': {
            'config': {
              'scale': 1.0,
              'mode': 'fan_avg',
              'seed': null,
              'distribution': 'uniform'
            },
            'class_name': 'VarianceScaling'
          },
          'activity_regularizer': null
        },
        'class_name': 'Conv2D'
      },
      {
        'config': {
          'strides': [2, 2],
          'padding': 'valid',
          'pool_size': [2, 2],
          'data_format': 'channels_last',
          'trainable': true,
          'name': 'max_pooling2d_1'
        },
        'class_name': 'MaxPooling2D'
      },
      {
        'config': {
          'seed': null,
          'name': 'dropout_1',
          'trainable': true,
          'noise_shape': null,
          'rate': 0.25
        },
        'class_name': 'Dropout'
      },
      {
        'config': {'name': 'flatten_1', 'trainable': true},
        'class_name': 'Flatten'
      },
      {
        'config': {
          'use_bias': true,
          'units': 128,
          'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
          'kernel_regularizer': null,
          'bias_regularizer': null,
          'trainable': true,
          'activation': 'relu',
          'bias_constraint': null,
          'kernel_constraint': null,
          'name': 'dense_1',
          'kernel_initializer': {
            'config': {
              'scale': 1.0,
              'mode': 'fan_avg',
              'seed': null,
              'distribution': 'uniform'
            },
            'class_name': 'VarianceScaling'
          },
          'activity_regularizer': null
        },
        'class_name': 'Dense'
      },
      {
        'config': {
          'seed': null,
          'name': 'dropout_2',
          'trainable': true,
          'noise_shape': null,
          'rate': 0.5
        },
        'class_name': 'Dropout'
      },
      {
        'config': {
          'use_bias': true,
          'units': 10,
          'bias_initializer': {'config': {}, 'class_name': 'Zeros'},
          'kernel_regularizer': null,
          'bias_regularizer': null,
          'trainable': true,
          'activation': 'softmax',
          'bias_constraint': null,
          'kernel_constraint': null,
          'name': 'dense_2',
          'kernel_initializer': {
            'config': {
              'scale': 1.0,
              'mode': 'fan_avg',
              'seed': null,
              'distribution': 'uniform'
            },
            'class_name': 'VarianceScaling'
          },
          'activity_regularizer': null
        },
        'class_name': 'Dense'
      }
    ],
    'keras_version': '2.1.1',
    'class_name': 'Sequential'
  }
};

const fakeRoundtripModel: ModelAndWeightsConfig = {
  modelTopology: {
    'backend': 'tensorflow',
    'class_name': 'Model',
    'keras_version': '2.1.1',
    'config': {
      'name': 'mnist',
      'output_layers': [['dense_16', 0, 0]],
      'layers': [
        {
          'class_name': 'InputLayer',
          'name': 'input_6',
          'inbound_nodes': [],
          'config': {
            'batch_input_shape': [null, 28, 28, 1],
            'sparse': false,
            'name': 'input_6',
            'dtype': 'float32'
          }
        },
        {
          'class_name': 'Conv2D',
          'name': 'conv2d_15',
          'inbound_nodes': [[['input_6', 0, 0, {}]]],
          'config': {
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'padding': 'valid',
            'use_bias': true,
            'strides': [1, 1],
            'bias_regularizer': null,
            'activity_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'data_format': 'channels_last',
            'dilation_rate': [1, 1],
            'kernel_constraint': null,
            'kernel_regularizer': null,
            'kernel_size': [3, 3],
            'activation': 'relu',
            'name': 'conv2d_15',
            'filters': 32,
            'trainable': true,
            'bias_constraint': null
          }
        },
        {
          'class_name': 'Conv2D',
          'name': 'conv2d_16',
          'inbound_nodes': [[['conv2d_15', 0, 0, {}]]],
          'config': {
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'padding': 'valid',
            'use_bias': true,
            'strides': [1, 1],
            'bias_regularizer': null,
            'activity_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'data_format': 'channels_last',
            'dilation_rate': [1, 1],
            'kernel_constraint': null,
            'kernel_regularizer': null,
            'kernel_size': [3, 3],
            'activation': 'relu',
            'name': 'conv2d_16',
            'filters': 64,
            'trainable': true,
            'bias_constraint': null
          }
        },
        {
          'class_name': 'MaxPooling2D',
          'name': 'max_pooling2d_8',
          'inbound_nodes': [[['conv2d_16', 0, 0, {}]]],
          'config': {
            'padding': 'valid',
            'strides': [2, 2],
            'pool_size': [2, 2],
            'data_format': 'channels_last',
            'name': 'max_pooling2d_8',
            'trainable': true
          }
        },
        {
          'class_name': 'Dropout',
          'name': 'dropout_15',
          'inbound_nodes': [[['max_pooling2d_8', 0, 0, {}]]],
          'config': {
            'rate': 0.25,
            'noise_shape': null,
            'name': 'dropout_15',
            'trainable': true,
            'seed': null
          }
        },
        {
          'class_name': 'Flatten',
          'name': 'flatten_8',
          'inbound_nodes': [[['dropout_15', 0, 0, {}]]],
          'config': {'name': 'flatten_8', 'trainable': true}
        },
        {
          'class_name': 'Dense',
          'name': 'dense_15',
          'inbound_nodes': [[['flatten_8', 0, 0, {}]]],
          'config': {
            'use_bias': true,
            'bias_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'kernel_constraint': null,
            'bias_constraint': null,
            'kernel_regularizer': null,
            'activation': 'relu',
            'name': 'dense_15',
            'activity_regularizer': null,
            'trainable': true,
            'units': 128
          }
        },
        {
          'class_name': 'Dropout',
          'name': 'dropout_16',
          'inbound_nodes': [[['dense_15', 0, 0, {}]]],
          'config': {
            'rate': 0.5,
            'noise_shape': null,
            'name': 'dropout_16',
            'trainable': true,
            'seed': null
          }
        },
        {
          'class_name': 'Dense',
          'name': 'dense_16',
          'inbound_nodes': [[['dropout_16', 0, 0, {}]]],
          'config': {
            'use_bias': true,
            'bias_regularizer': null,
            'kernel_initializer': {
              'class_name': 'VarianceScaling',
              'config': {
                'distribution': 'uniform',
                'scale': 1.0,
                'mode': 'fan_avg',
                'seed': null
              }
            },
            'bias_initializer': {'class_name': 'Zeros', 'config': {}},
            'kernel_constraint': null,
            'bias_constraint': null,
            'kernel_regularizer': null,
            'activation': 'softmax',
            'name': 'dense_16',
            'activity_regularizer': null,
            'trainable': true,
            'units': 10
          }
        }
      ],
      'input_layers': [['input_6', 0, 0]]
    }
  }
};
