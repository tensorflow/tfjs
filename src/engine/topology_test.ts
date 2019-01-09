/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {eye, memory, ones, Tensor, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import * as initializers from '../initializers';
import {Shape} from '../keras_format/types';
import {NamedTensorMap} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
import {LayerVariable, onesVariable, zerosVariable} from '../variables';

import {loadWeightsFromJson, loadWeightsFromNamedTensorMap} from './container';
import {InputSpec, Layer, LayerArgs, Node} from './topology';

class LayerForTest extends tfl.layers.Layer {
  static className = 'LayerForTest';
  constructor(args: LayerArgs) {
    super(args);
  }
}

describe('InputSpec', () => {
  it('initializes with expected default values.', () => {
    const inputSpec = new InputSpec({});
    expect(inputSpec.dtype).toBeUndefined();
    expect(inputSpec.shape).toBeUndefined();
    expect(inputSpec.ndim).toBeUndefined();
    expect(inputSpec.maxNDim).toBeUndefined();
    expect(inputSpec.minNDim).toBeUndefined();
    expect(inputSpec.axes).toEqual({});
  });

  it('initializes with inputSpec.ndim = shape.length when shape is specified.',
     () => {
       const shape = [1, 2, 3];
       const expectedValue = shape.length;
       const inputSpec = new InputSpec({shape: [1, 2, 3], ndim: -1});
       expect(inputSpec.ndim).toEqual(expectedValue);
     });

  it('initializes inputSpec.axes when axes specified.', () => {
    const expectedValue = {1: 2};
    const inputSpec = new InputSpec({axes: expectedValue});
    expect(inputSpec.axes).toEqual(expectedValue);
  });
});

describe('Node', () => {
  const outboundLayerName = 'outboundLayer';
  const inboundLayerName = 'inboundLayer';
  const outboundLayer = new LayerForTest({name: outboundLayerName});
  const inboundLayers = [new LayerForTest({name: inboundLayerName})];
  const nodeIndices = [0];
  const tensorIndices = [0];
  const inputTensors = [new tfl.SymbolicTensor('float32', [1], null, [], {})];
  const outputTensors =
      [new tfl.SymbolicTensor('float32', [2, 2], null, [], {})];
  const inputMasks = [zeros([1])];
  const outputMasks = [zeros([1])];
  const inputShapes = [[1]];
  const outputShapes = [[1], [1]];
  const callArgs = {mask: zeros([1])};
  const node = new Node(
      {
        outboundLayer,
        inboundLayers,
        nodeIndices,
        tensorIndices,
        inputTensors,
        outputTensors,
        inputMasks,
        outputMasks,
        inputShapes,
        outputShapes
      },
      callArgs);

  it('initializes object as expected.', () => {
    expect(node.outboundLayer).toEqual(outboundLayer);
    expect(node.inboundLayers).toEqual(inboundLayers);
    expect(node.nodeIndices).toEqual(nodeIndices);
    expect(node.tensorIndices).toEqual(tensorIndices);
    expect(node.inputTensors).toEqual(inputTensors);
    expect(node.outputTensors).toEqual(outputTensors);
    expect(node.inputMasks).toEqual(inputMasks);
    expect(node.outputMasks).toEqual(outputMasks);
    expect(node.inputShapes).toEqual(inputShapes);
    expect(node.outputShapes).toEqual(outputShapes);
    expect(node.callArgs).toEqual(callArgs);
    expect(inboundLayers[0].outboundNodes).toEqual([node]);
    expect(node.outboundLayer.inboundNodes).toEqual([node]);
  });

  it('generates expected SerializableNodeConfig.', () => {
    const nodeConfig = node.getConfig();
    expect(nodeConfig.outboundLayer).toEqual(outboundLayerName);
    expect(nodeConfig.inboundLayers).toEqual([inboundLayerName]);
    expect(nodeConfig.nodeIndices).toEqual(nodeIndices);
    expect(nodeConfig.tensorIndices).toEqual(tensorIndices);
  });

  it('generates unique IDs', () => {
    const secondNode = new Node(
        {
          outboundLayer,
          inboundLayers,
          nodeIndices,
          tensorIndices,
          inputTensors,
          outputTensors,
          inputMasks,
          outputMasks,
          inputShapes,
          outputShapes
        },
        callArgs);
    expect(secondNode.id).not.toEqual(node.id);
  });
});

describeMathCPU('Layer', () => {
  describe('initialized to its defaults', () => {
    // TODO(bileschi): This should be tfl.layers.Layer for some future version
    // of TS that doesn't fail to compile.
    let defaultLayer: Layer;

    beforeEach(() => {
      defaultLayer = new LayerForTest({});
    });

    it('has a default layer name of layer_....', () => {
      expect(defaultLayer.name).toMatch(/^layer_.+$/);
    });

    it('has null inputSpecs.', () => {
      expect(defaultLayer.inputSpec).toBeNull();
    });

    it('does not support masking (supportsMasking == false).', () => {
      expect(defaultLayer.supportsMasking).toEqual(false);
    });

    it('is trainable.', () => {
      expect(defaultLayer.trainable).toEqual(true);
    });

    it('has an undefined batchInputShape.', () => {
      expect(defaultLayer.batchInputShape).toBeUndefined();
    });

    it('has an undefined dtype.', () => {
      expect(defaultLayer.dtype).toBeUndefined();
    });

    it('has null initialWeights.', () => {
      expect(defaultLayer.initialWeights).toBeNull();
    });

    it('has an empty inboundNodes list.', () => {
      expect(defaultLayer.inboundNodes).toEqual([]);
    });

    it('has an empty outboundNodes list.', () => {
      expect(defaultLayer.outboundNodes).toEqual([]);
    });

    it('has an empty losses list.', () => {
      expect(defaultLayer.losses).toEqual([]);
    });

    it('has an empty updates list.', () => {
      expect(defaultLayer.updates).toEqual([]);
    });

    it('is not built (built == false).', () => {
      expect(defaultLayer.built).toEqual(false);
    });

    it('has an empty trainableWeights list.', () => {
      expect(defaultLayer.trainableWeights).toEqual([]);
    });

    it('has an empty nonTrainableWeights list.', () => {
      expect(defaultLayer.nonTrainableWeights).toEqual([]);
    });

    it('has an empty weights list.', () => {
      expect(defaultLayer.weights).toEqual([]);
    });

    it('produces a unique ID', () => {
      const secondLayer = new LayerForTest({});
      expect(defaultLayer.id).not.toEqual(secondLayer.id);
    });

    it('stateful is false by default', () => {
      const layer = new LayerForTest({});
      expect(layer.stateful).toBe(false);
    });

    it('returns null if it doesn`t support masking and no mask is passed in.',
       () => {
         expect(defaultLayer.computeMask([], null)).toBeNull();
       });

    it('throws exception if it doesn`t support masking and a ' +
           'mask is passed in.',
       () => {
         const mask = ones([1]);
         expect(() => defaultLayer.computeMask([], mask))
             .toThrowError(/does not support masking/);
       });

    it('returns the same mask passed in if it supports masking', () => {
      const mask = ones([1]);
      defaultLayer.supportsMasking = true;
      expect(defaultLayer.computeMask([], mask)).toEqual(mask);
    });
    it('correctly generates a config for serialization', () => {
      const config = defaultLayer.getConfig();
      expect(config.name).toEqual(defaultLayer.name);
      expect(config.trainable).toEqual(defaultLayer.trainable);
      expect(config.batchInputShape).toBeUndefined();
      expect(config.dtype).toBeUndefined();
    });
  });

  describe('A layer with non-default arguments', () => {
    it('initializes layer with given name.', () => {
      const name = 'layer name';
      const layer = new LayerForTest({name});
      expect(layer.name).toMatch(name);
      const config = layer.getConfig();
      expect(config.name).toEqual(layer.name);
    });

    for (const trainable of [true, false]) {
      it('initializes layer as trainable, if specified.', () => {
        const layer = new LayerForTest({trainable});
        expect(layer.trainable).toEqual(trainable);
        const config = layer.getConfig();
        expect(config.trainable).toEqual(layer.trainable);
      });
    }

    for (const batchInputShape of [[], [1]]) {
      it('initializes batchInputShape to layerConfig.batchInputShape.', () => {
        const layer = new LayerForTest({batchInputShape});
        expect(layer.batchInputShape).toEqual(batchInputShape);
        const config = layer.getConfig();
        expect(config.batchInputShape).toEqual(layer.batchInputShape);
      });
    }

    it('initializes batchInputShape to layerConfig.batchInputShape even if ' +
           'layerConfig.inputShape is defined.',
       () => {
         const batchInputShape = [1];
         const inputShape = [2, 3];
         const layer = new LayerForTest({batchInputShape, inputShape});
         expect(layer.batchInputShape).toEqual(batchInputShape);
       });

    for (const [batchSize, inputShape, expectedBatchInputShape] of [
             [null, [], [null]], [null, [1], [null, 1]], [3, [], [3]],
             [3, [1], [3, 1]]]) {
      it('initializes batchInputShape to layerConfig.inputShape.', () => {
        const layer = new LayerForTest(
            {batchSize: batchSize as number, inputShape: inputShape as Shape});
        expect(layer.batchInputShape).toEqual(expectedBatchInputShape as Shape);
      });
    }

    it('initializes dtype to float32 if layerConfig.inputShape is set.', () => {
      const layer = new LayerForTest({inputShape: []});
      expect(layer.dtype).toEqual('float32');
      const config = layer.getConfig();
      expect(config.dtype).toEqual(layer.dtype);
    });

    it('initializes dtype to float32 if layerConfig.batchInputShape is set.',
       () => {
         const layer = new LayerForTest({batchInputShape: []});
         expect(layer.dtype).toEqual('float32');
       });

    it('initializes initialWeights if present.', () => {
      const weights = [zeros([1])];
      const layer = new LayerForTest({weights});
      expect(layer.initialWeights).toEqual(weights);
    });

    it('Layer with duplicate weight names throws error', () => {
      class LayerForTest extends tfl.layers.Layer {
        static className = 'LayerForTest';
        constructor(args: LayerArgs) {
          super(args);
          this.addWeight(
              'foo', [1, 2], 'float32', initializers.getInitializer('zeros'));
          this.addWeight(
              'foo', [2, 3], 'float32', initializers.getInitializer('zeros'));
        }
      }
      expect(() => new LayerForTest({}))
          .toThrowError(/[Dd]uplicate weight name/);
    });
  });


  it('can be set to built.', () => {
    const layer = new LayerForTest({});
    layer.built = true;
    expect(layer.built).toEqual(true);
  });

  // Weights used for subsequent tests
  const trainableWeights = [zerosVariable([1])];
  const nonTrainableWeights = [onesVariable([1])];
  it('can set trainableWeights.', () => {
    const layer = new LayerForTest({});
    layer.trainableWeights = trainableWeights;
    expect(layer.trainableWeights).toEqual(trainableWeights);
  });

  it('doesn\'t return trainableWeights if layer is not trainable, even ' +
         'if they exist',
     () => {
       const layer = new LayerForTest({trainable: false});
       layer.trainableWeights = trainableWeights;
       expect(layer.trainableWeights).toEqual([]);
     });

  it('can set nonTrainableWeights.', () => {
    const layer = new LayerForTest({});
    layer.nonTrainableWeights = nonTrainableWeights;
    expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
  });

  it('only returns nonTrainableWeights for nonTrainableWeights if the layer ' +
         'is trainable.',
     () => {
       const layer = new LayerForTest({trainable: true});
       layer.trainableWeights = trainableWeights;
       layer.nonTrainableWeights = nonTrainableWeights;
       expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
     });

  it('concats trainable and nonTrainableWeights for nonTrainableWeights if ' +
         'not trainable.',
     () => {
       const layer = new LayerForTest({trainable: false});
       const expectedWeights = trainableWeights.concat(nonTrainableWeights);
       layer.trainableWeights = trainableWeights;
       layer.nonTrainableWeights = nonTrainableWeights;
       expect(layer.nonTrainableWeights).toEqual(expectedWeights);
     });

  for (const trainable of [true, false]) {
    it('concats trainable and nonTrainableWeights for weights regardless of ' +
           'whether the layer is trainable trainable.',
       () => {
         const layer = new LayerForTest({trainable});
         const expectedWeights = trainableWeights.concat(nonTrainableWeights);
         layer.trainableWeights = trainableWeights;
         layer.nonTrainableWeights = nonTrainableWeights;
         expect(layer.weights).toEqual(expectedWeights);
       });
  }

  describe('assertInputCompatibility()', () => {
    function runAssert(
        layer: Layer,
        inputs: Tensor|Tensor[]|tfl.SymbolicTensor|tfl.SymbolicTensor[]) {
      // tslint:disable-next-line:no-any
      (layer as any).assertInputCompatibility(inputs);
    }
    const testInputs = [
      () => ones([1]), () => [ones([1])],
      () => new tfl.SymbolicTensor('float32', [1], null, [], {}),
      () => [new tfl.SymbolicTensor('float32', [1], null, [], {})]
    ];

    for (const inputs of testInputs) {
      it('doesn\'t raise an exception if no inputSpec is provided.', () => {
        const layer = new LayerForTest({});
        runAssert(layer, inputs());
      });

      it('doesn\'t raise exception if number of inputs == number of ' +
             'inputSpecs.',
         () => {
           const inputSpecs = [new InputSpec({})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if number of inputs != number of inputSpecs.',
         () => {
           const inputSpecs = [new InputSpec({}), new InputSpec({})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs()))
               .toThrowError(/expects [0-9]+ inputs/);
         });

      it('doesn\'t raise exception if inputs\' ndim == inputSpecs.ndim.',
         () => {
           const inputSpecs = [new InputSpec({ndim: 1})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if inputs\' ndim != inputSpecs.ndim.', () => {
        const inputSpecs = [new InputSpec({ndim: 2})];
        const layer = new LayerForTest({});
        layer.inputSpec = inputSpecs;
        expect(() => runAssert(layer, inputs())).toThrowError(/expected ndim=/);
      });

      it('doesn\'t raise exception if inputs\' ndim <= inputSpecs.maxNdim.',
         () => {
           const inputSpecs = [new InputSpec({maxNDim: 1})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if inputs\' ndim > inputSpecs.maxNdim.', () => {
        const inputSpecs = [new InputSpec({maxNDim: 0})];
        const layer = new LayerForTest({});
        layer.inputSpec = inputSpecs;
        expect(() => runAssert(layer, inputs()))
            .toThrowError(/expected max_ndim=/);
      });

      it('doesn\'t raise exception if inputs\' ndim >= inputSpecs.minNdim.',
         () => {
           const inputSpecs = [new InputSpec({minNDim: 1})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if inputs\' ndim < inputSpecs.minNdim.', () => {
        const inputSpecs = [new InputSpec({minNDim: 2})];
        const layer = new LayerForTest({});
        layer.inputSpec = inputSpecs;
        expect(() => runAssert(layer, inputs()))
            .toThrowError(/expected min_ndim=/);
      });

      it('doesn\'t raise exception if inputs\' dtype == inputSpecs.dtype.',
         () => {
           const inputSpecs = [new InputSpec({dtype: 'float32'})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      // TODO(michaelterry): Add dtype test once more dtypes supported.

      it('doesn\'t raise exception if inputs\' dimensions == inputSpecs.axes.',
         () => {
           const inputSpecs = [new InputSpec({axes: {0: 1}})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if inputs\' dimensions != inputSpecs.axes.', () => {
        const inputSpecs = [new InputSpec({axes: {0: 2}})];
        const layer = new LayerForTest({});
        layer.inputSpec = inputSpecs;
        expect(() => runAssert(layer, inputs())).toThrowError(/expected axis/);
      });

      it('throws exception if inputs\' dimensions don\'t have the same ' +
             'number of inputSpecs.axes.',
         () => {
           const inputSpecs = [new InputSpec({axes: {0: 1, 2: 1}})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs()))
               .toThrowError(/expected axis/);
         });

      it('doesn\'t raise exception if inputs\' shape == inputSpecs.shape.',
         () => {
           const inputSpecs = [new InputSpec({shape: [1]})];
           const layer = new LayerForTest({});
           layer.inputSpec = inputSpecs;
           expect(() => runAssert(layer, inputs())).not.toThrowError();
         });

      it('throws exception if inputs\' shape != inputSpecs.shape.', () => {
        const inputSpecs = [new InputSpec({shape: [2]})];
        const layer = new LayerForTest({});
        layer.inputSpec = inputSpecs;
        expect(() => runAssert(layer, inputs())).toThrowError(/expected shape/);
      });
    }
  });

  describe('apply() passed 1 SymbolicTensor', () => {
    const firstLayer = new LayerForTest({name: 'firstLayer'});
    const secondLayer = new LayerForTest({name: 'secondLayer'});
    const callArgs = {a: 1};
    const singleSymbolicTensor =
        new tfl.SymbolicTensor('float32', [1], firstLayer, [], {});
    const returnedTensor =
        secondLayer.apply(singleSymbolicTensor, callArgs) as tfl.SymbolicTensor;

    it('returns a SymbolicTensor.', () => {
      expect(returnedTensor instanceof tfl.SymbolicTensor).toBe(true);
    });

    it('returns a SymbolicTensor with a reference to the source layer.', () => {
      expect(returnedTensor.sourceLayer).toEqual(secondLayer);
    });

    it('returns a SymbolicTensor with a reference to the inputs passed ' +
           'to apply().',
       () => {
         expect(returnedTensor.inputs).toEqual([singleSymbolicTensor]);
         expect(returnedTensor.callArgs).toEqual(callArgs);
       });

    it('returns a SymbolicTensor with nodeIndex and tensorIndex set.', () => {
      expect(returnedTensor.nodeIndex).toBeDefined();
      expect(returnedTensor.tensorIndex).toBeDefined();
    });

    it('returns a SymbolicTensor with the name set.', () => {
      expect(returnedTensor.name).toMatch(/secondLayer/);
    });

    it('is built.', () => {
      expect(secondLayer.built).toBe(true);
    });

    it('Incompatible inputShape leads to warning', () => {
      let recordedWarnMessage: string;
      spyOn(console, 'warn')
          .and.callFake((message: string) => recordedWarnMessage = message);
      const layer1 = tfl.layers.dense({units: 2, inputShape: [5]});
      layer1.apply(tfl.input({shape: [4]}));
      expect(recordedWarnMessage)
          .toMatch(/shape of the input tensor .*null,4.* not match .*null,5.*/);
    });

    it('Incompatible inputShape leads to warning: batchInputShape', () => {
      let recordedWarnMessage: string;
      spyOn(console, 'warn')
          .and.callFake((message: string) => recordedWarnMessage = message);
      const layer1 = tfl.layers.dense({units: 2, batchInputShape: [2, 3, 5]});
      layer1.apply(tfl.input({shape: [4, 5]}));
      expect(recordedWarnMessage)
          .toMatch(
              /shape of the input tensor .*null,4,5.* not match .*2,3,5.*/);
    });

    it('Incompatible inputShape rank leads to warning', () => {
      let recordedWarnMessage: string;
      spyOn(console, 'warn')
          .and.callFake((message: string) => recordedWarnMessage = message);
      const layer1 = tfl.layers.dense({units: 2, inputShape: [5]});
      layer1.apply(tfl.input({shape: [4, 3]}));
      expect(recordedWarnMessage)
          .toMatch(/rank .*null,4,3.* does not match .*null,5.*/);
    });

    it('Incompatible inputShape rank leads to warning: batchInputShape', () => {
      let recordedWarnMessage: string;
      spyOn(console, 'warn')
          .and.callFake((message: string) => recordedWarnMessage = message);
      const layer1 = tfl.layers.dense({units: 2, batchInputShape: [3, 5]});
      layer1.apply(tfl.input({shape: [4, 3]}));
      expect(recordedWarnMessage)
          .toMatch(/rank .*null,4,3.* does not match .*3,5.*/);
    });

    it('Compatible inputShape leads to NO warning', () => {
      let recordedWarnMessage: string;
      spyOn(console, 'warn')
          .and.callFake((message: string) => recordedWarnMessage = message);
      const layer1 = tfl.layers.dense({units: 2, inputShape: [5]});
      layer1.apply(tfl.input({shape: [5]}));
      expect(recordedWarnMessage).toEqual(undefined);
    });
  });

  describe('apply() passed >1 SymbolicTensor', () => {
    it('throws an exception for multiple symbolic inputs.', () => {
      const firstLayer = new LayerForTest({name: 'first layer'});
      const secondLayer = new LayerForTest({name: 'second layer'});
      const symbolicTensorList = [
        new tfl.SymbolicTensor(
            'float32', [1], firstLayer, [], {}, 'first_symbolic_tensor'),
        new tfl.SymbolicTensor(
            'float32', [1], firstLayer, [], {}, 'second_symbolic_tensor')
      ];
      // TODO(michaelterry): Update this once multiple symbolic tensors are
      // allowed.
      expect(() => secondLayer.apply(symbolicTensorList)).toThrowError();
    });
  });

  describe('apply() passed SymbolicTensor and Tensor', () => {
    it('throws an exception.', () => {
      const layer = new LayerForTest({});
      const inputs = [
        new tfl.SymbolicTensor(
            'float32', [1], null, [], {}, 'first_symbolic_tensor'),
        ones([1])
      ];
      expect(() => layer.apply(inputs as Tensor[]))
          .toThrowError(/must be all SymbolicTensors or all Tensors/);
    });
  });

  it('apply() returns multiple symbolic tensors for multiple ' +
         'output shapes',
     () => {
       const layer = new LayerForTest({});
       const outputShapes = [[1], [2, 3]];
       const input = new tfl.SymbolicTensor('float32', [1], null, [], {});
       // tslint:disable-next-line:no-any
       spyOn((layer as any), 'computeOutputShape').and.callFake(() => {
         return outputShapes;
       });
       const results = layer.apply(input) as tfl.SymbolicTensor[];
       expect(results.length).toEqual(2);
       expect(results.map(x => x.shape)).toEqual(outputShapes);
       expect(results.map(x => x.outputTensorIndex)).toEqual([0, 1]);
     });

  describe('apply() passed 1+ Tensors', () => {
    it('returns new values for output if the same as the input.', () => {
      const anArray = ones([1]);
      // Test with both an Tensor and an array of Tensors.
      for (const inputs of [anArray, [anArray, anArray]]) {
        const layer = new LayerForTest({});
        const result = layer.apply(inputs) as Tensor | Tensor[];

        expect(result instanceof Tensor || (result[0] instanceof Tensor))
            .toBe(true);

        expect(layer.built).toBe(true);

        if (result instanceof Array) {
          const inputArray = inputs as Tensor[];
          for (let i = 0; i < result.length; i++) {
            expectTensorsClose(result[i], inputArray[i]);
          }
        } else {
          expectTensorsClose(result, inputs as Tensor);
        }
        expect(result === inputs).toBe(false);
      }
    });
  });

  describe('initialized with weights at construction time', () => {
    it('sets those weights after calling apply().', () => {
      const initialWeights = eye(2);
      const arrayInput = zeros([1]);
      const symbolicInput =
          new tfl.SymbolicTensor('float32', [1], null, [], {});
      // Test with symbolic and concrete input.
      for (const inputs of [arrayInput, symbolicInput]) {
        const layer = new LayerForTest({weights: [initialWeights]});
        // Fake the build() method to test assignment to initialWeights.
        // tslint:disable-next-line:no-any
        spyOn((layer as any), 'build').and.callFake(() => {
          layer.built = true;
          layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
        });
        expect(layer.weights.length).toEqual(0);
        layer.apply(inputs);
        expect(layer.weights.length).toEqual(1);
        expectTensorsClose(layer.weights[0].read(), initialWeights);
      }
    });
  });

  describe('apply() (nodes)', () => {
    it('doesn\'t change inboundNodes or outboundNodes when called with ' +
           'concrete input',
       () => {
         const layer = new LayerForTest({});
         expect(layer.inboundNodes.length).toEqual(0);
         expect(layer.outboundNodes.length).toEqual(0);
         layer.apply(eye(1));
         expect(layer.inboundNodes.length).toEqual(0);
         expect(layer.outboundNodes.length).toEqual(0);
       });

    it('changes inboundNodes and outboundNodes when called with ' +
           'symbolic input',
       () => {
         const layer = new LayerForTest({});
         const input = new tfl.SymbolicTensor('float32', [1], null, [], {});
         expect(layer.inboundNodes.length).toEqual(0);
         expect(layer.outboundNodes.length).toEqual(0);
         layer.apply(input);
         expect(layer.inboundNodes.length).toEqual(1);
         expect(layer.outboundNodes.length).toEqual(0);
         expect(layer.inboundNodes[0].outboundLayer).toEqual(layer);
       });

    it('updates inbound and outboundNodes when there are multiple layers',
       () => {
         const firstLayer = new LayerForTest({name: 'first_layer'});
         const secondLayer = new LayerForTest({name: 'second_layer'});
         const initialInput =
             new tfl.SymbolicTensor('float32', [1], null, [], {});
         const firstOutput = firstLayer.apply(initialInput);
         secondLayer.apply(firstOutput);

         expect(firstLayer.inboundNodes.length).toEqual(1);
         expect(firstLayer.outboundNodes.length).toEqual(1);
         expect(secondLayer.inboundNodes.length).toEqual(1);
         expect(secondLayer.outboundNodes.length).toEqual(0);
         expect(firstLayer.outboundNodes[0].outboundLayer).toEqual(secondLayer);
       });
  });

  describe('Layer.outputShape', () => {
    it('Layers with one output', () => {
      const layer = tfl.layers.dense({units: 3});
      layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
      expect(layer.outputShape).toEqual([null, 3]);
    });

    it('Layers with two outputs', () => {
      const layer = tfl.layers.simpleRNN({units: 3, returnState: true});
      layer.apply(
          new tfl.SymbolicTensor('float32', [null, 4, 5], null, [], {}));
      expect(layer.outputShape).toEqual([[null, 3], [null, 3]]);
    });

    it('Layers with two inboundNodes of the same outputShape', () => {
      const layer = tfl.layers.dense({units: 3});
      layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
      layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
      expect(layer.inboundNodes.length).toEqual(2);
      expect(layer.outputShape).toEqual([null, 3]);
    });

    it('Layers with two inboundNodes of different outputShapes', () => {
      const layer = tfl.layers.dense({units: 3});
      layer.apply(
          new tfl.SymbolicTensor('float32', [null, 5, 4], null, [], {}));
      layer.apply(
          new tfl.SymbolicTensor('float32', [null, 6, 4], null, [], {}));
      expect(layer.inboundNodes.length).toEqual(2);
      expect(() => layer.outputShape)
          .toThrowError(/has multiple inbound nodes/);
    });

    it('Unbuilt layer throws Error', () => {
      const layer = tfl.layers.dense({units: 3});
      expect(() => layer.outputShape).toThrowError(/has never been called/);
    });
  });

  describe('Layer.countParams', () => {
    it('Layers with weights', () => {
      const units = 3;
      const inputSize = 4;
      const layer = tfl.layers.dense({units});
      layer.apply(zeros([1, inputSize]));
      const numParams = layer.countParams();
      expect(numParams).toEqual(units * inputSize + units);
    });

    it('Layer without weights', () => {
      const layer = tfl.layers.flatten();
      layer.apply(zeros([2, 2, 2]));
      const numParams = layer.countParams();
      expect(numParams).toEqual(0);
    });
  });

  describe('setWeights', () => {
    it('throws exception if weights are not the same length ' +
           'as existing weights',
       () => {
         const layer = new LayerForTest({});
         layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
         const onesTensor = ones([1]);
         expect(() => layer.setWeights([
           onesTensor, onesTensor
         ])).toThrowError(/with a weight list of length/);
       });

    it('throws exception if weights are not the same shape ' +
           'as existing weights',
       () => {
         const layer = new LayerForTest({});
         const onesTensor = ones([1]);
         layer.trainableWeights = [new LayerVariable(zeros([2, 2]))];
         expect(() => layer.setWeights([onesTensor]))
             .toThrowError(/not compatible with provided weight shape/);
       });

    it('updates weights.', () => {
      const layer = new LayerForTest({});
      const onesTensor = ones([1]);
      layer.trainableWeights = [new LayerVariable(zeros([1]))];
      layer.setWeights([onesTensor]);
      expectTensorsClose(layer.trainableWeights[0].read(), onesTensor);
    });
  });

  describe('computeOutputShape()', () => {
    it('returns the inputShape in the base class', () => {
      const layer = new LayerForTest({});
      const shape = [1];
      expect(layer.computeOutputShape(shape)).toEqual(shape);
    });
  });

  describe('input and output properties: ', () => {
    let input: tfl.SymbolicTensor;
    let layer: Layer;
    let output: tfl.SymbolicTensor;

    beforeEach(() => {
      input =
          new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
      layer = new LayerForTest({});
      output = layer.apply(input) as tfl.SymbolicTensor;
    });

    it('input retrieves layer\'s inputs.', () => {
      expect(layer.input).toEqual(input);
    });

    it('input retrieves layer\'s outputs.', () => {
      expect(layer.output).toEqual(output);
    });

    it('input throws exception if there is more than one input', () => {
      const secondInput =
          new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
      layer.apply(secondInput);
      expect(() => layer.input).toThrowError(/"layer input" is ill-defined/);
    });

    it('output throws exception if there is more than one output', () => {
      const secondInput =
          new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
      layer.apply(secondInput);
      expect(() => layer.output).toThrowError(/"layer output" is ill-defined/);
    });
  });

  describe('getInputAt and getOutputAt: ', () => {
    let input: tfl.SymbolicTensor;
    let layer: Layer;
    let output: tfl.SymbolicTensor;

    beforeEach(() => {
      input =
          new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
      layer = new LayerForTest({});
      output = layer.apply(input) as tfl.SymbolicTensor;
    });

    it('getInputAt() retrieves layer\'s inputs.', () => {
      expect(layer.getInputAt(0)).toEqual(input);
    });

    it('getOutputAt() retrieves layer\'s outputs.', () => {
      expect(layer.getOutputAt(0)).toEqual(output);
    });

    it('getInputAt() throws exception ask for incorrect index.', () => {
      expect(() => layer.getInputAt(1))
          .toThrowError(/Asked to get input at node 1, but/);
    });

    it('getOutputAt() throws exception ask for incorrect index.', () => {
      expect(() => layer.getOutputAt(1))
          .toThrowError(/Asked to get output at node 1, but/);
    });
  });
});

describeMathCPUAndGPU('Layer-dispose', () => {
  it('Dispose Dense Layer before build leads to Error', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    expect(() => dense.dispose()).toThrowError(/has not been built/);
  });

  it('Dispose Dense Layer after one tensor call frees memory', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    dense.apply(zeros([2, 4]));
    const numTensors0 = memory().numTensors;
    const result = dense.dispose();

    expect(result.refCountAfterDispose).toEqual(0);
    expect(result.numDisposedVariables).toEqual(2);
    // Two variables should have been freed: the kernel and the bias.
    expect(memory().numTensors).toEqual(numTensors0 - 2);
  });

  it('Symbolic apply() call after Dense disposal leads to Error', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    dense.apply(zeros([2, 4]));
    const result = dense.dispose();
    // This dispose() call should dispose the layer.

    expect(result.refCountAfterDispose).toEqual(0);
    expect(result.numDisposedVariables).toEqual(2);
    expect(
        () => dense.apply(
            new tfl.SymbolicTensor('float32', [2, 4], null, [], {})))
        .toThrowError(/Layer .* is already disposed/);
  });

  it('Non-symbolic apply() call after Dense disposal leads to Error', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    dense.apply(zeros([2, 4]));
    dense.dispose();  // This dispose() call should dispose the layer.

    expect(() => dense.apply(ones([2, 4])))
        .toThrowError(/Layer .* is already disposed/);
  });

  it('Calling defRec() repeatedly for two-Node Layer frees memory', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {}));
    dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {}));
    const numTensors0 = memory().numTensors;

    const result1 = dense.dispose();
    // After the first dispose call, no memory should have been freed.
    expect(memory().numTensors).toEqual(numTensors0);
    expect(result1.refCountAfterDispose).toEqual(1);
    expect(result1.numDisposedVariables).toEqual(0);

    const result2 = dense.dispose();
    // After the second dispose call, memory for the kernel and the bias should
    // have been freed.
    expect(memory().numTensors).toEqual(numTensors0 - 2);
    expect(result2.refCountAfterDispose).toEqual(0);
    expect(result2.numDisposedVariables).toEqual(2);
  });

  it('Calling dispose on already-disposed Layer leads to Error', () => {
    const dense = tfl.layers.dense({units: 1, inputShape: [4]});
    dense.apply(zeros([2, 4]));
    dense.dispose();
    expect(() => dense.dispose()).toThrowError(/Layer .* is already disposed/);
  });

  it('Symbolic apply() call after Flatten disposal leads to Error', () => {
    const dense = tfl.layers.flatten();
    dense.apply(zeros([2, 3, 4]));
    dense.dispose();  // This dispose() call should dispose the layer.

    expect(
        () => dense.apply(
            new tfl.SymbolicTensor('float32', [2, 4], null, [], {})))
        .toThrowError(/Layer .* is already disposed/);
  });

  it('Non-symbolic apply() call after Flatten disposal leads to Error', () => {
    const dense = tfl.layers.flatten();
    dense.apply(zeros([2, 3, 4]));
    dense.dispose();  // This dispose() call should dispose the layer.

    expect(() => dense.apply(zeros([2, 3, 4])))
        .toThrowError(/Layer .* is already disposed/);
  });

  it('dispose() call works on Input Layer', () => {
    const input = tfl.layers.input({shape: [2, 3]}) as tfl.SymbolicTensor;
    const output = tfl.layers.reshape({targetShape: [3, 2]}).apply(input) as
        tfl.SymbolicTensor;
    const model = tfl.model({inputs: [input], outputs: [output]});

    const result = model.dispose();
    // This model, consiting of only an input layer and a reshape layer, does
    // not have any weights to dispose.
    expect(result.numDisposedVariables).toEqual(0);
    expect(() => model.predict(zeros([1, 2, 3])))
        .toThrowError(/already disposed/);
  });
});

// TODO(cais): Maybe remove this test once loadWeightsFromJson is removed
//   (b/74015805).
describeMathCPUAndGPU('loadWeightsFromJson', () => {
  const inputTensor =
      tfl.layers.input({shape: [3], name: 'inputLayer', dtype: 'float32'});

  it('One layer', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    denseLayer.apply(inputTensor);
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer': [
          {
            'name': 'denseLayer/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },
          {
            'name': 'denseLayer/bias:0',
            'dtype': 'float32',
            'shape': [2],
            'value': [-0.1, -0.2],
          },
        ],
      },
    };
    loadWeightsFromJson(weightsJSON, [denseLayer]);
    // Run a concrete input value through the layer to check that the weights
    // are loaded properly.
    expectTensorsClose(
        denseLayer.apply(tensor2d([[1, 1, 1]], [1, 3])) as Tensor,
        tensor2d([[0.8, 1.0]], [1, 2]));
  });

  it('Two layers', () => {
    const denseLayer1 =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer1'});
    const denseLayer2 =
        tfl.layers.dense({units: 1, useBias: false, name: 'denseLayer2'});
    denseLayer2.apply(denseLayer1.apply(inputTensor));
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer1': [
          {
            'name': 'denseLayer1/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },
          {
            'name': 'denseLayer1/bias:0',
            'dtype': 'float32',
            'shape': [2],
            'value': [-0.1, -0.2],
          },
        ],
        'denseLayer2': [
          {
            'name': 'denseLayer2/kernel:0',
            'dtype': 'float32',
            'shape': [2, 1],
            'value': [[1.2], [1.3]],
          },
        ],
      },
    };
    loadWeightsFromJson(weightsJSON, [denseLayer1, denseLayer2]);
    // Run a concrete input value through the layer to check that the weights
    // are loaded properly.
    expectTensorsClose(
        denseLayer2.apply(denseLayer1.apply(tensor2d([[1, 1, 1]], [1, 3]))) as
            Tensor,
        tensor2d([[2.26]], [1, 1]));
  });

  it('Missing weights for a layer', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    denseLayer.apply(inputTensor);
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {},
    };
    expect(() => {
      loadWeightsFromJson(weightsJSON, [denseLayer]);
    })
        .toThrowError(
            /Layer.*denseLayer.*expects 2 weight.*but.*have 0 element.*/);
  });

  it('Missing a single weight', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    denseLayer.apply(inputTensor);
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer': [
          {
            'name': 'denseLayer1/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },
          {
            'name': 'denseLayer1/bias:0',
            'dtype': 'float32',
            'shape': [1],
            'value': [-0.1],
          },
        ],
      }
    };
    expect(() => {
      loadWeightsFromJson(weightsJSON, [denseLayer]);
    }).toThrowError(/Shape mismatch.*\[2\] vs\. \[1\].*/);
  });

  it('Shape mismatch in a single weight', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    denseLayer.apply(inputTensor);
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer': [
          {
            'name': 'denseLayer1/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },

        ],
      }
    };
    expect(() => {
      loadWeightsFromJson(weightsJSON, [denseLayer]);
    })
        .toThrowError(
            /Layer.*denseLayer.*expects 2 weight.*but.*have 1 element.*/);
  });

  it('skipMismatch=true tolerates a single missing weight', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'denseLayer'});
    denseLayer.apply(inputTensor);
    const weightsJSON = {
      'keras_version': '2.1.2',
      'backend': 'tensorflow',
      'weights': {
        'denseLayer': [
          {
            'name': 'denseLayer1/kernel:0',
            'dtype': 'float32',
            'shape': [3, 2],
            'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
          },
        ],
      }
    };
    spyOn(console, 'warn');
    loadWeightsFromJson(weightsJSON, [denseLayer], true);
    expect(console.warn).toHaveBeenCalled();
    // Run a concrete input value through the layer to check that the only
    // weight available (i.e., kernel) is loaded properly. The missing weight
    // from the JSON object (i.e., bias) should not have been loaded and hence
    // should retain the initial value (all zeros), which ought to be reflected
    // in the output.
    expectTensorsClose(
        denseLayer.apply(tensor2d([[1, 1, 1]], [1, 3])) as Tensor,
        tensor2d([[0.9, 1.2]], [1, 2]));
  });
});

describeMathCPUAndGPU('loadWeightsFromNamedTensorMap', () => {
  const inputTensor =
      tfl.layers.input({shape: [3], name: 'inputLayer', dtype: 'float32'});

  it('One layer', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    namedWeightsMap[denseLayer.weights[1].originalName] = tensor1d([10, 20]);
    loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]);
    expectTensorsClose(
        denseLayer.weights[0].read(), tensor2d([1, 2, 3, 4, 5, 6], [3, 2]));
    expectTensorsClose(denseLayer.weights[1].read(), tensor1d([10, 20]));
  });

  it('Mismatching shape throws an error even in non-strict mode', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    namedWeightsMap[denseLayer.weights[1].originalName] = tensor1d([10, 20]);
    expect(
        () =>
            loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false))
        .toThrowError('Shape mismatch: [3,2] vs. [4,2]');
  });

  it('Extra weights leads to error', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    namedWeightsMap[denseLayer.weights[1].originalName] = tensor1d([10, 20]);
    namedWeightsMap['extra'] = tensor1d([10, 20]);
    expect(() => loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]))
        .toThrowError(/Provided weight data has no target variable: extra/);
  });

  it('Extra weights are allowed in non-strict mode', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    namedWeightsMap[denseLayer.weights[1].originalName] = tensor1d([10, 20]);
    namedWeightsMap['extra'] = tensor1d([10, 20]);
    loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false);
    expectTensorsClose(
        denseLayer.weights[0].read(), tensor2d([1, 2, 3, 4, 5, 6], [3, 2]));
    expectTensorsClose(denseLayer.weights[1].read(), tensor1d([10, 20]));
  });

  it('Unset weights leads to error', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    expect(() => loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]))
        .toThrowError(/1 of 2 weights are not set: .*bias.*/);
  });

  it('Unset weights are allowed in non-strict mode', () => {
    const denseLayer =
        tfl.layers.dense({units: 2, useBias: true, name: 'dense_layer'});
    denseLayer.apply(inputTensor);
    const namedWeightsMap: NamedTensorMap = {};
    namedWeightsMap[denseLayer.weights[0].originalName] =
        tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
    loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false);
    // No exception thrown.
  });
});
