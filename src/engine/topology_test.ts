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
import {eye, ones, scalar, Tensor, tensor1d, tensor2d, zeros} from '@tensorflow/tfjs-core';
import * as tfl from '../index';
import * as initializers from '../initializers';
import {DType, NamedTensorMap, Shape} from '../types';
import {describeMathCPU, describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';
import {LayerVariable, onesVariable, zerosVariable} from '../variables';

import {execute, FeedDict} from './executor';
import {Container, ContainerConfig, getSourceInputs, Input, InputLayer, InputSpec, Layer, LayerConfig, loadWeightsFromJson, loadWeightsFromNamedTensorMap, Node} from './topology';

// tslint:enable

class LayerForTest extends tfl.layers.Layer {
  static className = 'LayerForTest';
  constructor(config: LayerConfig) {
    super(config);
  }
}

class ContainerForTest extends Container {
  static className = 'ContainerForTest';
  constructor(config: ContainerConfig) {
    super(config);
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
  const inputTensors =
      [new tfl.SymbolicTensor(DType.float32, [1], null, [], {})];
  const outputTensors =
      [new tfl.SymbolicTensor(DType.float32, [2, 2], null, [], {})];
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
      expect(layer.dtype).toEqual(DType.float32);
      const config = layer.getConfig();
      expect(config.dtype).toEqual(layer.dtype);
    });

    it('initializes dtype to float32 if layerConfig.batchInputShape is set.',
       () => {
         const layer = new LayerForTest({batchInputShape: []});
         expect(layer.dtype).toEqual(DType.float32);
       });

    it('initializes initialWeights if present.', () => {
      const weights = [zeros([1])];
      const layer = new LayerForTest({weights});
      expect(layer.initialWeights).toEqual(weights);
    });

    it('Layer with duplicate weight names throws error', () => {
      class LayerForTest extends tfl.layers.Layer {
        static className = 'LayerForTest';
        constructor(config: LayerConfig) {
          super(config);
          this.addWeight(
              'foo', [1, 2], DType.float32,
              initializers.getInitializer('zeros'));
          this.addWeight(
              'foo', [2, 3], DType.float32,
              initializers.getInitializer('zeros'));
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
      () => new tfl.SymbolicTensor(DType.float32, [1], null, [], {}),
      () => [new tfl.SymbolicTensor(DType.float32, [1], null, [], {})]
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
           const inputSpecs = [new InputSpec({dtype: DType.float32})];
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
        new tfl.SymbolicTensor(DType.float32, [1], firstLayer, [], {});
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
  });

  describe('apply() passed >1 SymbolicTensor', () => {
    it('throws an exception for multiple symbolic inputs.', () => {
      const firstLayer = new LayerForTest({name: 'first layer'});
      const secondLayer = new LayerForTest({name: 'second layer'});
      const symbolicTensorList = [
        new tfl.SymbolicTensor(
            DType.float32, [1], firstLayer, [], {}, 'first_symbolic_tensor'),
        new tfl.SymbolicTensor(
            DType.float32, [1], firstLayer, [], {}, 'second_symbolic_tensor')
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
            DType.float32, [1], null, [], {}, 'first_symbolic_tensor'),
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
       const input = new tfl.SymbolicTensor(DType.float32, [1], null, [], {});
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
          new tfl.SymbolicTensor(DType.float32, [1], null, [], {});
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
         const input = new tfl.SymbolicTensor(DType.float32, [1], null, [], {});
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
             new tfl.SymbolicTensor(DType.float32, [1], null, [], {});
         const firstOutput = firstLayer.apply(initialInput);
         secondLayer.apply(firstOutput);

         expect(firstLayer.inboundNodes.length).toEqual(1);
         expect(firstLayer.outboundNodes.length).toEqual(1);
         expect(secondLayer.inboundNodes.length).toEqual(1);
         expect(secondLayer.outboundNodes.length).toEqual(0);
         expect(firstLayer.outboundNodes[0].outboundLayer).toEqual(secondLayer);
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
      input = new tfl.SymbolicTensor(
          DType.float32, [1], null, [], {}, 'firstInput');
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
      const secondInput = new tfl.SymbolicTensor(
          DType.float32, [1], null, [], {}, 'secondInput');
      layer.apply(secondInput);
      expect(() => layer.input).toThrowError(/"layer input" is ill-defined/);
    });

    it('output throws exception if there is more than one output', () => {
      const secondInput = new tfl.SymbolicTensor(
          DType.float32, [1], null, [], {}, 'secondInput');
      layer.apply(secondInput);
      expect(() => layer.output).toThrowError(/"layer output" is ill-defined/);
    });
  });

  describe('getInputAt and getOutputAt: ', () => {
    let input: tfl.SymbolicTensor;
    let layer: Layer;
    let output: tfl.SymbolicTensor;

    beforeEach(() => {
      input = new tfl.SymbolicTensor(
          DType.float32, [1], null, [], {}, 'firstInput');
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

describeMathCPU('InputLayer', () => {
  it('when initialized to its defaults throws an exception', () => {
    expect(() => tfl.layers.inputLayer({}))
        .toThrowError(/InputLayer should be passed either/);
  });
  describe('initialized with only an inputShape', () => {
    const inputShape = [1];
    const inputLayer = tfl.layers.inputLayer({inputShape}) as InputLayer;

    it('is not trainable.', () => {
      expect(inputLayer.trainable).toBe(false);
    });

    it('is built.', () => {
      expect(inputLayer.built).toBe(true);
    });

    it('is not sparse.', () => {
      expect(inputLayer.sparse).toBe(false);
    });

    it('automatically assigns a name.', () => {
      expect(inputLayer.name).toMatch(/^input.*$/);
    });

    it('creates a batchInputShape of [null].concat(inputShape).', () => {
      expect(inputLayer.batchInputShape).toEqual([null].concat(inputShape));
    });

    it('has no outboundNodes', () => {
      expect(inputLayer.outboundNodes.length).toEqual(0);
    });

    it('has one inboundNode', () => {
      expect(inputLayer.inboundNodes.length).toEqual(1);
    });

    describe('creates an inbound Node', () => {
      const inboundNode = inputLayer.inboundNodes[0];
      it('with no inboundLayers, nodeIndices, or tensorIndices', () => {
        expect(inboundNode.inboundLayers.length).toEqual(0);
        expect(inboundNode.nodeIndices.length).toEqual(0);
        expect(inboundNode.tensorIndices.length).toEqual(0);
      });

      it('with [null] inputMasks and outputMasks', () => {
        expect(inboundNode.inputMasks).toEqual([null]);
        expect(inboundNode.outputMasks).toEqual([null]);
      });

      it('with equal inputShapes and outputShapes', () => {
        expect(inboundNode.inputShapes).toEqual(inboundNode.outputShapes);
        expect(inboundNode.inputShapes).toEqual([[null].concat(inputShape)]);
      });

      describe('with a SymbolicTensor', () => {
        const symbolicTensor = inboundNode.inputTensors[0];

        it('that is defined.', () => {
          expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
        });

        it('assigned to both the input and outputTensors.', () => {
          expect(inboundNode.inputTensors.length).toEqual(1);
          expect(inboundNode.outputTensors.length).toEqual(1);
          expect(inboundNode.inputTensors).toEqual(inboundNode.outputTensors);
        });

        it('with a node and tensorIndex of 0.', () => {
          expect(symbolicTensor.nodeIndex).toEqual(0);
          expect(symbolicTensor.tensorIndex).toEqual(0);
        });

        it('with a sourceLayer of the inputLayer.', () => {
          expect(symbolicTensor.sourceLayer).toEqual(inputLayer);
        });

        it('with a name matching the inputLayer name.', () => {
          expect(symbolicTensor.name).toEqual(inputLayer.name);
        });

        it('with a dtype equal to the inputLayer.', () => {
          expect(symbolicTensor.dtype).toEqual(inputLayer.dtype);
        });

        it('with a shape matching the inputLayer.batchInputShape', () => {
          expect(symbolicTensor.shape).toEqual(inputLayer.batchInputShape);
        });
      });
    });
  });

  it('throws an exception if both inputShape and batchInputShape ' +
         'are specified during initialization.',
     () => {
       expect(
           () => tfl.layers.inputLayer({inputShape: [1], batchInputShape: [1]}))
           .toThrowError(/Only provide the inputShape OR batchInputShape/);
     });

  for (const batchSize of [null, 5]) {
    it('initializes with batchSize when inputShape specified', () => {
      const inputShape = [1];
      const inputLayer = tfl.layers.inputLayer({inputShape, batchSize});
      expect(inputLayer.batchInputShape).toEqual([
        batchSize
      ].concat(inputShape));
    });
  }

  it('initializes with batchInputShape if specified.', () => {
    const batchInputShape = [1, 2];
    const inputLayer = tfl.layers.inputLayer({batchInputShape});
    expect(inputLayer.batchInputShape).toEqual(batchInputShape);
  });

  it('initializes with batchInputShape if null specified for the batch size.',
     () => {
       const batchInputShape = [1, 2];
       const inputLayer = tfl.layers.inputLayer({batchInputShape});
       expect(inputLayer.batchInputShape).toEqual(batchInputShape);
     });

  it('throws exception if batchSize and batchInputShape are specified.', () => {
    expect(() => tfl.layers.inputLayer({batchInputShape: [1], batchSize: 5}))
        .toThrowError(/Cannot specify batchSize if batchInputShape/);
  });

  for (const sparse of [true, false]) {
    it('uses config.sparse during initialization.', () => {
      const inputLayer =
          tfl.layers.inputLayer({inputShape: [1], sparse}) as InputLayer;
      expect(inputLayer.sparse).toEqual(sparse);
    });
  }

  it('use config.dtype during initialization.', () => {
    const dtype = DType.float32;
    const inputLayer = tfl.layers.inputLayer({inputShape: [1], dtype});
    expect(inputLayer.dtype).toEqual(dtype);
  });

  it('use config.name during initialization.', () => {
    const name = 'abc';
    const inputLayer = tfl.layers.inputLayer({inputShape: [1], name});
    expect(inputLayer.name).toEqual(name);
  });

  it('throws an exception if apply() is called with any input.', () => {
    const inputLayer = tfl.layers.inputLayer({inputShape: [1]});
    const symbolicTensor =
        new tfl.SymbolicTensor(DType.float32, [2], null, [], {});
    expect(() => inputLayer.apply(symbolicTensor))
        .toThrowError(/Cannot pass any input to an InputLayer's apply/);
  });

  it('throws an exception if its inputs differ in shape to what it ' +
         'was initialized to.',
     () => {
       const inputLayer = tfl.layers.inputLayer({inputShape: [1]});
       const inputs = ones([2, 2]);
       expect(() => inputLayer.apply(inputs)).toThrowError();
     });

  it('returns a serializable config.', () => {
    const batchInputShape = [1];
    const dtype = DType.float32;
    const sparse = true;
    const name = 'my_name';
    const inputLayer =
        tfl.layers.inputLayer({batchInputShape, dtype, sparse, name});
    expect(inputLayer.getConfig())
        .toEqual({batchInputShape, dtype, sparse, name});
  });
});

describe('Input()', () => {
  it('throws an exception if neither shape nor batchShape are specified',
     () => {
       expect(() => tfl.layers.input({}))
           .toThrowError(/Please provide to Input either/);
     });

  const shape = [1];
  const batchShape = [2, 2];
  const name = 'abc';
  const dtype = DType.float32;

  it('returns an initialized symbolicTensor given a shape.', () => {
    const symbolicTensor = tfl.layers.input({shape, name, dtype});
    expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
    expect(symbolicTensor.shape).toEqual([null].concat(shape));
    expect(symbolicTensor.name).toMatch(/abc/);
    expect(symbolicTensor.dtype).toEqual(dtype);
  });

  it('returns a SymbolicTensor given a batchShape', () => {
    const symbolicTensor = tfl.layers.input({batchShape});
    expect(symbolicTensor.shape).toEqual(batchShape);
  });

  it('throws exception if both shape and batchShape are specified.', () => {
    expect(() => tfl.layers.input({shape, batchShape}))
        .toThrowError(/Please provide either a `shape`/);
  });

  it('produces output that can feed into a Layer.', () => {
    const inputTensor = Input({shape, name});
    const otherLayer = new LayerForTest({name: 'firstLayer'});
    const output = otherLayer.apply(inputTensor) as tfl.SymbolicTensor;
    expect(output instanceof tfl.SymbolicTensor).toBe(true);
    expect(output.name).toEqual('firstLayer/firstLayer');
  });
});

describeMathCPUAndGPU('Container.fromConfig', () => {
  it('creates a minimal Container from simplest config', () => {
    // tslint:disable:no-any
    const config = {
      name: 'test',
      layers: [] as any[],
      inputLayers: [] as any[],
      outputLayers: [] as any[]
    };
    // tslint:enable
    const container =
        Container.fromConfig(ContainerForTest, config) as Container;
    expect(container.name).toEqual('test');
  });

  it('creates a simple network', () => {
    /* python generating code
    a=Input(shape=(32,))
    b=Dense(32)(a)
    model = Container(inputs=a, outputs=b, name="test")
    pprint.pprint(model.get_config())
    */
    const config = {
      inputLayers: [['input_2', 0, 0]],
      layers: [
        {
          className: 'InputLayer',
          config: {
            batchInputShape: [null, 32],
            dtype: 'float32',
            name: 'input_2',
            sparse: false
          },
          inboundNodes: [] as string[][],
          name: 'input_2'
        },
        {
          className: 'Dense',
          config: {
            activation: 'linear',
            activityRegularizer: null as string,
            biasConstraint: null as string,
            biasInitializer: {className: 'Zeros', config: {}},
            biasRegularizer: null as string,
            kernelConstraint: null as string,
            kernelInitializer: {
              className: 'VarianceScaling',
              config: {
                distribution: 'uniform',
                mode: 'fanAvg',
                scale: 1.0,
                seed: null as number
              }
            },
            kernelRegularizer: null as string,
            name: 'dense_2',
            trainable: null as boolean,
            units: 32,
            use_bias: true
          },
          inboundNodes: [[['input_2', 0, 0, {}]]],
          name: 'dense_2'
        }
      ],
      name: 'test',
      outputLayers: [['dense_2', 0, 0]]
    };
    const container =
        Container.fromConfig(ContainerForTest, config) as Container;
    expect(container.name).toEqual('test');
    const allZeros = zeros([1, 32]);
    expectTensorsClose(container.apply(allZeros) as Tensor, allZeros);
  });
});

describeMathCPUAndGPU('Container', () => {
  const inputLayerName = 'inputLayerName';
  const layerName = 'layerName';
  const containerName = 'simpleContainer';
  let inputTensor: tfl.SymbolicTensor;
  let layer: Layer;
  let output: tfl.SymbolicTensor;
  let simpleContainer: Container;

  beforeEach(() => {
    inputTensor =
        Input({shape: [1], name: inputLayerName, dtype: DType.float32});
    layer = new LayerForTest({name: layerName});
    output = layer.apply(inputTensor) as tfl.SymbolicTensor;
    simpleContainer = new ContainerForTest(
        {inputs: [inputTensor], outputs: [output], name: containerName});
  });

  it('initializes with no inputs or outputs and a default name', () => {
    const container = new ContainerForTest({inputs: [], outputs: []});
    expect(container.name).toMatch(/^container.+$/);
  });

  it('initializes with no inputs or outputs and a given name', () => {
    const name = 'xyz';
    const container = new ContainerForTest({inputs: [], outputs: [], name});
    expect(container.name).toMatch(name);
  });

  it('throws an exception if same input provided twice', () => {
    const makeContainer = () => {
      // tslint:disable-next-line:no-unused-expression
      new ContainerForTest({inputs: [inputTensor, inputTensor], outputs: []});
    };
    expect(makeContainer).toThrowError(/inputs.*redundant/);
  });

  it('throws an exception if graph is disconnected', () => {
    const makeContainer = () => {
      // tslint:disable-next-line:no-unused-expression
      new ContainerForTest({inputs: [], outputs: [output]});
    };
    expect(makeContainer).toThrowError(/disconnected/);
  });

  it('creates inputLayers', () => {
    expect(simpleContainer.inputLayers).toEqual([inputTensor.sourceLayer]);
  });

  it('creates outputLayers', () => {
    expect(simpleContainer.outputLayers).toEqual([layer]);
  });

  it('creates inputNames', () => {
    expect(simpleContainer.inputNames).toEqual([inputLayerName]);
  });

  it('creates outputNames', () => {
    expect(simpleContainer.outputNames).toEqual([layerName]);
  });

  it('throws exception if given a non-input layer as input', () => {
    const makeContainer = () => {
      // tslint:disable-next-line:no-unused-expression
      new ContainerForTest({inputs: [output], outputs: []});
    };
    expect(makeContainer).toThrowError(/must be InputLayer objects/);
  });

  it('creates layers for simplest case', () => {
    expect(simpleContainer.layers).toEqual([inputTensor.sourceLayer, layer]);
  });

  it('creates layers when multiple layers specified', () => {
    const layer1 = new LayerForTest({name: 'layer1'});
    const layer2 = new LayerForTest({name: 'layer2'});
    const output =
        layer2.apply(layer1.apply(inputTensor)) as tfl.SymbolicTensor;
    const container =
        new ContainerForTest({inputs: [inputTensor], outputs: [output]});
    expect(container.layers).toEqual([inputTensor.sourceLayer, layer1, layer2]);
  });

  it('correctly creates model with shared subgraphs.', () => {
    /*
      The graph:

        A
      /  \
      B  X
      |  |
      C  B
         |
         C
    */
    const layerA = new LayerForTest({name: 'A'});
    const layerB = new LayerForTest({name: 'B'});
    const layerC = new LayerForTest({name: 'C'});
    const layerX = new LayerForTest({name: 'X'});
    const aOutput = layerA.apply(inputTensor);
    const output1 = layerC.apply(layerB.apply(aOutput)) as tfl.SymbolicTensor;
    const output2 =
        layerC.apply(layerB.apply(layerX.apply(aOutput))) as tfl.SymbolicTensor;

    const container = new ContainerForTest(
        {inputs: [inputTensor], outputs: [output1, output2]});

    const compareFunction = (a: Layer, b: Layer) => {
      if (a.name < b.name) {
        return -1;
      } else if (a.name > b.name) {
        return 1;
      } else {
        return 0;
      }
    };
    const sortedLayers = container.layers.slice().sort(compareFunction);
    const expectedSortedLayers = [
      inputTensor.sourceLayer, layerA, layerB, layerC, layerX
    ].sort(compareFunction);

    expect(sortedLayers).toEqual(expectedSortedLayers);
  });

  it('throws exception if multiple layers have the same name', () => {
    const name = 'abc';
    const layer1 = new LayerForTest({name});
    const layer2 = new LayerForTest({name});
    const output =
        layer2.apply(layer1.apply(inputTensor)) as tfl.SymbolicTensor;
    const makeContainer = () => {
      // tslint:disable-next-line:no-unused-expression
      new ContainerForTest({inputs: [inputTensor], outputs: [output]});
    };
    expect(makeContainer).toThrowError(/layer names should be unique/);
  });

  it('weights gets all weights.', () => {
    const inputShape = [1, 6];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.dense({units: 2, useBias: false});
    const layer2 = tfl.layers.dense({units: 1, useBias: true});
    const output = layer2.apply(layer1.apply(inputLayer)) as tfl.SymbolicTensor;

    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});
    expect(container.weights.length).toEqual(3);
    expect(container.weights[0].name).toEqual(layer1.weights[0].name);
    expect(container.weights[1].name).toEqual(layer2.weights[0].name);
    expect(container.weights[2].name).toEqual(layer2.weights[1].name);
  });

  it('trainableWeights and nonTrainableWeights.', () => {
    const inputShape = [1, 6];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.dense({units: 2, useBias: false});
    const layer2 = tfl.layers.dense({units: 1, useBias: true});
    const output = layer2.apply(layer1.apply(inputLayer)) as tfl.SymbolicTensor;

    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});
    expect(container.trainableWeights.length).toEqual(3);
    expect(container.trainableWeights[0].name).toEqual(layer1.weights[0].name);
    expect(container.trainableWeights[1].name).toEqual(layer2.weights[0].name);
    expect(container.trainableWeights[2].name).toEqual(layer2.weights[1].name);
    expect(container.nonTrainableWeights.length).toEqual(0);
  });

  it('call() executes all layers.', () => {
    const inputShape = [1, 6];
    const finalShape = [3, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.reshape({name: 'layer1', targetShape: [2, 3]});
    const layer2 =
        tfl.layers.reshape({name: 'layer2', targetShape: finalShape});
    const output = layer2.apply(layer1.apply(inputLayer)) as tfl.SymbolicTensor;

    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});
    const result = container.call(ones([1, 1, 6]), {}) as Tensor[];
    const resultShape = [1].concat(finalShape);
    expectTensorsClose(result[0], ones(resultShape));
  });

  it('apply() executes all layers with concrete tensors.', () => {
    const inputShape = [1, 6];
    const finalShape = [3, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.reshape({name: 'layer1', targetShape: [2, 3]});
    const layer2 =
        tfl.layers.reshape({name: 'layer2', targetShape: finalShape});
    const output = layer2.apply(layer1.apply(inputLayer)) as tfl.SymbolicTensor;

    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});
    const result = container.apply(ones([1, 1, 6])) as Tensor;
    const resultShape = [1].concat(finalShape);
    expectTensorsClose(result, ones(resultShape));
  });

  it('apply() executes all layers with symbolic tensors.', () => {
    const inputShape = [1, 6];
    const finalShape = [3, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.reshape({name: 'layer1', targetShape: [2, 3]});
    const layer2 =
        tfl.layers.reshape({name: 'layer2', targetShape: finalShape});
    const output = layer2.apply(layer1.apply(inputLayer)) as tfl.SymbolicTensor;

    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});

    const newInput = tfl.layers.input({shape: [1, 6]});
    const symbolicResult = container.apply(newInput);
    expect(symbolicResult instanceof tfl.SymbolicTensor).toEqual(true);
    const concreteResult = execute(
        symbolicResult as tfl.SymbolicTensor,
        new FeedDict([{key: newInput, value: ones([1, 1, 6])}]));
    const resultShape = [1].concat(finalShape);
    expectTensorsClose(concreteResult as Tensor, ones(resultShape));
  });

  it('computeOutputShape() computes the correct outputShape', () => {
    const inputShape = [2, 3];
    const finalShape = [3, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer = tfl.layers.reshape({targetShape: finalShape});
    const output = layer.apply(inputLayer) as tfl.SymbolicTensor;
    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [output]});
    expect(container.computeOutputShape([1].concat(inputShape))).toEqual([
      1
    ].concat(finalShape));
  });

  it('trainableWeights is initially an empty Array', () => {
    expect(simpleContainer.trainableWeights).toEqual([]);
  });

  it('trainableWeights tracks only trainable weights', () => {
    const inputShape = [2, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.reshape({targetShape: [4], name: 'reshapeLayer'});
    const layer1Output = layer1.apply(inputLayer) as tfl.SymbolicTensor;
    const layer2 =
        tfl.layers.dense({units: 2, useBias: false, name: 'denseLayer'});
    const layer2Output = layer2.apply(layer1Output) as tfl.SymbolicTensor;
    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [layer2Output]});
    expect(container.trainableWeights.length).toEqual(1);
  });

  it('stateful is initially false', () => {
    expect(simpleContainer.stateful).toEqual(false);
  });

  function createSimpleTwoLayerContainer(): [Container, Layer[]] {
    const inputShape = [2, 2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const layer1 = tfl.layers.reshape({targetShape: [4], name: 'reshapeLayer'});
    const layer1Output = layer1.apply(inputLayer) as tfl.SymbolicTensor;
    const layer2 =
        tfl.layers.dense({units: 2, useBias: false, name: 'denseLayer'});
    const layer2Output = layer2.apply(layer1Output) as tfl.SymbolicTensor;
    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [layer2Output]});
    return [container, [container.inputLayers[0], layer1, layer2]];
  }

  it('getLayer works by name', () => {
    const [container, layers] = createSimpleTwoLayerContainer();
    expect(container.getLayer(layers[0].name)).toEqual(layers[0]);
    expect(container.getLayer(layers[1].name)).toEqual(layers[1]);
    expect(container.getLayer(layers[2].name)).toEqual(layers[2]);
  });

  it('getLayer works by index', () => {
    const [container, layers] = createSimpleTwoLayerContainer();
    expect(container.getLayer(null, 0)).toEqual(layers[0]);
    expect(container.getLayer(null, 1)).toEqual(layers[1]);
    expect(container.getLayer(null, 2)).toEqual(layers[2]);
  });

  it('getLayer throws error for nonexistent layer name', () => {
    const [container, layers] = createSimpleTwoLayerContainer();
    expect(
        () => container.getLayer(
            layers[0].name + '_suffixToMakeLayerNameNonexistent'))
        .toThrowError(/No such layer/);
  });

  it('getLayer throws error for index out of bound', () => {
    const container = createSimpleTwoLayerContainer()[0];
    expect(() => container.getLayer(null, 3)).toThrowError(/only has 3 layer/);
  });

  it('getLayer throws error when neither name or index is specified', () => {
    const container = createSimpleTwoLayerContainer()[0];
    expect(() => container.getLayer())
        .toThrowError(/Provide either a layer name or layer index/);
  });
});

describeMathCPUAndGPU('Container.calculateLosses', () => {
  function createSimpleOneLayerContainer(useRegularizers: boolean):
      [Container, Layer[]] {
    const inputShape = [2];
    const inputLayer = tfl.layers.input({shape: inputShape});
    const kernelRegularizer =
        useRegularizers ? tfl.regularizers.l1({l1: 2}) : null;
    const biasRegularizer =
        useRegularizers ? tfl.regularizers.l2({l2: 3}) : null;
    const denseLayer = tfl.layers.dense({
      units: 2,
      kernelInitializer: 'ones',
      biasInitializer: 'ones',
      kernelRegularizer,
      biasRegularizer,
      name: 'denseLayer'
    });
    const layer2Output = denseLayer.apply(inputLayer) as tfl.SymbolicTensor;
    const container =
        new ContainerForTest({inputs: [inputLayer], outputs: [layer2Output]});
    return [container, [denseLayer]];
  }

  it('L1 and L2', () => {
    const container = createSimpleOneLayerContainer(true)[0];
    const losses = container.calculateLosses();
    expect(losses.length).toEqual(2);
    expectTensorsClose(losses[0], scalar(2 * (1 + 1 + 1 + 1)));
    expectTensorsClose(losses[1], scalar(3 * (1 + 1)));
  });

  it('No regularizers', () => {
    const container = createSimpleOneLayerContainer(false)[0];
    const losses = container.calculateLosses();
    expect(losses.length).toEqual(0);
  });
});

describe('getSourceInputs()', () => {
  it('returns the single source input', () => {
    const inputTensor = tfl.layers.input({shape: [1]});
    const layer1 = new LayerForTest({name: 'layer1'});
    const layer2 = new LayerForTest({name: 'layer2'});
    const output =
        layer2.apply(layer1.apply(inputTensor)) as tfl.SymbolicTensor;
    expect(getSourceInputs(output)).toEqual([inputTensor]);
  });

  it('returns all inputs', () => {
    const input1 = tfl.layers.input({shape: [1], name: 'input1'});
    const input2 = tfl.layers.input({shape: [1], name: 'input2'});
    const layer = new LayerForTest({});
    const output1 = layer.apply(input1) as tfl.SymbolicTensor;
    const output2 = layer.apply(input2) as tfl.SymbolicTensor;
    expect(getSourceInputs(output1)).toEqual([input1]);
    expect(getSourceInputs(output2)).toEqual([input2]);
  });
});

// TODO(cais): Maybe remove this test once loadWeightsFromJson is removed
//   (b/74015805).
describeMathCPUAndGPU('loadWeightsFromJson', () => {
  const inputTensor =
      tfl.layers.input({shape: [3], name: 'inputLayer', dtype: DType.float32});

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
      tfl.layers.input({shape: [3], name: 'inputLayer', dtype: DType.float32});

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
});
