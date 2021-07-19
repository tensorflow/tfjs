/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ones} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPU} from '../utils/test_utils';

import {Input} from './input_layer';
import {LayerArgs} from './topology';

describeMathCPU('InputLayer', () => {
  it('when initialized to its defaults throws an exception', () => {
    expect(() => tfl.layers.inputLayer({}))
        .toThrowError(/InputLayer should be passed either/);
  });
  describe('initialized with only an inputShape', () => {
    const inputShape = [1];
    const inputLayer = tfl.layers.inputLayer({inputShape});

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

  // See https://github.com/tensorflow/tfjs/issues/1341
  it('allow `null` in shape', () => {
    const inputShape = [null, 2];
    const inputs = tfl.layers.inputLayer({inputShape});
    expect(inputs.inputSpec[0].shape).toEqual([null].concat(inputShape));
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
          tfl.layers.inputLayer({inputShape: [1], sparse});
      expect(inputLayer.sparse).toEqual(sparse);
    });
  }

  it('use config.dtype during initialization.', () => {
    const dtype = 'float32';
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
    const symbolicTensor = new tfl.SymbolicTensor('float32', [2], null, [], {});
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
    const dtype = 'float32';
    const sparse = true;
    const name = 'my_name';
    const inputLayer =
        tfl.layers.inputLayer({batchInputShape, dtype, sparse, name});
    expect(inputLayer.getConfig())
        .toEqual({batchInputShape, dtype, sparse, name});
  });
});

class LayerForTest extends tfl.layers.Layer {
  static className = 'LayerForTest';
  constructor(args: LayerArgs) {
    super(args);
  }
}

describe('Input()', () => {
  it('throws an exception if neither shape nor batchShape are specified',
     () => {
       expect(() => tfl.layers.input({}))
           .toThrowError(/Please provide to Input either/);
     });

  const shape = [1];
  const batchShape = [2, 2];
  const name = 'abc';
  const dtype = 'float32';

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
