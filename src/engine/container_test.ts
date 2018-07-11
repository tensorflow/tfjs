/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ones, scalar, Tensor, zeros} from '@tensorflow/tfjs-core';

import * as tfl from '../index';
import {describeMathCPUAndGPU, expectTensorsClose} from '../utils/test_utils';

import {Container, ContainerConfig} from './container';
import {execute, FeedDict} from './executor';
import {getSourceInputs, Layer, LayerConfig} from './topology';

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
        tfl.input({shape: [1], name: inputLayerName, dtype: 'float32'});
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
