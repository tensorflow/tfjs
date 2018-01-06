/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as conv_util from '../math/conv_util';
import {NDArray} from '../math/ndarray';

import {ConstantNode, Graph, Node, Tensor, VariableNode} from './graph';
import {FeedDictionary} from './session';
import * as session_util from './session_util';

class TestNode extends Node {
  validate() {}
}

describe('Graph', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('nodes have ascending ids', () => {
    const a = new TestNode(g, '', {}, new Tensor([]));
    const b = new TestNode(g, '', {}, new Tensor([]));
    expect(b.id).toEqual(a.id + 1);
  });

  it('variable creates a node in the graph', () => {
    const v = g.variable('', NDArray.zeros([1]));
    expect(v.node.graph).toEqual(g);
  });

  it('variable creates a VariableNode in the graph', () => {
    const v = g.variable('', NDArray.zeros([1]));
    expect(v.node instanceof VariableNode).toEqual(true);
  });

  it('variable passes name to graph node', () => {
    const v = g.variable('hello', NDArray.zeros([1]));
    expect(v.node.name).toEqual('hello');
  });

  it('mnist fully-connected', () => {
    const input = g.placeholder('input', [28 * 28]);
    const fc0W = g.variable('fc0W', NDArray.zeros([32, 28 * 28]));
    const fc0B = g.variable('fc0B', NDArray.zeros([32]));
    const fc0 = g.add(g.matmul(fc0W, input), fc0B);
    const relu0 = g.relu(fc0);
    const fc1W = g.variable('fc1W', NDArray.zeros([32, 32]));
    const fc1B = g.variable('fc1B', NDArray.zeros([32]));
    const fc1 = g.add(g.matmul(fc1W, relu0), fc1B);
    const relu1 = g.relu(fc1);
    const fc2W = g.variable('fc2W', NDArray.zeros([32, 32]));
    const fc2B = g.variable('fc2B', NDArray.zeros([32]));
    const fc2 = g.add(g.matmul(fc2W, relu1), fc2B);
    const relu2 = g.relu(fc2);
    const fc3W = g.variable('fc3W', NDArray.zeros([10, 32]));
    const fc3B = g.variable('fc3B', NDArray.zeros([10]));
    const fc3 = g.add(g.matmul(fc3W, relu2), fc3B);

    const fd = new FeedDictionary([{tensor: input, data: NDArray.zeros([1])}]);
    const orderedEvaluationSet =
        session_util.getOrderedEvaluationSetFromEvalTensor([fc3], fd);
    expect(orderedEvaluationSet.length).toBeGreaterThan(1);
  });
});

describe('Variable validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('null data throws', () => {
    expect(() => g.variable('test', null)).toThrowError();
  });

  it('non null data does not throw', () => {
    g.variable('test', NDArray.zeros([5]));
  });
});

describe('Placeholder validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('does not throw', () => {
    expect(g.placeholder('test', [1, 2, 3]).shape).toEqual([1, 2, 3]);
  });
});

describe('Constant', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('null data throws', () => {
    expect(() => g.constant(null)).toThrowError();
  });

  it('non null data does not throw', () => {
    expect(g.constant(NDArray.zeros([5])).shape).toEqual([5]);
  });

  it('from a single value', () => {
    const c = g.constant(3);
    expect(c.shape).toEqual([]);
    const values = (c.node as ConstantNode).data.dataSync();
    expect(values).toEqual(new Float32Array([3]));
  });

  it('from 1d array', () => {
    const c = g.constant([1, 2, 3]);
    expect(c.shape).toEqual([3]);
    const values = (c.node as ConstantNode).data.dataSync();
    expect(values).toEqual(new Float32Array([1, 2, 3]));
  });

  it('from 2d array', () => {
    const c = g.constant([[1, 2, 3], [4, 5, 6]]);
    expect(c.shape).toEqual([2, 3]);
    const values = (c.node as ConstantNode).data.dataSync();
    expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('from 3d array', () => {
    const c = g.constant([[[1], [2], [3]], [[4], [5], [6]]]);
    expect(c.shape).toEqual([2, 3, 1]);
    const values = (c.node as ConstantNode).data.dataSync();
    expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it('from 4d array', () => {
    const c = g.constant([[[[1]], [[2]], [[3]]], [[[4]], [[5]], [[6]]]]);
    expect(c.shape).toEqual([2, 3, 1, 1]);
    const values = (c.node as ConstantNode).data.dataSync();
    expect(values).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });
});

describe('Reshape validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different sizes throws', () => {
    expect(() => g.reshape(new Tensor([5, 4]), [3, 3])).toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.reshape(new Tensor([5, 4]), [20]).shape).toEqual([20]);
  });
});

describe('FusedLinearCombination validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shape tensors throws', () => {
    expect(
        () => g.fusedLinearCombination(
            new Tensor([3, 4]), new Tensor([1]), new Tensor([]),
            new Tensor([])))
        .toThrowError();
  });

  it('Non scalar c1 throws', () => {
    expect(
        () => g.fusedLinearCombination(
            new Tensor([3, 4]), new Tensor([3, 4]), new Tensor([1, 2]),
            new Tensor([])))
        .toThrowError();
  });

  it('Non scalar c2 throws', () => {
    expect(
        () => g.fusedLinearCombination(
            new Tensor([3, 4]), new Tensor([3, 4]), new Tensor([]),
            new Tensor([1, 2])))
        .toThrowError();
  });

  it('does not throw when shapes correct', () => {
    expect(g.fusedLinearCombination(
                new Tensor([3, 4]), new Tensor([3, 4]), new Tensor([]),
                new Tensor([]))
               .shape)
        .toEqual([3, 4]);
  });
});

describe('Add validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shapes throws', () => {
    expect(() => g.add(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.add(new Tensor([5, 4]), new Tensor([5, 4])).shape).toEqual([5, 4]);
  });

  it('1D broadcasted to 2D does not throw', () => {
    expect(g.add(new Tensor([5, 3]), new Tensor([3])).shape).toEqual([5, 3]);
  });

  it('Another 1D broadcasted to 2D does not throw', () => {
    expect(g.add(new Tensor([3]), new Tensor([7, 3])).shape).toEqual([7, 3]);
  });

  it('Non-matching broadcast throws', () => {
    expect(() => g.add(new Tensor([5, 3]), new Tensor([5]))).toThrowError();
  });
});

describe('Subtract validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shapes throws', () => {
    expect(() => g.subtract(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.subtract(new Tensor([5, 4]), new Tensor([5, 4])).shape).toEqual([
      5, 4
    ]);
  });
});

describe('Multiply validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shapes throws', () => {
    expect(() => g.multiply(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.multiply(new Tensor([5, 4]), new Tensor([5, 4])).shape).toEqual([
      5, 4
    ]);
  });
});

describe('Divide validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shapes throws', () => {
    expect(() => g.divide(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.divide(new Tensor([5, 4]), new Tensor([5, 4])).shape).toEqual([
      5, 4
    ]);
  });
});

describe('Reduce sum validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('does not throw', () => {
    expect(g.reduceSum(new Tensor([5, 4, 4, 9])).shape).toEqual([]);
  });
});

describe('Concat1d validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Non 1-rank tensor x1 throws', () => {
    expect(() => g.concat1d(new Tensor([5, 4]), new Tensor([1])))
        .toThrowError();
  });

  it('Non 1-rank tensor x2 throws', () => {
    expect(() => g.concat1d(new Tensor([5]), new Tensor([1, 2])).shape)
        .toThrowError();
  });

  it('Axis=0 shapes the same does not throw', () => {
    expect(g.concat1d(new Tensor([5]), new Tensor([1])).shape)
        .toEqual([6]);
  });
});

describe('Concat2d validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Non 2-rank tensor x1 throws', () => {
    expect(() => g.concat2d(new Tensor([5]), new Tensor([1, 2]), 0))
        .toThrowError();
  });

  it('Non 2-rank tensor x2 throws', () => {
    expect(() => g.concat2d(new Tensor([5, 4]), new Tensor([1]), 0))
        .toThrowError();
  });

  it('Axis=0 different shapes throw', () => {
    expect(() => g.concat2d(new Tensor([2, 3]), new Tensor([4, 4]), 0))
        .toThrowError();
  });

  it('Axis=0 shapes the same doe not throw', () => {
    expect(g.concat2d(new Tensor([2, 3]), new Tensor([4, 3]), 0).shape)
        .toEqual([6, 3]);
  });

  it('Axis=1 different shapes throw', () => {
    expect(() => g.concat2d(new Tensor([2, 3]), new Tensor([4, 4]), 1))
        .toThrowError();
  });

  it('Axis=1 shapes the same doe not throw', () => {
    expect(g.concat2d(new Tensor([2, 4]), new Tensor([2, 3]), 1).shape)
        .toEqual([2, 7]);
  });
});

describe('Concat3d validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Non 3-rank tensor x1 throws', () => {
    expect(() => g.concat3d(new Tensor([5, 4]), new Tensor([1, 2, 3]), 0))
        .toThrowError();
  });

  it('Non 3-rank tensor x2 throws', () => {
    expect(() => g.concat3d(new Tensor([5, 4, 1]), new Tensor([1, 2]), 0))
        .toThrowError();
  });

  it('Axis=0 different shapes throws', () => {
    expect(() => g.concat3d(new Tensor([5, 4, 1]), new Tensor([1, 2, 1]), 0))
        .toThrowError();
  });

  it('Axis=1 different shapes throws', () => {
    expect(() => g.concat3d(new Tensor([5, 4, 1]), new Tensor([1, 2, 1]), 1))
        .toThrowError();
  });

  it('Axis=2 different shapes throws', () => {
    expect(() => g.concat3d(new Tensor([5, 4, 1]), new Tensor([1, 2, 1]), 2))
        .toThrowError();
  });

  it('Axis=0 shapes the same does not throw', () => {
    expect(g.concat3d(new Tensor([5, 4, 3]), new Tensor([1, 4, 3]), 0).shape)
        .toEqual([6, 4, 3]);
  });

  it('Axis=1 shapes the same does not throw', () => {
    expect(g.concat3d(new Tensor([5, 3, 3]), new Tensor([5, 4, 3]), 1).shape)
        .toEqual([5, 7, 3]);
  });

  it('Axis=2 shapes the same does not throw', () => {
    expect(g.concat3d(new Tensor([5, 4, 3]), new Tensor([5, 4, 1]), 2).shape)
        .toEqual([5, 4, 4]);
  });
});

describe('Concat4d validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Non 4-rank tensor x1 throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4]), new Tensor([1, 2, 3, 4]), 0))
        .toThrowError();
  });

  it('Non 4-rank tensor x2 throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4, 1]), new Tensor([1, 2, 3, 4]), 0))
        .toThrowError();
  });

  it('Axis=0 different shapes throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4, 1, 1]),
      new Tensor([1, 2, 1, 1]), 0)).toThrowError();
  });

  it('Axis=1 different shapes throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4, 1, 1]),
      new Tensor([1, 2, 1, 1]), 1)).toThrowError();
  });

  it('Axis=2 different shapes throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4, 1, 1]),
      new Tensor([1, 2, 1, 1]), 2)).toThrowError();
  });

  it('Axis=3 different shapes throws', () => {
    expect(() => g.concat4d(new Tensor([5, 4, 1, 1]),
      new Tensor([1, 2, 1, 1]), 3)).toThrowError();
  });

  it('Axis=0 shapes the same does not throw', () => {
    expect(g.concat4d(new Tensor([5, 4, 3, 1]),
      new Tensor([1, 4, 3, 1]), 0).shape).toEqual([6, 4, 3, 1]);
  });

  it('Axis=1 shapes the same does not throw', () => {
    expect(g.concat4d(new Tensor([5, 3, 3, 1]),
      new Tensor([5, 4, 3, 1]), 1).shape).toEqual([5, 7, 3, 1]);
  });

  it('Axis=2 shapes the same does not throw', () => {
    expect(g.concat4d(new Tensor([5, 4, 3, 1]),
      new Tensor([5, 4, 1, 1]), 2).shape).toEqual([5, 4, 4, 1]);
  });

  it('Axis=3 shapes the same does not throw', () => {
    expect(g.concat4d(new Tensor([5, 4, 3, 1]), 
      new Tensor([5, 4, 3, 2]), 3).shape).toEqual([5, 4, 3, 3]);
  });
});

describe('matmul validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Wrong rank x1 throws', () => {
    expect(() => g.matmul(new Tensor([5, 4, 3]), new Tensor([1, 2])))
        .toThrowError();
  });

  it('Wrong rank x2 throws', () => {
    expect(() => g.matmul(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Inner dimensions of matrix multiply do not match throws', () => {
    expect(() => g.matmul(new Tensor([5, 4]), new Tensor([5, 5])))
        .toThrowError();
  });

  it('Inner dimensions of matrix times vector does not match throws', () => {
    expect(() => g.matmul(new Tensor([5, 4]), new Tensor([5]))).toThrowError();
  });

  it('Inner dimensions of vector times matrix does not match throws', () => {
    expect(() => g.matmul(new Tensor([5]), new Tensor([4, 5]))).toThrowError();
  });

  it('Vector times vector shapes dont match throws', () => {
    expect(() => g.matmul(new Tensor([5]), new Tensor([4]))).toThrowError();
  });

  it('Matrix times matrix inner dimensions match does not throw', () => {
    expect(g.matmul(new Tensor([5, 4]), new Tensor([4, 6])).shape).toEqual([
      5, 6
    ]);
  });

  it('Vector times matrix inner dimensions match does not throw', () => {
    expect(g.matmul(new Tensor([4]), new Tensor([4, 6])).shape).toEqual([6]);
  });

  it('Matrix times vector inner dimensions match does not throw', () => {
    expect(g.matmul(new Tensor([4, 6]), new Tensor([6])).shape).toEqual([4]);
  });
});

describe('conv2d validation', () => {
  let g: Graph;
  let fieldSize: number;
  let outputDepth: number;
  let stride: number;
  let zeroPad: number;

  beforeEach(() => {
    g = new Graph();
    fieldSize = 4;
    outputDepth = 10;
    stride = 1;
    zeroPad = 1;
  });

  it('Wrong rank x throws', () => {
    expect(
        () => g.conv2d(
            new Tensor([5, 4]), new Tensor([1, 2, 3, 4]),
            new Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad))
        .toThrowError();
  });

  it('Wrong rank weights throws', () => {
    expect(
        () => g.conv2d(
            new Tensor([5, 4, 3]), new Tensor([1, 2, 3]),
            new Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad))
        .toThrowError();
  });

  it('Wrong rank biases throws', () => {
    expect(
        () => g.conv2d(
            new Tensor([5, 4, 3]), new Tensor([1, 2, 3, 4]), new Tensor([5, 5]),
            fieldSize, outputDepth, stride, zeroPad))
        .toThrowError();
  });

  it('Input depths dont match throws', () => {
    expect(
        () => g.conv2d(
            new Tensor([5, 4, 3]), new Tensor([1, 2, 100, 4]),
            new Tensor([outputDepth]), fieldSize, outputDepth, stride, zeroPad))
        .toThrowError();
  });

  it('Shapes matches does not throw', () => {
    const expectedShape = conv_util.computeOutputShape3D(
        [5, 4, 3], fieldSize, outputDepth, stride, zeroPad);
    expect(g.conv2d(
                new Tensor([5, 4, 3]), new Tensor([1, 2, 3, 4]),
                new Tensor([outputDepth]), fieldSize, outputDepth, stride,
                zeroPad)
               .shape)
        .toEqual(expectedShape);
  });
});

describe('maxpool validation', () => {
  let g: Graph;
  let fieldSize: number;
  let stride: number;
  let zeroPad: number;

  beforeEach(() => {
    g = new Graph();
    fieldSize = 4;
    stride = 1;
    zeroPad = 1;
  });

  it('Wrong rank x throws', () => {
    expect(() => g.maxPool(new Tensor([5, 4]), fieldSize, stride, zeroPad))
        .toThrowError();
  });

  it('Shapes matches does not throw', () => {
    const expectedShape = conv_util.computeOutputShape3D(
        [5, 4, 3], fieldSize, 3, stride, zeroPad);
    expect(g.maxPool(new Tensor([5, 4, 3]), fieldSize, stride, zeroPad).shape)
        .toEqual(expectedShape);
  });
});

describe('relu validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.relu(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('leakyRelu validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.leakyRelu(new Tensor([5, 4]), 0.2).shape).toEqual([5, 4]);
  });
});

describe('pRelu validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Different shapes throws', () => {
    expect(() => g.prelu(new Tensor([5, 4]), new Tensor([1, 2, 3])))
        .toThrowError();
  });

  it('Same size does not throw', () => {
    expect(g.prelu(new Tensor([5, 4]), new Tensor([5, 4])).shape).toEqual([
      5, 4
    ]);
  });
});

describe('elu validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.elu(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('exp validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.exp(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('log validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.log(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('tanh validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.tanh(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('sigmoid validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.sigmoid(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('square validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Does not throw', () => {
    expect(g.square(new Tensor([5, 4])).shape).toEqual([5, 4]);
  });
});

describe('softmaxCrossEntropy validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Shapes not equal throws', () => {
    expect(
        () => g.softmaxCrossEntropyCost(
            new Tensor([5, 4]), new Tensor([5, 4, 3])))
        .toThrowError();
  });

  it('Does not throw', () => {
    expect(
        g.softmaxCrossEntropyCost(new Tensor([5, 4]), new Tensor([5, 4])).shape)
        .toEqual([]);
  });
});

describe('meanSquaredCost validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Shapes not equal throws', () => {
    expect(() => g.meanSquaredCost(new Tensor([5, 4]), new Tensor([5, 4, 3])))
        .toThrowError();
  });

  it('Does not throw', () => {
    expect(g.meanSquaredCost(new Tensor([5, 4]), new Tensor([5, 4])).shape)
        .toEqual([]);
  });
});

describe('argmaxEquals validation', () => {
  let g: Graph;

  beforeEach(() => {
    g = new Graph();
  });

  it('Shapes not equal throws', () => {
    expect(() => g.argmaxEquals(new Tensor([5, 4]), new Tensor([5, 4, 3])))
        .toThrowError();
  });

  it('Does not throw', () => {
    expect(g.argmaxEquals(new Tensor([5, 4]), new Tensor([5, 4])).shape)
        .toEqual([1]);
  });
});

describe('Tensor', () => {
  it('captures shape from constructor', () => {
    const t = new Tensor([1, 2, 3, 4]);
    expect(t.shape).toEqual([1, 2, 3, 4]);
  });

  it('has unique ascending ids', () => {
    const a = new Tensor([]);
    const b = new Tensor([]);
    expect(b.id).toEqual(a.id + 1);
  });
});
