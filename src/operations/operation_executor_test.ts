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
// tslint:disable-next-line:max-line-length
import * as dl from 'deeplearn';

// tslint:disable-next-line:max-line-length
import {executeOp, getParamValue, Node, ParamValue} from './index';

describe('node', () => {
  let node: Node;
  beforeEach(() => {
    node = {
      name: 'test',
      op: 'const',
      inputNames: [],
      inputs: [],
      params: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('add', () => {
      it('should call dl.add', () => {
        spyOn(dl, 'add');
        node.op = 'add';
        node.inputNames = ['input1', 'input2'];
        const input1 = dl.Scalar.new(1);
        const input2 = dl.Scalar.new(1);

        executeOp(node, {input1, input2});

        expect(dl.add).toHaveBeenCalledWith(input1, input2);
      });
    });
    describe('const', () => {
      it('should return from weight hash', () => {
        const test = dl.Scalar.new(1);
        node.op = 'const';
        spyOn(dl, 'add');

        expect(executeOp(node, {test})).toBe(test);
      });
    });
    describe('placeholder', () => {
      it('should return from feedDict hash', () => {
        const test = dl.Scalar.new(1);
        node.op = 'placeholder';
        spyOn(dl, 'add');

        expect(executeOp(node, {test})).toBe(test);
      });
    });
    describe('floor', () => {
      it('should call dl.floor', () => {
        spyOn(dl, 'floor');
        node.op = 'floor';
        node.inputNames = ['input1'];
        const input1 = dl.Scalar.new(1);

        executeOp(node, {input1});

        expect(dl.floor).toHaveBeenCalledWith(input1);
      });
    });
    describe('mul', () => {
      it('should call dl.mul', () => {
        spyOn(dl, 'mul');
        node.op = 'mul';
        node.inputNames = ['input1', 'input2'];

        const input1 = dl.Scalar.new(1.0);
        const input2 = dl.Scalar.new(1.0);

        executeOp(node, {input1, input2});

        expect(dl.mul).toHaveBeenCalledWith(input1, input2);
      });
    });

    describe('matMul', () => {
      it('should call dl.matMul', () => {
        spyOn(dl, 'matMul');
        node.op = 'matMul';
        node.params['transposeA'] = createBoolAttr(false);
        node.params['transposeB'] = createBoolAttr(true);
        const input1 = dl.Scalar.new(1.0);
        const input2 = dl.Scalar.new(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.matMul).toHaveBeenCalledWith(input1, input2, false, true);
      });
    });

    describe('Conv2d', () => {
      it('should call dl.conv2d', () => {
        spyOn(dl, 'conv2d');
        node.op = 'conv2d';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        const input1 = dl.Scalar.new(1.0);
        const input2 = dl.Scalar.new(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.conv2d).toHaveBeenCalledWith(input1, input2, [2, 2], 'same');
      });
    });

    describe('depthwiseConv2d', () => {
      it('should call dl.depthwiseConv2d', () => {
        spyOn(dl, 'depthwiseConv2d');
        node.op = 'depthwiseConv2d';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['rate'] = createNumericArrayAttr([2, 2]);
        const input1 = dl.Scalar.new(1.0);
        const input2 = dl.Scalar.new(1.0);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.depthwiseConv2d)
            .toHaveBeenCalledWith(input1, input2, [2, 2], 'same', [2, 2]);
      });
    });

    describe('avgPool', () => {
      it('should call dl.avgPool', () => {
        spyOn(dl, 'avgPool');
        node.op = 'avgPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);
        const input = dl.Scalar.new(1.0);
        node.inputNames = ['input'];

        executeOp(node, {input});

        expect(dl.avgPool).toHaveBeenCalledWith(input, [2, 2], [2, 2], 'same');
      });
    });

    describe('maxPool', () => {
      it('should call dl.maxPool', () => {
        spyOn(dl, 'maxPool');
        node.op = 'maxPool';
        node.params['strides'] = createNumericArrayAttr([1, 2, 2, 1]);
        node.params['pad'] = createStrAttr('same');
        node.params['kernelSize'] = createNumericArrayAttr([1, 2, 2, 1]);
        const input = dl.Scalar.new(1.0);
        node.inputNames = ['input'];

        executeOp(node, {input});

        expect(dl.maxPool).toHaveBeenCalledWith(input, [2, 2], [2, 2], 'same');
      });
    });
    describe('randomUniform', () => {
      it('should call dl.randomUniform', () => {
        spyOn(dl, 'randomUniform');
        spyOn(dl, 'add');
        node.op = 'randomUniform';
        const input1 = dl.Tensor1D.new([2, 2, 2]);
        node.inputNames = ['input1'];
        node.params['maxVal'] = createNumberAttr(1);
        node.params['minVal'] = createNumberAttr(0);

        executeOp(node, {input1});

        expect(dl.randomUniform)
            .toHaveBeenCalledWith([2, 2, 2], 0, 1, 'float32');
      });
    });

    describe('div', () => {
      it('should call dl.div', () => {
        spyOn(dl, 'div');
        node.op = 'div';
        const input1 = dl.Scalar.new(1);
        const input2 = dl.Scalar.new(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.div).toHaveBeenCalledWith(input1, input2);
      });
    });

    describe('Reshape', () => {
      it('should call input reshape', () => {
        spyOn(dl, 'div');
        node.op = 'reshape';
        const input1 = dl.Tensor1D.new([1, 2, 3, 4]);
        const input2 = dl.Tensor1D.new([2, 2]);
        spyOn(input1, 'reshape');
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(input1.reshape).toHaveBeenCalledWith([2, 2]);
      });
    });
    describe('squeeze', () => {
      it('should call dl.squeeze', () => {
        spyOn(dl, 'squeeze');
        node.op = 'squeeze';
        node.params['axis'] = createNumericArrayAttr([1, 2]);
        const input = dl.Tensor3D.new([2, 1, 1], [1.0, 1.0]);
        node.inputNames = ['input'];

        executeOp(node, {input});

        expect(dl.squeeze).toHaveBeenCalledWith(input, [1, 2]);
      });
    });
    describe('sub', () => {
      it('should call dl.sub', () => {
        spyOn(dl, 'sub');
        node.op = 'sub';
        const input1 = dl.Scalar.new(1);
        const input2 = dl.Scalar.new(1);
        node.inputNames = ['input1', 'input2'];

        executeOp(node, {input1, input2});

        expect(dl.sub).toHaveBeenCalledWith(input1, input2);
      });
    });
    describe('relu', () => {
      it('should call dl.relu', () => {
        spyOn(dl, 'relu');
        node.op = 'relu';
        const input1 = dl.Scalar.new(1);
        node.inputNames = ['input1'];

        executeOp(node, {input1});

        expect(dl.relu).toHaveBeenCalledWith(input1);
      });
    });

    describe('clip', () => {
      it('should call dl.clipByValue', () => {
        spyOn(dl, 'clipByValue');
        node.op = 'clip';
        const input1 = dl.Scalar.new(1);
        node.inputNames = ['input1'];
        node.params['max'] = createNumberAttr(6);
        node.params['min'] = createNumberAttr(0);

        executeOp(node, {input1});

        expect(dl.clipByValue).toHaveBeenCalledWith(input1, 0, 6);
      });
    });

    describe('rsqrt', () => {
      it('should call dl.div', () => {
        const input1 = dl.Scalar.new(1);
        node.op = 'rsqrt';
        node.inputNames = ['input1'];
        spyOn(dl, 'div');
        spyOn(dl, 'sqrt').and.returnValue(input1);

        executeOp(node, {input1});

        expect(dl.sqrt).toHaveBeenCalledWith(input1);
        expect(dl.div).toHaveBeenCalledWith(jasmine.any(dl.Tensor), input1);
      });
    });

    describe('softmax', () => {
      it('should call dl.softmax', () => {
        spyOn(dl, 'softmax');
        node.op = 'softmax';
        const input1 = dl.Scalar.new(1);
        node.inputNames = ['input1'];

        executeOp(node, {input1});

        expect(dl.softmax).toHaveBeenCalledWith(input1);
      });
    });
    describe('identity', () => {
      it('should return the input', () => {
        node.op = 'identity';
        const input1 = dl.Scalar.new(1);
        node.inputNames = ['input1'];

        expect(executeOp(node, {input1})).toBe(input1);
      });
    });
    describe('concat', () => {
      it('should call dl.concat', () => {
        spyOn(dl, 'concat');
        node.op = 'concat';
        const input1 = dl.Tensor1D.new([1]);
        const input2 = dl.Tensor1D.new([1]);
        const input3 = dl.Scalar.new(0);
        node.inputNames = ['input1', 'input2', 'input3'];
        node.params['axis'] = createNumericArrayAttr([0]);

        executeOp(node, {input1, input2, input3});

        expect(dl.concat).toHaveBeenCalledWith([input1, input2], 0);
      });
    });

    describe('slice', () => {
      it('should call dl.slice', () => {
        spyOn(dl, 'slice');
        node.op = 'slice';
        const input1 = dl.Tensor1D.new([1, 2, 3]);
        const input2 = dl.Scalar.new(1);
        const input3 = dl.Scalar.new(1);
        node.inputNames = ['input1', 'input2', 'input3'];
        node.params['begin'] = createNumericArrayAttrFromIndex(1);
        node.params['size'] = createNumericArrayAttrFromIndex(2);

        executeOp(node, {input1, input2, input3});

        expect(dl.slice).toHaveBeenCalledWith(input1, [1], [1]);
      });
    });

    describe('fill', () => {
      it('should call dl.fill', () => {
        spyOn(dl, 'fill');
        node.op = 'fill';
        const input1 = dl.Tensor1D.new([1, 2, 3]);
        const input2 = dl.Scalar.new(1);
        node.inputNames = ['input1', 'input2'];
        node.params['shape'] = createNumericArrayAttrFromIndex(0);
        node.params['value'] = createNumberAttrFromIndex(1);

        executeOp(node, {input1, input2});

        expect(dl.fill).toHaveBeenCalledWith([1, 2, 3], 1);
      });
    });
  });

  describe('getParamValue', () => {
    it('should load bool correctly', () => {
      node.params['bool'] = createBoolAttr(false);
      expect(getParamValue('bool', node, {})).toBe(false);

      node.params['bool2'] = createBoolAttr(true);
      expect(getParamValue('bool2', node, {})).toBe(true);
    });
    it('should load str correctly', () => {
      const testString = 'test';
      node.params['string'] = createStrAttr(testString);

      expect(getParamValue('string', node, {})).toBe(testString);
    });

    it('should load tensor correctly', () => {
      const test = dl.Scalar.new(1, 'int32');
      node.inputNames = ['noused', 'test'];
      node.params['tensor'] = createTensorAttr(1);
      expect(getParamValue('tensor', node, {test})).toBe(test);
    });
  });
  it('should load number array correctly', () => {
    const test = dl.Scalar.new(1, 'int32');
    node.inputNames = ['test'];
    node.params['array'] = createNumericArrayAttr([1, 2]);
    expect(getParamValue('array', node, {test})).toEqual([1, 2]);
  });
});

function createNumberAttr(value: number): ParamValue {
  return {value, type: 'string'};
}

function createNumberAttrFromIndex(inputIndex: number): ParamValue {
  return {inputIndex, type: 'number'};
}

function createStrAttr(str: string): ParamValue {
  return {value: str, type: 'string'};
}

function createBoolAttr(value: boolean): ParamValue {
  return {value, type: 'bool'};
}

function createNumericArrayAttr(value: number[]): ParamValue {
  return {value, type: 'number[]'};
}

function createNumericArrayAttrFromIndex(inputIndex: number): ParamValue {
  return {inputIndex, type: 'number[]'};
}

function createTensorAttr(index: number): ParamValue {
  return {inputIndex: index, type: 'tensor'};
}
