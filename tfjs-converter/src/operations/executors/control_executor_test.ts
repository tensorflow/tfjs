/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import * as tfc from '@tensorflow/tfjs-core';
import {scalar, tensor1d, tensor2d} from '@tensorflow/tfjs-core';
import {test_util} from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../../executor/execution_context';
import {GraphExecutor} from '../../executor/graph_executor';
import {TensorArray} from '../../executor/tensor_array';
import * as control from '../op_list/control';
import {Graph, Node} from '../types';

import {executeOp} from './control_executor';
import {createBoolAttr, createDtypeAttr, createNumberAttrFromIndex, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr, createTensorsAttr, createTensorShapeAttr, validateParam} from './test_helper';

describe('control', () => {
  let node: Node;
  const input1 = [tfc.scalar(1, 'int32')];
  const input2 = [tfc.scalar(0, 'bool')];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'control',
      inputNames: ['input1', 'pred'],
      inputs: [],
      inputParams: {},
      attrParams: {},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('Switch', () => {
      it('should set the output condition is true', async () => {
        node.op = 'Switch';
        node.inputParams['pred'] = createTensorAttr(1);
        node.inputParams['data'] = createTensorAttr(0);

        const pred = [tfc.scalar(true)];
        const result = await executeOp(node, {pred, input1}, context);
        expect(result[0]).toBeUndefined();
        test_util.expectArraysEqual(
            await result[1].array(), await input1[0].array());
      });
      it('should set the output condition is false', async () => {
        node.op = 'Switch';
        node.inputParams['pred'] = createTensorAttr(1);
        node.inputParams['data'] = createTensorAttr(0);

        const pred = [tfc.scalar(false)];
        const result = await executeOp(node, {pred, input1}, context);
        test_util.expectArraysEqual(
            await result[0].array(), await input1[0].array());
        expect(result[1]).toBeUndefined();
      });
      it('should match json def', () => {
        node.op = 'Switch';
        node.inputParams['pred'] = createTensorAttr(1);
        node.inputParams['data'] = createTensorAttr(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });
    describe('Merge', () => {
      it('should return the first available input', async () => {
        node.op = 'Merge';

        const pred = [tfc.scalar(true)];
        test_util.expectArraysEqual(
            await (await executeOp(node, {pred: undefined, input1}, context))[0]
                .array(),
            await input1[0].array());
        test_util.expectArraysEqual(
            await (await executeOp(node, {pred, input1: undefined}, context))[0]
                .array(),
            await pred[0].array());
      });
      it('should return undefined if no inputs are available', async () => {
        node.op = 'Merge';
        expect(await executeOp(
                   node, {pred: undefined, input1: undefined}, context))
            .toEqual(undefined);
      });
      it('should match json def', () => {
        node.op = 'Merge';

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('Enter', () => {
      it('should call enterFrame on context', async () => {
        spyOn(context, 'enterFrame');
        node.op = 'Enter';
        node.inputParams['tensor'] = createTensorAttr(0);
        node.attrParams['frameName'] = createStrAttr('test');
        node.inputNames = ['input1'];

        test_util.expectArraysEqual(
            await (await executeOp(node, {input1}, context))[0].array(),
            await input1[0].array());
        expect(context.enterFrame).toHaveBeenCalled();
      });
      it('should match json def', () => {
        node.op = 'Enter';
        node.inputParams['tensor'] = createTensorAttr(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });
    describe('Exit', () => {
      it('should call existFrame on context', async () => {
        spyOn(context, 'exitFrame');
        node.op = 'Exit';
        node.inputParams['tensor'] = createTensorAttr(0);
        node.inputNames = ['input1'];

        test_util.expectArraysEqual(
            await (await executeOp(node, {input1}, context))[0].array(),
            await input1[0].array());
        expect(context.exitFrame).toHaveBeenCalled();
      });
      it('should match json def', () => {
        node.op = 'Exit';
        node.inputParams['tensor'] = createTensorAttr(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });
    describe('NextIteration', () => {
      it('should call nextIteration on context', async () => {
        spyOn(context, 'nextIteration');
        node.op = 'NextIteration';
        node.inputParams['tensor'] = createTensorAttr(0);
        node.inputNames = ['input1'];

        test_util.expectArraysEqual(
            await (await executeOp(node, {input1}, context))[0].array(),
            await input1[0].array());
        expect(context.nextIteration).toHaveBeenCalled();
      });
      it('should match json def', () => {
        node.op = 'NextIteration';
        node.inputParams['tensor'] = createTensorAttr(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayV3', () => {
      it('should create new tensor on the context', async () => {
        node.op = 'TensorArrayV3';
        node.inputParams['size'] = createNumberAttrFromIndex(0);
        node.attrParams['name'] = createStrAttr('');
        node.attrParams['dtype'] = createDtypeAttr('int32');
        node.attrParams['elementShape'] = createTensorShapeAttr([10, 10]);
        node.attrParams['dynamicSize'] = createBoolAttr(false);
        node.attrParams['clearAfterRead'] = createBoolAttr(true);
        node.attrParams['identicalElementShapes'] = createBoolAttr(true);
        node.inputNames = ['input1'];

        const tensorId =
            (await executeOp(node, {input1}, context))[0].dataSync()[0];
        expect(context.getTensorArray(tensorId)).toBeDefined();
      });
      it('should match json def', () => {
        node.op = 'TensorArrayV3';
        node.inputParams['size'] = createNumberAttrFromIndex(0);
        node.attrParams['name'] = createStrAttr('');
        node.attrParams['dtype'] = createDtypeAttr('int32');
        node.attrParams['elementShape'] = createTensorShapeAttr([10, 10]);
        node.attrParams['dynamicSize'] = createBoolAttr(false);
        node.attrParams['clearAfterRead'] = createBoolAttr(true);
        node.attrParams['identicalElementShapes'] = createBoolAttr(true);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayWriteV3', () => {
      it('should write the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [], true, false, true);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayWriteV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['index'] = createNumberAttrFromIndex(1);
        node.inputParams['tensor'] = createTensorAttr(2);
        node.inputNames = ['input2', 'input3', 'input1'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [scalar(0)];
        await executeOp(node, {input1, input2, input3}, context);

        expect(tensorArray.size()).toEqual(1);
      });
      it('should match json def', () => {
        node.op = 'TensorArrayWriteV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['index'] = createNumberAttrFromIndex(1);
        node.inputParams['tensor'] = createTensorAttr(2);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayReadV3', () => {
      it('should read the tensor from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        tensorArray.write(0, input4);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayReadV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['index'] = createNumberAttrFromIndex(1);
        node.inputNames = ['input2', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [scalar(0)];
        const read = await executeOp(node, {input1, input2, input3}, context);

        test_util.expectArraysClose(
            await read[0].array(), await input4.array());
      });
      it('should match json def', () => {
        node.op = 'TensorArrayReadV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['index'] = createNumberAttrFromIndex(1);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayGatherV3', () => {
      it('should gather the tensors from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayGatherV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['indices'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');
        node.inputNames = ['input2', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([0, 1])];
        const gather = await executeOp(node, {input2, input3}, context);
        expect(gather.length).toEqual(1);
        expect(gather[0].shape).toEqual([2, 3]);
        test_util.expectArraysClose(
            gather[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
      });
      it('should match json def', () => {
        node.op = 'TensorArrayGatherV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['indices'] = createNumericArrayAttrFromIndex(1);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayScatterV3', () => {
      it('should scatter the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = [tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayScatterV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['indices'] = createNumericArrayAttrFromIndex(1);
        node.inputParams['tensor'] = createTensorAttr(2);
        node.inputNames = ['input2', 'input3', 'input4'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([0, 1], 'int32')];
        await executeOp(node, {input2, input3, input4}, context);

        expect(tensorArray.size()).toEqual(2);
      });

      it('should match json def', () => {
        node.op = 'TensorArrayScatterV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['indices'] = createNumericArrayAttrFromIndex(1);
        node.inputParams['tensor'] = createTensorAttr(2);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArraySplitV3', () => {
      it('should split the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 2, [3], true, false, true);
        const input4 = [tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
        context.addTensorArray(tensorArray);
        node.op = 'TensorArraySplitV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['tensor'] = createTensorAttr(1);
        node.inputParams['lengths'] = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input2', 'input4', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([1, 1], 'int32')];
        await executeOp(node, {input2, input3, input4}, context);

        expect(tensorArray.size()).toEqual(2);
      });
      it('should match json def', () => {
        node.op = 'TensorArraySplitV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputParams['tensor'] = createTensorAttr(1);
        node.inputParams['lengths'] = createNumericArrayAttrFromIndex(2);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayConcatV3', () => {
      it('should concat the tensors from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayConcatV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('int32');
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        const concat = await executeOp(node, {input2}, context);
        expect(concat.length).toEqual(1);
        expect(concat[0].shape).toEqual([6]);
        test_util.expectArraysClose(
            concat[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
      });
      it('should match json def', () => {
        node.op = 'TensorArrayConcatV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.attrParams['dtype'] = createDtypeAttr('int32');

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArraySizeV3', () => {
      it('should get the size of tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArraySizeV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        const size = await executeOp(node, {input2}, context);
        expect(size.length).toEqual(1);
        expect(size[0].shape).toEqual([]);
        test_util.expectArraysClose(size[0].dataSync(), new Int32Array([2]));
      });
      it('should match json def', () => {
        node.op = 'TensorArraySizeV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });

    describe('TensorArrayCloseV3', () => {
      it('should close the tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'TensorArrayCloseV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        await executeOp(node, {input2}, context);
        expect(tensorArray.closed).toBeTruthy();
      });
      it('should match json def', () => {
        node.op = 'TensorArrayCloseV3';
        node.inputParams['tensorArrayId'] = createNumberAttrFromIndex(0);

        expect(validateParam(node, control.json)).toBeTruthy();
      });
    });
  });
  describe('StatelessWhile', () => {
    it('should set the output', async () => {
      node.op = 'StatelessWhile';
      node.inputNames = ['input1', 'input2'];
      node.inputParams['args'] = createTensorsAttr(0, 0);
      node.attrParams['cond'] = {'value': 'condFunc', 'type': 'func'};
      node.attrParams['body'] = {'value': 'bodyFunc', 'type': 'func'};

      const cond = [tfc.scalar(false)];
      const graph: Graph = {
        inputs: [],
        nodes: {

        },
        outputs: [],
        weights: [],
        placeholders: [],
        functions: {},
        signature: {}
      };
      const condExecutor = new GraphExecutor(graph);
      let firstTime = true;
      spyOn(condExecutor, 'executeFunctionAsync').and.callFake(() => {
        if (firstTime) {
          firstTime = false;
          return input1;
        }
        return input2;
      });
      const bodyExecutor = new GraphExecutor(graph);
      spyOn(bodyExecutor, 'executeFunctionAsync').and.returnValue(input2);
      context.functionMap['bodyFunc'] = bodyExecutor;
      context.functionMap['condFunc'] = condExecutor;
      const result = await executeOp(node, {cond, input1, input2}, context);

      test_util.expectArraysEqual(
          await result[0].array(), await input2[0].array());
    });

    it('should match json def', () => {
      node.op = 'StatelessWhile';
      node.inputNames = ['input1', 'input2'];
      node.inputParams['args'] = createTensorsAttr(0, 0);
      node.attrParams['cond'] = {'value': 'condFunc', 'type': 'func'};
      node.attrParams['body'] = {'value': 'bodyFunc', 'type': 'func'};

      expect(validateParam(node, control.json)).toBeTruthy();
    });
  });

  describe('StatelessIf', () => {
    it('should set the output condition is true', async () => {
      node.op = 'StatelessIf';
      node.inputNames = ['cond', 'input1', 'input2'];
      node.inputParams['args'] = createTensorsAttr(1, 0);
      node.inputParams['cond'] = createTensorAttr(0);
      node.attrParams['thenBranch'] = {'value': 'thenFunc', 'type': 'func'};
      node.attrParams['elseBranch'] = {'value': 'elseFunc', 'type': 'func'};

      const cond = [tfc.scalar(true)];
      const graph: Graph = {
        inputs: [],
        nodes: {

        },
        outputs: [],
        weights: [],
        placeholders: [],
        functions: {},
        signature: {}
      };
      const thenExecutor = new GraphExecutor(graph);
      spyOn(thenExecutor, 'executeFunctionAsync').and.returnValue(input1);
      const elseExecutor = new GraphExecutor(graph);
      spyOn(elseExecutor, 'executeFunctionAsync').and.returnValue(input2);
      context.functionMap['thenFunc'] = thenExecutor;
      context.functionMap['elseFunc'] = elseExecutor;
      const result = await executeOp(node, {cond, input1, input2}, context);

      test_util.expectArraysEqual(
          await result[0].array(), await input1[0].array());
    });
    it('should set the output condition is false', async () => {
      node.op = 'StatelessIf';
      node.inputNames = ['cond', 'input1'];
      node.inputParams['args'] = createTensorsAttr(1, 0);
      node.inputParams['cond'] = createTensorAttr(0);
      node.attrParams['thenBranch'] = {'value': 'thenFunc', 'type': 'func'};
      node.attrParams['elseBranch'] = {'value': 'elseFunc', 'type': 'func'};

      const cond = [tfc.scalar(false)];
      const graph: Graph = {
        inputs: [],
        nodes: {

        },
        outputs: [],
        weights: [],
        placeholders: [],
        functions: {},
        signature: {}
      };
      const thenExecutor = new GraphExecutor(graph);
      spyOn(thenExecutor, 'executeFunctionAsync').and.returnValue(input1);
      const elseExecutor = new GraphExecutor(graph);
      spyOn(elseExecutor, 'executeFunctionAsync').and.returnValue(input2);
      context.functionMap['thenFunc'] = thenExecutor;
      context.functionMap['elseFunc'] = elseExecutor;
      const result = await executeOp(node, {cond, input1, input2}, context);

      test_util.expectArraysEqual(
          await result[0].array(), await input2[0].array());
    });
    it('should match json def', () => {
      node.op = 'StatelessIf';
      node.inputNames = ['cond', 'input1'];
      node.inputParams['args'] = createTensorsAttr(1, 0);
      node.inputParams['cond'] = createTensorAttr(0);
      node.attrParams['thenBranch'] = {'value': 'thenFunc', 'type': 'func'};
      node.attrParams['elseBranch'] = {'value': 'elseFunc', 'type': 'func'};

      expect(validateParam(node, control.json)).toBeTruthy();
    });
  });
});
