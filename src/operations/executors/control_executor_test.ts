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
import {expectArraysClose} from '@tensorflow/tfjs-core/dist/test_util';

import {ExecutionContext} from '../../executor/execution_context';
import {TensorArray} from '../../executor/tensor_array';
import {Node} from '../types';

import {executeOp} from './control_executor';
// tslint:disable-next-line:max-line-length
import {createBoolAttr, createDtypeAttr, createNumberAttrFromIndex, createNumericArrayAttr, createNumericArrayAttrFromIndex, createStrAttr, createTensorAttr} from './test_helper';

describe('control', () => {
  let node: Node;
  const input1 = [tfc.scalar(1, 'int32')];
  const context = new ExecutionContext({}, {});

  beforeEach(() => {
    node = {
      name: 'test',
      op: '',
      category: 'control',
      inputNames: ['pred', 'input1'],
      inputs: [],
      params: {x: createTensorAttr(0)},
      children: []
    };
  });

  describe('executeOp', () => {
    describe('switch', () => {
      it('should set the output condition is true', async () => {
        node.op = 'switch';
        node.params['pred'] = createTensorAttr(0);
        node.params['data'] = createTensorAttr(1);

        const pred = [tfc.scalar(true)];
        expect(await executeOp(node, {pred, input1}, context)).toEqual([
          undefined, input1[0]
        ]);
      });
      it('should set the output condition is false', async () => {
        node.op = 'switch';
        node.params['pred'] = createTensorAttr(0);
        node.params['data'] = createTensorAttr(1);

        const pred = [tfc.scalar(false)];
        expect(await executeOp(node, {pred, input1}, context)).toEqual([
          input1[0], undefined
        ]);
      });
    });
    describe('merge', () => {
      it('should return the first available input', async () => {
        node.op = 'merge';

        const pred = [tfc.scalar(true)];
        expect(await executeOp(node, {pred: undefined, input1}, context))
            .toEqual(input1);
        expect(await executeOp(node, {pred, input1: undefined}, context))
            .toEqual(pred);
      });
      it('should return undefined if no inputs are available', async () => {
        node.op = 'merge';
        expect(await executeOp(
                   node, {pred: undefined, input1: undefined}, context))
            .toEqual(undefined);
      });
    });

    describe('enter', () => {
      it('should call enterFrame on context', async () => {
        spyOn(context, 'enterFrame');
        node.op = 'enter';
        node.params['tensor'] = createTensorAttr(0);
        node.inputNames = ['input1'];

        expect(await executeOp(node, {input1}, context)).toEqual(input1);
        expect(context.enterFrame).toHaveBeenCalled();
      });
    });
    describe('exit', () => {
      it('should call existFrame on context', async () => {
        spyOn(context, 'exitFrame');
        node.op = 'exit';
        node.params['tensor'] = createTensorAttr(0);
        node.inputNames = ['input1'];

        expect(await executeOp(node, {input1}, context)).toEqual(input1);
        expect(context.exitFrame).toHaveBeenCalled();
      });
    });
    describe('nextIteration', () => {
      it('should call nextIteration on context', async () => {
        spyOn(context, 'nextIteration');
        node.op = 'nextIteration';
        node.params['tensor'] = createTensorAttr(0);
        node.inputNames = ['input1'];

        expect(await executeOp(node, {input1}, context)).toEqual(input1);
        expect(context.nextIteration).toHaveBeenCalled();
      });
    });

    describe('tensorArray', () => {
      it('should create new tensor on the context', async () => {
        node.op = 'tensorArray';
        node.params['name'] = createStrAttr('');
        node.params['dtype'] = createDtypeAttr('int32');
        node.params['elementShape'] = createNumericArrayAttr([10, 10]);
        node.params['dynamicSize'] = createBoolAttr(false);
        node.params['clearAfterRead'] = createBoolAttr(true);
        node.params['identicalElementShapes'] = createBoolAttr(true);
        node.inputNames = ['input1'];

        const tensorId =
            (await executeOp(node, {input1}, context))[0].dataSync()[0];
        expect(context.getTensorArray(tensorId)).toBeDefined();
      });
    });

    describe('tensorArrayWrite', () => {
      it('should write the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [], true, false, true);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayWrite';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['index'] = createNumberAttrFromIndex(1);
        node.params['tensor'] = createTensorAttr(2);
        node.inputNames = ['input2', 'input3', 'input1'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [scalar(0)];
        await executeOp(node, {input1, input2, input3}, context);

        expect(tensorArray.size()).toEqual(1);
      });
    });

    describe('tensorArrayRead', () => {
      it('should read the tensor from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        tensorArray.write(0, input4);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayRead';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['index'] = createNumberAttrFromIndex(1);
        node.inputNames = ['input2', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [scalar(0)];
        const read = await executeOp(node, {input1, input2, input3}, context);

        expectArraysClose(read[0], input4);
      });
    });

    describe('tensorArrayGather', () => {
      it('should gather the tensors from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayGather';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['indices'] = createNumericArrayAttrFromIndex(1);
        node.params['dtype'] = createDtypeAttr('int32');
        node.inputNames = ['input2', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([0, 1])];
        const gather = await executeOp(node, {input2, input3}, context);
        expect(gather.length).toEqual(1);
        expect(gather[0].shape).toEqual([2, 3]);
        expectArraysClose(
            gather[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
      });
    });

    describe('tensorArrayScatter', () => {
      it('should scatter the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = [tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayScatter';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['indices'] = createNumericArrayAttrFromIndex(1);
        node.params['tensor'] = createTensorAttr(2);
        node.inputNames = ['input2', 'input3', 'input4'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([0, 1], 'int32')];
        await executeOp(node, {input2, input3, input4}, context);

        expect(tensorArray.size()).toEqual(2);
      });
    });

    describe('tensorArraySplit', () => {
      it('should split the tensor to tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 2, [3], true, false, true);
        const input4 = [tensor2d([0, 0, 0, 1, 1, 1], [2, 3], 'int32')];
        context.addTensorArray(tensorArray);
        node.op = 'tensorArraySplit';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['tensor'] = createTensorAttr(1);
        node.params['lengths'] = createNumericArrayAttrFromIndex(2);
        node.inputNames = ['input2', 'input4', 'input3'];
        const input2 = [scalar(tensorArray.id)];
        const input3 = [tensor1d([1, 1], 'int32')];
        await executeOp(node, {input2, input3, input4}, context);

        expect(tensorArray.size()).toEqual(2);
      });
    });

    describe('tensorArrayConcat', () => {
      it('should concat the tensors from tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayConcat';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.params['dtype'] = createDtypeAttr('int32');
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        const concat = await executeOp(node, {input2}, context);
        expect(concat.length).toEqual(1);
        expect(concat[0].shape).toEqual([6]);
        expectArraysClose(
            concat[0].dataSync(), new Int32Array([0, 0, 0, 1, 1, 1]));
      });
    });

    describe('tensorArraySize', () => {
      it('should get the size of tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArraySize';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        const size = await executeOp(node, {input2}, context);
        expect(size.length).toEqual(1);
        expect(size[0].shape).toEqual([]);
        expectArraysClose(size[0].dataSync(), new Int32Array([2]));
      });
    });

    describe('tensorArrayClose', () => {
      it('should close the tensorArray', async () => {
        const tensorArray =
            new TensorArray('', 'int32', 5, [3], true, false, true);
        const input4 = tensor1d([0, 0, 0], 'int32');
        const input5 = tensor1d([1, 1, 1], 'int32');
        tensorArray.writeMany([0, 1], [input4, input5]);
        context.addTensorArray(tensorArray);
        node.op = 'tensorArrayClose';
        node.params['tensorArrayId'] = createNumberAttrFromIndex(0);
        node.inputNames = ['input2'];
        const input2 = [scalar(tensorArray.id)];
        await executeOp(node, {input2}, context);
        expect(tensorArray.closed).toBeTruthy();
      });
    });
  });
});
