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

import {DataType, scalar, Tensor} from '@tensorflow/tfjs-core';

import {NamedTensorsMap} from '../../data/types';
import {ExecutionContext} from '../../executor/execution_context';
import {TensorArray} from '../../executor/tensor_array';
import {fromTensor, reserve, scatter, split} from '../../executor/tensor_list';
import {InternalOpAsyncExecutor, Node} from '../types';

import {cloneTensor, getParamValue, getTensor} from './utils';

export const executeOp: InternalOpAsyncExecutor = async(
    node: Node, tensorMap: NamedTensorsMap,
    context: ExecutionContext): Promise<Tensor[]> => {
  switch (node.op) {
    case 'If':
    case 'StatelessIf': {
      const thenFunc =
          getParamValue('thenBranch', node, tensorMap, context) as string;
      const elseFunc =
          getParamValue('elseBranch', node, tensorMap, context) as string;
      const cond = getParamValue('cond', node, tensorMap, context) as Tensor;
      const args = getParamValue('args', node, tensorMap, context) as Tensor[];
      const condValue = await cond.data();
      if (condValue[0]) {
        return context.functionMap[thenFunc].executeFunctionAsync(
            args, context.tensorArrayMap, context.tensorListMap);
      } else {
        return context.functionMap[elseFunc].executeFunctionAsync(
            args, context.tensorArrayMap, context.tensorListMap);
      }
    }
    case 'While':
    case 'StatelessWhile': {
      const bodyFunc =
          getParamValue('body', node, tensorMap, context) as string;
      const condFunc =
          getParamValue('cond', node, tensorMap, context) as string;
      const args = getParamValue('args', node, tensorMap, context) as Tensor[];

      // Calculate the condition of the loop
      const condResult =
          (await context.functionMap[condFunc].executeFunctionAsync(
              args, context.tensorArrayMap, context.tensorListMap));
      const argIds = args.map(tensor => tensor.id);
      let condValue = await condResult[0].data();
      // Dispose the intermediate tensors for condition function
      condResult.forEach(tensor => {
        if (!tensor.kept && argIds.indexOf(tensor.id) === -1) {
          tensor.dispose();
        }
      });

      let result: Tensor[] = args;

      while (condValue[0]) {
        // Record the previous result for intermediate tensor tracking
        const origResult = result;
        // Execution the body of the loop
        result = await context.functionMap[bodyFunc].executeFunctionAsync(
            result, context.tensorArrayMap, context.tensorListMap);
        const resultIds = result.map(tensor => tensor.id);

        // Dispose the intermediate tensor for body function that is not global
        // kept, not input/output of the body function
        origResult.forEach(tensor => {
          if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
              resultIds.indexOf(tensor.id) === -1) {
            tensor.dispose();
          }
        });

        // Recalcuate the condition of the loop using the latest results.
        const condResult =
            (await context.functionMap[condFunc].executeFunctionAsync(
                result, context.tensorArrayMap, context.tensorListMap));
        condValue = await condResult[0].data();
        // Dispose the intermediate tensors for condition function
        condResult.forEach(tensor => {
          if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
              resultIds.indexOf(tensor.id) === -1) {
            tensor.dispose();
          }
        });
      }
      return result;
    }
    case 'LoopCond': {
      const pred = getParamValue('pred', node, tensorMap, context) as Tensor;
      return [cloneTensor(pred)];
    }
    case 'Switch': {
      const pred = getParamValue('pred', node, tensorMap, context) as Tensor;
      let data = getParamValue('data', node, tensorMap, context) as Tensor;
      if (!data.kept) {
        data = cloneTensor(data);
      }
      // Outputs nodes :0 => false, :1 => true
      return (await pred.data())[0] ? [undefined, data] : [data, undefined];
    }
    case 'Merge': {
      const inputName = node.inputNames.find(
          name => getTensor(name, tensorMap, context) !== undefined);
      if (inputName) {
        const data = getTensor(inputName, tensorMap, context);
        return [cloneTensor(data)];
      }
      return undefined;
    }
    case 'Enter': {
      const frameId =
          getParamValue('frameName', node, tensorMap, context) as string;
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.enterFrame(frameId);
      return [cloneTensor(data)];
    }
    case 'Exit': {
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.exitFrame();
      return [cloneTensor(data)];
    }
    case 'NextIteration': {
      const data = getParamValue('tensor', node, tensorMap, context) as Tensor;
      context.nextIteration();
      return [cloneTensor(data)];
    }
    case 'TensorArrayV3': {
      const size = getParamValue('size', node, tensorMap, context) as number;
      const dtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const dynamicSize =
          getParamValue('dynamicSize', node, tensorMap, context) as boolean;
      const clearAfterRead =
          getParamValue('clearAfterRead', node, tensorMap, context) as boolean;
      const identicalElementShapes =
          getParamValue('identicalElementShapes', node, tensorMap, context) as
          boolean;
      const name = getParamValue('name', node, tensorMap, context) as string;
      const tensorArray = new TensorArray(
          name, dtype, size, elementShape, identicalElementShapes, dynamicSize,
          clearAfterRead);
      context.addTensorArray(tensorArray);
      return [tensorArray.idTensor, scalar(1.0)];
    }
    case 'TensorArrayWriteV3': {
      const id =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const index = getParamValue('index', node, tensorMap, context) as number;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const writeTensorArray = context.getTensorArray(id.id);
      writeTensorArray.write(index, writeTensor);
      return [writeTensorArray.idTensor];
    }
    case 'TensorArrayReadV3': {
      const readId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const readIndex =
          getParamValue('index', node, tensorMap, context) as number;
      const readTensorArray = context.getTensorArray(readId.id);
      return [readTensorArray.read(readIndex)];
    }
    case 'TensorArrayGatherV3': {
      const gatherId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const gatherIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const gatherDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const gatherTensorArray = context.getTensorArray(gatherId.id);
      return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
    }
    case 'TensorArrayScatterV3': {
      const scatterId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const scatterIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const scatterTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const scatterTensorArray = context.getTensorArray(scatterId.id);
      scatterTensorArray.scatter(scatterIndices, scatterTensor);
      return [scatterTensorArray.idTensor];
    }
    case 'TensorArrayConcatV3': {
      const concatId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const concatTensorArray = context.getTensorArray(concatId.id);
      const concatDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      return [concatTensorArray.concat(concatDtype)];
    }
    case 'TensorArraySplitV3': {
      const splitId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const splitTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const lengths =
          getParamValue('lengths', node, tensorMap, context) as number[];
      const splitTensorArray = context.getTensorArray(splitId.id);
      splitTensorArray.split(lengths, splitTensor);
      return [splitTensorArray.idTensor];
    }
    case 'TensorArraySizeV3': {
      const sizeId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const sizeTensorArray = context.getTensorArray(sizeId.id);
      return [scalar(sizeTensorArray.size(), 'int32')];
    }
    case 'TensorArrayCloseV3': {
      const closeId =
          getParamValue('tensorArrayId', node, tensorMap, context) as Tensor;
      const closeTensorArray = context.getTensorArray(closeId.id);
      closeTensorArray.clearAndClose();
      return [closeTensorArray.idTensor];
    }
    case 'TensorListSetItem': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const index = getParamValue('index', node, tensorMap, context) as number;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(idTensor.id);
      tensorList.setItem(index, writeTensor);
      return [tensorList.idTensor];
    }
    case 'TensorListGetItem': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const readIndex =
          getParamValue('index', node, tensorMap, context) as number;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];

      const elementDType =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.getItem(readIndex, elementShape, elementDType)];
    }
    case 'TensorListScatterV2':
    case 'TensorListScatter': {
      const scatterIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const scatterTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const numElements =
          getParamValue('numElements', node, tensorMap, context) as number;
      const tensorList =
          scatter(scatterTensor, scatterIndices, elementShape, numElements);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListReserve':
    case 'EmptyTensorList': {
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      let numElementsParam;

      if (node.op === 'TensorListReserve') {
        numElementsParam = 'numElements';
      } else {
        numElementsParam = 'maxNumElements';
      }

      const numElements =
          getParamValue(numElementsParam, node, tensorMap, context) as number;

      const tensorList = reserve(elementShape, elementDtype, numElements);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListGather': {
      const gatherId =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const gatherIndices =
          getParamValue('indices', node, tensorMap, context) as number[];
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(gatherId.id);
      return [tensorList.gather(gatherIndices, elementDtype, elementShape)];
    }
    case 'TensorListStack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const numElements =
          getParamValue('numElements', node, tensorMap, context) as number;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.stack(elementShape, elementDtype, numElements)];
    }
    case 'TensorListFromTensor': {
      const tensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDtype =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = fromTensor(tensor, elementShape, elementDtype);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    case 'TensorListConcat': {
      const concatId =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(concatId.id);
      const concatDtype =
          getParamValue('dtype', node, tensorMap, context) as DataType;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      return [tensorList.concat(concatDtype, elementShape)];
    }
    case 'TensorListPushBack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const writeTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const tensorList = context.getTensorList(idTensor.id);
      tensorList.pushBack(writeTensor);
      return [tensorList.idTensor];
    }
    case 'TensorListPopBack': {
      const idTensor =
          getParamValue('tensorListId', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const elementDType =
          getParamValue('elementDType', node, tensorMap, context) as DataType;
      const tensorList = context.getTensorList(idTensor.id);
      return [tensorList.popBack(elementShape, elementDType)];
    }
    case 'TensorListSplit': {
      const splitTensor =
          getParamValue('tensor', node, tensorMap, context) as Tensor;
      const elementShape =
          getParamValue('elementShape', node, tensorMap, context) as number[];
      const lengths =
          getParamValue('lengths', node, tensorMap, context) as number[];

      const tensorList = split(splitTensor, lengths, elementShape);
      context.addTensorList(tensorList);
      return [tensorList.idTensor];
    }
    default:
      throw TypeError(`Node type ${node.op} is not implemented`);
  }
};

export const CATEGORY = 'control';
