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

import {buildWeightMap, tensorflow} from './index';

describe('loaders', () => {
  describe('buildWeightMap', () => {
    let graph: tensorflow.GraphDef;
    let weight: ArrayBuffer;
    let graphPromise: Promise<tensorflow.GraphDef>;
    let weightPromise: Promise<ArrayBuffer>;
    beforeEach(() => {
      graph = new tensorflow.GraphDef();

      graphPromise =
          new Promise<tensorflow.GraphDef>((resolve) => resolve(graph));
    });

    it('should build int32 weights correctly', async () => {
      weight = new Int32Array([
                 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
               ]).buffer;
      weightPromise = new Promise<ArrayBuffer>((resolve) => resolve(weight));
      graph.node = [
        createConstNode('const1', [4], 0, 4, tensorflow.DataType.DT_INT32),
        createConstNode('const2', [2, 4], 16, 8, tensorflow.DataType.DT_INT32),
        createConstNode(
            'const3', [2, 2, 2, 2], 48, 16, tensorflow.DataType.DT_INT32)
      ];
      const constMap = await buildWeightMap(graphPromise, weightPromise);

      expect(constMap['const1'].rank).toBe(1);
      expect(constMap['const1'].dtype).toBe('int32');
      expect(Array.prototype.slice.call(constMap['const1'].dataSync()))
          .toEqual([1, 2, 3, 4]);

      expect(constMap['const2'].rank).toBe(2);
      expect(constMap['const2'].dtype).toBe('int32');
      expect(Array.prototype.slice.call(constMap['const2'].dataSync()))
          .toEqual([5, 6, 7, 8, 9, 10, 11, 12]);

      expect(constMap['const3'].rank).toBe(4);
      expect(constMap['const3'].dtype).toBe('int32');
      expect(Array.prototype.slice.call(constMap['const3'].dataSync()))
          .toEqual(
              [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]);
    });
    it('should build float32 weights correctly', async () => {
      weight = new Float32Array([1.0, 2.0, 3.0, 4.0]).buffer;
      weightPromise = new Promise<ArrayBuffer>((resolve) => resolve(weight));
      graph.node = [createConstNode(
          'const1', [1, 1, 4], 0, 4, tensorflow.DataType.DT_FLOAT)];
      const constMap = await buildWeightMap(graphPromise, weightPromise);

      expect(constMap['const1'].rank).toBe(3);
      expect(constMap['const1'].dtype).toBe('float32');
      expect(Array.prototype.slice.call(constMap['const1'].dataSync()))
          .toEqual([1.0, 2.0, 3.0, 4.0]);
    });
    it('should build bool weights correctly', async () => {
      weight = new Uint8Array([1, 0, 0, 1]).buffer;
      weightPromise = new Promise<ArrayBuffer>((resolve) => resolve(weight));
      graph.node = [createConstNode(
          'const1', [1, 1, 1, 4], 0, 4, tensorflow.DataType.DT_BOOL)];
      const constMap = await buildWeightMap(graphPromise, weightPromise);

      expect(constMap['const1'].rank).toBe(4);
      expect(constMap['const1'].dtype).toBe('bool');
      expect(Array.prototype.slice.call(constMap['const1'].dataSync()))
          .toEqual([1, 0, 0, 1]);
    });
  });

  function createConstNode(
      name: string, dims: number[], index: number, length: number,
      dtype: tensorflow.DataType): tensorflow.INodeDef {
    const node = new tensorflow.NodeDef();
    node.name = name;
    node.op = 'Const';
    const indexAttr = new tensorflow.AttrValue();
    indexAttr.i = index;
    const lengthAttr = new tensorflow.AttrValue();
    lengthAttr.i = length;
    const tensor = new tensorflow.Tensor();
    const tensorShape = new tensorflow.TensorShape();
    tensorShape.dim = dims.map((size) => {
      const dim = new tensorflow.TensorShape.Dim();
      dim.size = size;
      return dim;
    });
    tensor.tensorShape = tensorShape;
    tensor.dtype = dtype;
    const tensorAttr = new tensorflow.AttrValue();
    tensorAttr.tensor = tensor;

    node.attr = {index: indexAttr, length: lengthAttr, value: tensorAttr};
    return node;
  }
});
