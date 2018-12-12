/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

// tslint:disable-next-line:no-require-imports
import bindings = require('bindings');
import {TFJSBinding, TFEOpAttr} from './tfjs_binding';
const binding = bindings('tfjs_binding.node') as TFJSBinding;

describe('Exposes TF_DataType enum values', () => {
  it('contains TF_FLOAT', () => {
    expect(binding.TF_FLOAT).toEqual(1);
  });
  it('contains TF_INT32', () => {
    expect(binding.TF_INT32).toEqual(3);
  });
  it('contains TF_BOOL', () => {
    expect(binding.TF_BOOL).toEqual(10);
  });
  it('contains TF_COMPLEX64', () => {
    expect(binding.TF_COMPLEX64).toEqual(8);
  });
  it('contains TF_STRING', () => {
    expect(binding.TF_STRING).toEqual(7);
  });
});

describe('Exposes TF_AttrType enum values', () => {
  it('contains TF_ATTR_STRING', () => {
    expect(binding.TF_ATTR_STRING).toEqual(0);
  });
  it('contains TF_ATTR_INT', () => {
    expect(binding.TF_ATTR_INT).toEqual(1);
  });
  it('contains TF_ATTR_FLOAT', () => {
    expect(binding.TF_ATTR_FLOAT).toEqual(2);
  });
  it('contains TF_ATTR_BOOL', () => {
    expect(binding.TF_ATTR_BOOL).toEqual(3);
  });
  it('contains TF_ATTR_TYPE', () => {
    expect(binding.TF_ATTR_TYPE).toEqual(4);
  });
  it('contains TF_ATTR_SHAPE', () => {
    expect(binding.TF_ATTR_SHAPE).toEqual(5);
  });
});

describe('Exposes TF Version', () => {
  it('contains a version string', () => {
    expect(binding.TF_Version).toBeDefined();
  });
});

describe('tensor management', () => {
  it('Creates and deletes a valid tensor', () => {
    const values = new Int32Array([1, 2]);
    const id = binding.createTensor([2], binding.TF_INT32, values);
    expect(id).toBeDefined();

    binding.deleteTensor(id);
  });
  it('throws exception when shape does not match data', () => {
    expect(() => {
      binding.createTensor([2], binding.TF_INT32, new Int32Array([1, 2, 3]));
    }).toThrowError();
    expect(() => {
      binding.createTensor([4], binding.TF_INT32, new Int32Array([1, 2, 3]));
    }).toThrowError();
  });
  it('throws exception with invalid dtype', () => {
    expect(() => {
      // tslint:disable-next-line:no-unused-expression
      binding.createTensor([1], 1000, new Int32Array([1]));
    }).toThrowError();
  });
  it('works with 0-dim tensors', () => {
    // Reduce op (e.g 'Max') will produce a 0-dim TFE_Tensor.

    const inputId =
        binding.createTensor([3], binding.TF_INT32, new Int32Array([1, 2, 3]));
    const axesId =
        binding.createTensor([1], binding.TF_INT32, new Int32Array([0]));

    const attrs = [
      {name: 'keep_dims', type: binding.TF_ATTR_BOOL, value: false},
      {name: 'T', type: binding.TF_ATTR_TYPE, value: binding.TF_INT32},
      {name: 'Tidx', type: binding.TF_ATTR_TYPE, value: binding.TF_INT32}
    ];

    const outputMetadata =
        binding.executeOp('Max', attrs, [inputId, axesId], 1);
    expect(outputMetadata.length).toBe(1);

    expect(outputMetadata[0].id).toBeDefined();
    expect(outputMetadata[0].shape).toEqual([]);
    expect(outputMetadata[0].dtype).toEqual(binding.TF_INT32);
    expect(binding.tensorDataSync(outputMetadata[0].id))
        .toEqual(new Int32Array([3]));
  });
});

describe('executeOp', () => {
  const name = 'MatMul';
  const matMulOpAttrs = [
    {name: 'transpose_a', type: binding.TF_ATTR_BOOL, value: false},
    {name: 'transpose_b', type: binding.TF_ATTR_BOOL, value: false},
    {name: 'T', type: binding.TF_ATTR_TYPE, value: binding.TF_FLOAT}
  ];
  const aId = binding.createTensor(
      [2, 2], binding.TF_FLOAT, new Float32Array([1, 2, 3, 4]));
  const bId = binding.createTensor(
      [2, 2], binding.TF_FLOAT, new Float32Array([4, 3, 2, 1]));
  const matMulInput = [aId, bId];

  it('throws exception with invalid Op Name', () => {
    expect(() => {
      binding.executeOp(null, [] as TFEOpAttr[], [] as number[], null);
    }).toThrowError();
  });
  it('throws exception with invalid TFEOpAttr', () => {
    expect(() => {
      binding.executeOp('Equal', null, [] as number[], null);
    }).toThrowError();
  });
  it('throws excpetion with invalid inputs', () => {
    expect(() => {
      binding.executeOp(name, matMulOpAttrs, [] as number[], null);
    }).toThrowError();
  });
  it('throws exception with invalid output number', () => {
    expect(() => {
      binding.executeOp(name, matMulOpAttrs, matMulInput, null);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_STRING op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_STRING, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_STRING, value: false}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_STRING, value: 1}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_STRING, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_STRING, value: [1, 2, 3]}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_INT op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_INT, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_INT, value: false}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_INT, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_INT, value: 'test'}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_INT, value: [1, 2, 3]}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_FLOAT op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_FLOAT, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_FLOAT, value: false}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_FLOAT, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_FLOAT, value: 'test'}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_FLOAT, value: [1, 2, 3]}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_BOOL op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_BOOL, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_BOOL, value: 10}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_BOOL, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_BOOL, value: 'test'}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_BOOL, value: [1, 2, 3]}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_TYPE op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: 'test'}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: [1, 2, 3]}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('throws exception with invalid TF_ATTR_SHAPE op attr', () => {
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: null}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: new Object()}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
    expect(() => {
      const badOpAttrs: TFEOpAttr[] =
          [{name: 'T', type: binding.TF_ATTR_TYPE, value: 'test'}];
      binding.executeOp(name, badOpAttrs, matMulInput, 1);
    }).toThrowError();
  });
  it('should work for matmul', () => {
    const output = binding.executeOp(name, matMulOpAttrs, matMulInput, 1);
    expect(binding.tensorDataSync(output[0].id)).toEqual(new Float32Array([
      8, 5, 20, 13
    ]));
  });
});
