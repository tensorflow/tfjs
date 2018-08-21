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

import * as tfc from '@tensorflow/tfjs-core';

import {NodeJSKernelBackend} from '../nodejs_kernel_backend';

// tslint:disable-next-line:max-line-length
import {createTensorsTypeOpAttr, createTypeOpAttr, getTFDType, nodeBackend} from './op_utils';

describe('Exposes Backend for internal Op execution.', () => {
  it('Provides the Node backend over a function', () => {
    const backend = nodeBackend();
    expect(backend instanceof NodeJSKernelBackend).toBeTruthy();
  });

  it('Provides internal access to the binding', () => {
    expect(nodeBackend().binding).toBeDefined();
  });
});

describe('getTFDType()', () => {
  const binding = nodeBackend().binding;

  it('handles float32', () => {
    expect(getTFDType('float32')).toBe(binding.TF_FLOAT);
  });
  it('handles int32', () => {
    expect(getTFDType('int32')).toBe(binding.TF_INT32);
  });
  it('handles bool', () => {
    expect(getTFDType('bool')).toBe(binding.TF_BOOL);
  });
  it('handles unknown types', () => {
    expect(() => getTFDType(null)).toThrowError();
  });
});

describe('createTypeOpAttr()', () => {
  const binding = nodeBackend().binding;

  it('Creates a valid type attribute', () => {
    const attr = createTypeOpAttr('foo', 'float32');
    expect(attr.name).toBe('foo');
    expect(attr.type).toBe(binding.TF_ATTR_TYPE);
    expect(attr.value).toBe(binding.TF_FLOAT);
  });

  it('handles unknown dtypes', () => {
    expect(() => createTypeOpAttr('foo', null)).toThrowError();
  });
});

describe('Returns TFEOpAttr for a Tensor or list of Tensors', () => {
  const binding = nodeBackend().binding;

  it('handles a single Tensor', () => {
    const result = createTensorsTypeOpAttr('T', tfc.scalar(13, 'float32'));
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_FLOAT);
  });
  it('handles a list of Tensors', () => {
    const tensors = [tfc.scalar(1, 'int32'), tfc.scalar(20.1, 'float32')];
    const result = createTensorsTypeOpAttr('T', tensors);
    expect(result.name).toBe('T');
    expect(result.type).toBe(binding.TF_ATTR_TYPE);
    expect(result.value).toBe(binding.TF_INT32);
  });
  it('handles null', () => {
    expect(() => createTensorsTypeOpAttr('T', null)).toThrowError();
  });
  it('handles list of null', () => {
    const inputs = [null, null] as tfc.Tensor[];
    expect(() => createTensorsTypeOpAttr('T', inputs)).toThrowError();
  });
});
