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

import * as tf from './index';

describe('Union package', () => {
  it('has core ops', () => {
    expect(tf.matMul).not.toBeNull();
    expect(tf.tensor).not.toBeNull();
    expect(tf.scalar).not.toBeNull();
    expect(tf.square).not.toBeNull();
  });

  it('has layers', () => {
    expect(tf.sequential).not.toBeNull();
    expect(tf.model).not.toBeNull();
    expect(tf.layers.dense).not.toBeNull();
  });

  it('has converter', () => {
    expect(tf.GraphModel).not.toBeNull();
    expect(tf.loadGraphModel).not.toBeNull();
  });

  it('has data', () => {
    expect(tf.data.csv).not.toBeNull();
    expect(tf.data.zip).not.toBeNull();
  });

  it('version', () => {
    // tslint:disable-next-line:no-require-imports
    const expected = require('../package.json').version;
    expect(tf.version.tfjs).toBe(expected);
    expect(tf.version['tfjs-core']).not.toBeNull();
    expect(tf.version['tfjs-converter']).not.toBeNull();
    expect(tf.version['tfjs-data']).not.toBeNull();
    expect(tf.version['tfjs-layers']).not.toBeNull();
  });

  it('has cpu backend', () => {
    const backend = tf.findBackend('cpu');
    expect(backend).not.toBeNull();
  });

  it('has webgl backend', () => {
    const backend = tf.findBackend('webgl');
    expect(backend).not.toBeNull();
  });
});
