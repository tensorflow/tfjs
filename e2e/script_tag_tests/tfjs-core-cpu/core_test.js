/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

describe('loaded packages', () => {
  it('should have core', () => {
    expect(tf.version_core).toBeTruthy();
  });

  it('should have cpu backend', () => {
    expect(tf.version_cpu).toBeTruthy();
  });
});

describe('ops', () => {
  beforeAll(async () => {
    await tf.setBackend('cpu');
  });

  it('should support basic math', () => {
    tf.tidy(() => {
      const a = tf.scalar(3);
      const b = tf.scalar(4);
      expect(tf.add(a, b).dataSync()[0]).toBe(7);
    });
  });

  it('should clone', () => {
    tf.tidy(() => {
      const a = tf.tensor([1, 2, 3, 4]);
      const b = tf.tensor([1, 2, 3, 4]).clone();
      expect(a.dataSync()).toEqual(b.dataSync());
    });
  });

  it('should reshape', () => {
    tf.tidy(() => {
      const a = tf.tensor([1, 2, 3, 4], [1, 4]);
      const b = tf.reshape(a, [2, 2]);
      expect(a.dataSync()).toEqual(b.dataSync());
    });
  });

  it('should cast', () => {
    tf.tidy(() => {
      const a = tf.tensor([1, 2, 3, 4], [1, 4], 'float32');
      const b = a.cast('int32');
      expect(Array.from(a.dataSync())).toEqual(Array.from(b.dataSync()));
    });
  });
});

describe('backends are registered', () => {
  it('should find cpu backend', () => {
    expect(tf.findBackend('cpu')).toBeTruthy();
  });

  it('should not find fake backend', () => {
    expect(tf.findBackend('fake')).toBeFalsy();
  });
});
