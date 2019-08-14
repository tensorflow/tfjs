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
 *
 * =============================================================================
 */

import {RingBuffer} from './ring_buffer';

describe('RingBuffer', () => {
  it('Works as a stack (LIFO)', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.push(i);
    }
    expect(ring.length()).toEqual(10);

    const result: number[] = [];
    for (let i = 0; i < 10; i++) {
      result[i] = ring.pop();
    }
    expect(result).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    expect(ring.length()).toEqual(0);
  });

  it('Works as a queue (FIFO)', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.push(i);
    }
    expect(ring.length()).toEqual(10);

    const result: number[] = [];
    for (let i = 0; i < 7; i++) {
      result[i] = ring.shift();
    }
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(ring.length()).toEqual(3);

    // test wrapping
    for (let i = 10; i < 15; i++) {
      ring.push(i);
    }
    expect(ring.length()).toEqual(8);

    const result2: number[] = [];
    for (let i = 0; i < 8; i++) {
      result2[i] = ring.shift();
    }
    expect(result2).toEqual([7, 8, 9, 10, 11, 12, 13, 14]);
    expect(ring.length()).toEqual(0);
  });

  it('Works as a reverse stack (LIFO)', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.unshift(i);
    }
    expect(ring.length()).toEqual(10);

    const result: number[] = [];
    for (let i = 0; i < 10; i++) {
      result[i] = ring.shift();
    }
    expect(result).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    expect(ring.length()).toEqual(0);
  });

  it('Works as a reverse queue (FIFO)', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.unshift(i);
    }
    expect(ring.length()).toEqual(10);

    const result: number[] = [];
    for (let i = 0; i < 10; i++) {
      result[i] = ring.pop();
    }
    expect(result).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(ring.length()).toEqual(0);
  });

  it('Works as a shuffling queue', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.push(i);
    }
    expect(ring.length()).toEqual(10);

    expect(ring.shuffleExcise(3)).toEqual(3);
    expect(ring.shuffleExcise(6)).toEqual(6);

    const result: number[] = [];
    for (let i = 0; i < 8; i++) {
      result[i] = ring.shift();
    }
    // note how position 3 got the last element (9) and position 6 got the
    // next-to-last element(8).
    expect(result).toEqual([0, 1, 2, 9, 4, 5, 8, 7]);
    expect(ring.length()).toEqual(0);
  });

  it('Throws error on push over capacity', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.push(i);
    }
    expect(() => ring.push(10)).toThrowError(/full/);
  });

  it('Throws error on pop when empty', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    expect(() => ring.pop()).toThrowError(/empty/);
  });

  it('Throws error on unshift over capacity', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    for (let i = 0; i < 10; i++) {
      ring.unshift(i);
    }
    expect(() => ring.unshift(10)).toThrowError(/full/);
  });

  it('Throws error on shift when empty', () => {
    const ring = new RingBuffer<number>(10);
    expect(ring.length()).toEqual(0);

    expect(() => ring.shift()).toThrowError(/empty/);
  });
});
