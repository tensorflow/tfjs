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

export class GrowingRingBuffer<T> extends RingBuffer<T> {
  private static INITIAL_CAPACITY = 32;

  /**
   * Constructs a `GrowingRingBuffer`.
   */
  constructor() {
    super(GrowingRingBuffer.INITIAL_CAPACITY);
  }

  isFull() {
    return false;
  }

  push(value: T) {
    if (super.isFull()) {
      this.expand();
    }
    super.push(value);
  }

  unshift(value: T) {
    if (super.isFull()) {
      this.expand();
    }
    super.unshift(value);
  }

  /**
   * Doubles the capacity of the buffer.
   */
  private expand() {
    const newCapacity = this.capacity * 2;
    const newData = new Array<T>(newCapacity);
    const len = this.length();

    // Rotate the buffer to start at index 0 again, since we can't just
    // allocate more space at the end.
    for (let i = 0; i < len; i++) {
      newData[i] = this.get(this.wrap(this.begin + i));
    }

    this.data = newData;
    this.capacity = newCapacity;
    this.doubledCapacity = 2 * this.capacity;
    this.begin = 0;
    this.end = len;
  }
}
