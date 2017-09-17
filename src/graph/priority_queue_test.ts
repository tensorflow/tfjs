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

import * as priority_queue from './priority_queue';
import {PriorityQueue} from './priority_queue';

describe('defaultCompare', () => {
  it('returns 0 if a === b', () => {
    expect(priority_queue.defaultCompare(123, 123)).toEqual(0);
  });

  it('returns 1 if a > b', () => {
    expect(priority_queue.defaultCompare(1000, 999)).toEqual(1);
  });

  it('returns -1 if a < b', () => {
    expect(priority_queue.defaultCompare(999, 1000)).toEqual(-1);
  });
});

describe('PriorityQueue', () => {
  let pq: PriorityQueue<number>;

  beforeEach(() => {
    pq = new PriorityQueue<number>(priority_queue.defaultCompare);
  });

  it('is empty by default', () => {
    expect(pq.empty()).toEqual(true);
  });

  it('isn\'t empty after enqueue call', () => {
    pq.enqueue(0);
    expect(pq.empty()).toEqual(false);
  });

  it('returns to empty after dequeueing only element', () => {
    pq.enqueue(0);
    pq.dequeue();
    expect(pq.empty()).toEqual(true);
  });

  it('returns to empty after dequeueing last element', () => {
    for (let i = 0; i < 10; ++i) {
      pq.enqueue(i);
    }
    for (let i = 0; i < 9; ++i) {
      pq.dequeue();
      expect(pq.empty()).toEqual(false);
    }
    pq.dequeue();
    expect(pq.empty()).toEqual(true);
  });

  it('dequeue throws when queue is empty', () => {
    expect(() => pq.dequeue())
        .toThrow(new Error('dequeue called on empty priority queue.'));
  });

  it('dequeues the only enqueued item', () => {
    pq.enqueue(1);
    expect(pq.dequeue()).toEqual(1);
  });

  it('dequeues the lowest-priority of 2 items', () => {
    pq.enqueue(1000);
    pq.enqueue(0);
    expect(pq.dequeue()).toEqual(0);
  });

  it('dequeues items in min-priority order', () => {
    pq.enqueue(5);
    pq.enqueue(8);
    pq.enqueue(2);
    pq.enqueue(9);
    pq.enqueue(3);
    pq.enqueue(7);
    pq.enqueue(4);
    pq.enqueue(0);
    pq.enqueue(6);
    pq.enqueue(1);
    const dequeued: number[] = [];
    while (!pq.empty()) {
      dequeued.push(pq.dequeue());
    }
    expect(dequeued).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('PriorityQueue index observer', () => {
  let pq: PriorityQueue<number>;
  let indices: {[value: number]: number};

  beforeEach(() => {
    pq = new PriorityQueue<number>(
        priority_queue.defaultCompare,
        (value: number, newIndex: number) => indices[value] = newIndex);
    indices = {};
  });

  it('notifies of new index when enqueuing', () => {
    pq.enqueue(0);
    expect(indices[0]).not.toBe(null);
  });

  it('puts first enqueued element at root of heap (index 0)', () => {
    pq.enqueue(0);
    expect(indices[0]).toEqual(0);
  });

  it('puts second greater element at left child of root (index 1)', () => {
    pq.enqueue(0);
    pq.enqueue(1);
    expect(indices[0]).toEqual(0);
    expect(indices[1]).toEqual(1);
  });

  it('puts third greater element at right child of root (index 2)', () => {
    pq.enqueue(0);
    pq.enqueue(1);
    pq.enqueue(2);
    expect(indices[0]).toEqual(0);
    expect(indices[1]).toEqual(1);
    expect(indices[2]).toEqual(2);
  });

  it('swaps root with new min enqueued element', () => {
    pq.enqueue(1000);
    pq.enqueue(0);
    expect(indices[1000]).toEqual(1);
    expect(indices[0]).toEqual(0);
  });
});

class TestEntry {
  constructor(public id: number, public priority: number) {}
}

describe('PriorityQueue.update', () => {
  let pq: PriorityQueue<TestEntry>;
  let indices: {[id: number]: number};

  beforeEach(() => {
    pq = new PriorityQueue<TestEntry>(
        (a: TestEntry, b: TestEntry) =>
            priority_queue.defaultCompare(a.priority, b.priority),
        (entry: TestEntry, newIndex: number) => indices[entry.id] = newIndex);
    indices = {};
  });

  it('no longer dequeues original min element after priority change', () => {
    const e0 = new TestEntry(0, 10);
    const e1 = new TestEntry(1, 100);
    pq.enqueue(e0);
    pq.enqueue(e1);
    e0.priority = 101;
    pq.update(e0, 0);
    expect(pq.dequeue()).toBe(e1);
    expect(pq.dequeue()).toBe(e0);
  });

  it('doesn\'t change index when priority doesn\'t change', () => {
    const e = new TestEntry(0, 0);
    pq.enqueue(e);
    expect(indices[0]).toEqual(0);
    pq.update(e, 0);
    expect(indices[0]).toEqual(0);
  });

  it('doesn\'t change index when priority doesn\'t trigger sift', () => {
    const e = new TestEntry(0, 0);
    pq.enqueue(e);
    expect(indices[0]).toEqual(0);
    e.priority = 1234;
    pq.update(e, 0);
    expect(indices[0]).toEqual(0);
  });

  it('changes index when priority change triggers sift', () => {
    const e = new TestEntry(0, 10);
    pq.enqueue(e);
    pq.enqueue(new TestEntry(1, 100));
    e.priority = 1000;
    pq.update(e, 0);
    expect(indices[0]).toEqual(1);
  });
});
