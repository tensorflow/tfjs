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

/**
 * Default comparison function for the priority queue.
 * @param a The first element to compare.
 * @param b The second element to compare.
 * @return "a > b" returns > 0. "a < b" returns < 0. "a === b" returns 0.
 */
export function defaultCompare<T>(a: T, b: T): number {
  if (a === b) {
    return 0;
  } else if (a < b) {
    return -1;
  } else {
    return 1;
  }
}

/**
 * A Comparator is a user-provided function that compares two T instances. The
 * convention for defaultCompare is expected to be followed to maintain the
 * binary min-heap integrity.
 * @param a The first element to compare.
 * @param b The second element to compare.
 */
export type Comparator<T> = (a: T, b: T) => number;

/**
 * IndexObserver is a user-provided callback that informs the caller when an
 * element in the priority queue's binary min-heap has been relocated.
 * @param t The element that was relocated.
 * @param newIndex The new location in the binary min-heap of the element.
 */
export type IndexObserver<T> = (t: T, newIndex: number) => void;

/**
 * A priority queue, implemented in terms of a binary min-heap. Lower priority
 * numbers are considered higher priority.
 * enqueue, dequeue, and update are all O(log N) with respect to the number of
 * elements in the queue.
 */
export class PriorityQueue<T> {
  private heap: T[] = [];

  /**
   * @param comparator A function that compares two queue elements.
   * @param indexObserver An optional callback raised when the priority queue
   * changes the order of elements in its min-heap. Useful for tracking the
   * positions of elements that need updating.
   */
  constructor(
      private comparator: Comparator<T>,
      private indexObserver?: IndexObserver<T>) {}

  /**
   * Add an element to the priority queue.
   * @param t The element to enqueue.
   */
  enqueue(t: T) {
    this.heap.push(t);
    this.onIndexChanged(t, this.heap.length - 1);
    this.siftUp(this.heap.length - 1);
  }

  /**
   * Remove an element from the priority queue.
   * @return The element in the priority queue with the highest priority
   * (lowest numeric priority value).
   */
  dequeue(): T {
    if (this.empty()) {
      throw new Error('dequeue called on empty priority queue.');
    }
    const t = this.heap[0];
    this.swap(0, this.heap.length - 1);
    this.heap.pop();
    this.siftDown(0);
    return t;
  }

  /**
   * Updates an element at the specified index. This can be a full element
   * replacement, or it can be an in-place update. The priority is assumed to be
   * changed, and the internal storage is updated. This function is only useful
   * if the storage index of the updated element is known; construct the
   * PriorityQueue with an IndexObserver to track element locations.
   * @param newT The new element to replace in the priority queue.
   * @param index The index to insert the new element into.
   */
  update(newT: T, index: number) {
    /* If the element is at the very end of the heap, no sifting is necessary,
     * it can be safely removed. */
    const last = (index === this.heap.length - 1);
    if (!last) {
      this.swap(index, this.heap.length - 1);
    }
    this.heap.pop();
    if (!last) {
      /* The element at 'index' has been removed, and replaced with whatever was
       * at the end of the heap. Since that element might have come from a
       * different subtree (and not be a direct descendant of the node at
       * 'index'), we might need to sift this new value up instead of down. Test
       * both directions, and sift to wherever the node needs to go.
       */
      if (this.siftUpIndex(index) !== -1) {
        this.siftUp(index);
      } else if (this.siftDownIndex(index) !== -1) {
        this.siftDown(index);
      }
    }
    this.enqueue(newT);
  }

  /**
   * Predicate for testing whether the PriorityQueue is empty.
   * @return True if the PriorityQueue is empty, otherwise False.
   */
  empty(): boolean {
    return this.heap.length === 0;
  }

  private onIndexChanged(t: T, newIndex: number) {
    if (this.indexObserver) {
      this.indexObserver(t, newIndex);
    }
  }

  /*
   * Standard zero-indexed binary heap array layout:
   *   Parent(N) = Floor((N - 1) / 2)
   *   LeftChild(N) = (N * 2) + 1
   *   RightChild(N) = (N * 2) + 2
   */

  private getParentIndex(index: number): number {
    if (index === 0) {
      return -1;
    }
    return Math.floor((index - 1) / 2);
  }

  private getLeftChildIndex(index: number): number {
    const candidate = index * 2 + 1;
    return candidate < this.heap.length ? candidate : -1;
  }

  private getRightChildIndex(index: number): number {
    const candidate = index * 2 + 2;
    return candidate < this.heap.length ? candidate : -1;
  }

  private siftUpIndex(index: number): number {
    const parentIndex = this.getParentIndex(index);
    if (parentIndex === -1) {
      return -1;
    }
    if (this.compare(parentIndex, index) > 0) {
      return parentIndex;
    }
    return -1;
  }

  private siftUp(index: number) {
    let siftIndex = this.siftUpIndex(index);
    while (siftIndex !== -1) {
      this.swap(index, siftIndex);
      index = siftIndex;
      siftIndex = this.siftUpIndex(index);
    }
  }

  private siftDownIndex(index: number): number {
    if (index >= this.heap.length) {
      return -1;
    }
    let largestChildIndex = index;
    const leftChildIndex = this.getLeftChildIndex(index);
    if ((leftChildIndex !== -1) &&
        (this.compare(leftChildIndex, largestChildIndex) < 0)) {
      largestChildIndex = leftChildIndex;
    }
    const rightChildIndex = this.getRightChildIndex(index);
    if ((rightChildIndex !== -1) &&
        (this.compare(rightChildIndex, largestChildIndex) < 0)) {
      largestChildIndex = rightChildIndex;
    }
    return (largestChildIndex === index) ? -1 : largestChildIndex;
  }

  private siftDown(index: number) {
    let siftIndex = this.siftDownIndex(index);
    while (siftIndex !== -1) {
      this.swap(index, siftIndex);
      index = siftIndex;
      siftIndex = this.siftDownIndex(index);
    }
  }

  private compare(aIndex: number, bIndex: number): number {
    return this.comparator(this.heap[aIndex], this.heap[bIndex]);
  }

  private swap(a: number, b: number) {
    const temp = this.heap[a];
    this.heap[a] = this.heap[b];
    this.heap[b] = temp;
    this.onIndexChanged(this.heap[a], a);
    this.onIndexChanged(this.heap[b], b);
  }
}
