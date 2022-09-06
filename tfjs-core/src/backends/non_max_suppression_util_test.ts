/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as non_max_suppression_util from './non_max_suppression_util';

describe('non_max_suppression_util', () => {
  const insertionPoint = (i: number) => -(i + 1);

  describe('binarySearch', () => {
    const d = [
      -897123.9, -321434.58758, -1321.3124, -324, -9, -3, 0, 0, 0, 0.31255, 5,
      142.88888708, 334, 342, 453, 54254
    ];

    it('-897123.9 should be found at index 0', () => {
      const result = non_max_suppression_util.binarySearch(d, -897123.9);
      expect(result).toBe(0);
    });

    it(`54254 should be found at index ${d.length - 1}`, () => {
      const result = non_max_suppression_util.binarySearch(d, 54254);
      expect(result).toBe(d.length - 1);
    });

    it('-3 should be found at index 5', () => {
      const result = non_max_suppression_util.binarySearch(d, -3);
      expect(result).toBe(5);
    });

    it('0 should be found at index 6', () => {
      const result = non_max_suppression_util.binarySearch(d, 0);
      expect(result).toBe(6);
    });

    it('-900000 should have an insertion point of 0', () => {
      const result = non_max_suppression_util.binarySearch(d, -900000);
      expect(result).toBeLessThan(0);
      expect(insertionPoint(result)).toBe(0);
    });

    it(`54255 should have an insertion point of ${d.length}`, () => {
      const result = non_max_suppression_util.binarySearch(d, 54255);
      expect(result).toBeLessThan(0);
      expect(insertionPoint(result)).toEqual(d.length);
    });

    it('1.1 should have an insertion point of 10', () => {
      const result = non_max_suppression_util.binarySearch(d, 1.1);
      expect(result).toBeLessThan(0);
      expect(insertionPoint(result)).toEqual(10);
    });
  });

  describe('binarySearch with custom comparator', () => {
    const e = [
      54254,
      453,
      342,
      334,
      142.88888708,
      5,
      0.31255,
      0,
      0,
      0,
      -3,
      -9,
      -324,
      -1321.3124,
      -321434.58758,
      -897123.9,
    ];

    const revComparator = (a: number, b: number) => (b - a);

    it('54254 should be found at index 0', () => {
      const result =
          non_max_suppression_util.binarySearch(e, 54254, revComparator);
      expect(result).toBe(0);
    });

    it(`-897123.9 should be found at index ${e.length - 1}`, () => {
      const result =
          non_max_suppression_util.binarySearch(e, -897123.9, revComparator);
      expect(result).toBe(e.length - 1);
    });

    it('-3 should be found at index 10', () => {
      const result =
          non_max_suppression_util.binarySearch(e, -3, revComparator);
      expect(result).toBe(10);
    });

    it('0 should be found at index 7', () => {
      const result = non_max_suppression_util.binarySearch(e, 0, revComparator);
      expect(result).toBe(7);
    });

    it('54254.1 should have an insertion point of 0', () => {
      const result =
          non_max_suppression_util.binarySearch(e, 54254.1, revComparator);
      expect(result).toBeLessThan(0);
      expect(insertionPoint(result)).toBe(0);
    });

    it(`-897124 should have an insertion point of ${e.length}`, () => {
      const result =
          non_max_suppression_util.binarySearch(e, -897124, revComparator);
      expect(result).toBeLessThan(0);
      expect(insertionPoint(result)).toBe(e.length);
    });
  });

  describe(
      'binarySearch with custom comparator with single element array', () => {
        const g = [1];

        const revComparator = (a: number, b: number) => (b - a);

        it('1 should be found at index 0', () => {
          const result =
              non_max_suppression_util.binarySearch(g, 1, revComparator);
          expect(result).toBe(0);
        });

        it('2 should have an insertion point of 0', () => {
          const result =
              non_max_suppression_util.binarySearch(g, 2, revComparator);
          expect(result).toBeLessThan(0);
          expect(insertionPoint(result)).toBe(0);
        });

        it('0 should have an insertion point of 1', () => {
          const result =
              non_max_suppression_util.binarySearch(g, 0, revComparator);
          expect(result).toBeLessThan(0);
          expect(insertionPoint(result)).toBe(1);
        });
      });

  describe('binarySearch test left-most duplicated element', () => {
    it('should find the index of the first 0', () => {
      const result = non_max_suppression_util.binarySearch([0, 0, 1], 0);
      expect(result).toBe(0);
    });

    it('should find the index of the first 1', () => {
      const result = non_max_suppression_util.binarySearch([0, 1, 1], 1);
      expect(result).toBe(1);
    });
  });

  describe('binaryInsert', () => {
    it('inserts correctly', () => {
      const a: number[] = [];

      non_max_suppression_util.binaryInsert(a, 3);
      expect(a).toEqual([3]);

      non_max_suppression_util.binaryInsert(a, 3);
      expect(a).toEqual([3, 3]);

      non_max_suppression_util.binaryInsert(a, 1);
      expect(a).toEqual([1, 3, 3]);

      non_max_suppression_util.binaryInsert(a, 5);
      expect(a).toEqual([1, 3, 3, 5]);
    });
  });
});
