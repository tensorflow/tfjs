/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original source: utils/generic_utils.py */

import {Tensor} from '@tensorflow/tfjs-core';
import {ValueError} from '../errors';
import {Shape} from '../keras_format/common';
// tslint:enable

/**
 * Determine whether the input is an Array of Shapes.
 */
export function isArrayOfShapes(x: Shape|Shape[]): boolean {
  return Array.isArray(x) && Array.isArray(x[0]);
}

/**
 * Special case of normalizing shapes to lists.
 *
 * @param x A shape or list of shapes to normalize into a list of Shapes.
 * @return A list of Shapes.
 */
export function normalizeShapeList(x: Shape|Shape[]): Shape[] {
  if (x.length === 0) {
    return [];
  }
  if (!Array.isArray(x[0])) {
    return [x] as Shape[];
  }
  return x as Shape[];
}

/**
 * Helper function to obtain exactly one Tensor.
 * @param xs: A single `tf.Tensor` or an `Array` of `tf.Tensor`s.
 * @return A single `tf.Tensor`. If `xs` is an `Array`, return the first one.
 * @throws ValueError: If `xs` is an `Array` and its length is not 1.
 */
export function getExactlyOneTensor(xs: Tensor|Tensor[]): Tensor {
  let x: Tensor;
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new ValueError(`Expected Tensor length to be 1; got ${xs.length}`);
    }
    x = xs[0];
  } else {
    x = xs;
  }
  return x;
}

/**
 * Helper function to obtain exactly on instance of Shape.
 *
 * @param shapes Input single `Shape` or Array of `Shape`s.
 * @returns If input is a single `Shape`, return it unchanged. If the input is
 *   an `Array` containing exactly one instance of `Shape`, return the instance.
 *   Otherwise, throw a `ValueError`.
 * @throws ValueError: If input is an `Array` of `Shape`s, and its length is not
 *   1.
 */
export function getExactlyOneShape(shapes: Shape|Shape[]): Shape {
  if (Array.isArray(shapes) && Array.isArray(shapes[0])) {
    if (shapes.length === 1) {
      shapes = shapes as Shape[];
      return shapes[0];
    } else {
      throw new ValueError(`Expected exactly 1 Shape; got ${shapes.length}`);
    }
  } else {
    return shapes as Shape;
  }
}
