/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Explicit error types.
 *
 * See the following link for more information about why the code includes
 * calls to setPrototypeOf:
 *
 * https://github.com/Microsoft/TypeScript-wiki/blob/master/Breaking-Changes.md#extending-built-ins-like-error-array-and-map-may-no-longer-work
 */
// tslint:enable

/**
 * Equivalent of Python's AttributeError.
 */
export class AttributeError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, AttributeError.prototype);
  }
}

/**
 * Equivalent of Python's RuntimeError.
 */
export class RuntimeError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, RuntimeError.prototype);
  }
}

/**
 * Equivalent of Python's ValueError.
 */
export class ValueError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, ValueError.prototype);
  }
}

/**
 * Equivalent of Python's NotImplementedError.
 */
export class NotImplementedError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, NotImplementedError.prototype);
  }
}

/**
 * Equivalent of Python's AssertionError.
 */
export class AssertionError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, AssertionError.prototype);
  }
}

/**
 * Equivalent of Python's IndexError.
 */
export class IndexError extends Error {
  constructor(message?: string) {
    super(message);
    // Set the prototype explicitly.
    Object.setPrototypeOf(this, IndexError.prototype);
  }
}
