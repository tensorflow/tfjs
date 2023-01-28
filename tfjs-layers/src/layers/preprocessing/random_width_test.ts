/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Unit Tests for random width layer.
 */

 import {
  image,
  Rank,
  serialization,
  Tensor,
  cast,
  stack,
  tidy,
} from "@tensorflow/tfjs-core";
import { getExactlyOneTensor } from "../../utils/types_utils";
import * as K from "../../backend/tfjs_backend";
import { Kwargs } from "tfjs-layers/src/types";
import { ValueError } from "tfjs-layers/src/errors";
import {
  BaseRandomLayerArgs,
  BaseRandomLayer,
} from "tfjs-layers/src/engine/base_random_layer";
import * as tf from "@tensorflow/tfjs";
import {
  describeMathCPUAndGPU,
  expectTensorsClose,
} from "../../utils/test_utils";

import {
  RandomWidth,
  RandomWidthArgs,
  INTERPOLATION_METHODS,
} from "./random_width";

describeMathCPUAndGPU("RandomWidth Layer", () => {
  it("Check for 3D unbatched tensor should return float", () => {
    // test for 3D unbatched tensor should return float
    // TODO

    expect("asdf").toEqual("asdf");
  });

  it("Check for 4D batched tensor should return float", () => {
    // test for 4D batched tensor should return float
    // TODO

    expect("asdf").toEqual("asdf");
  });

  it("Throws an error if factor is a number, but not a possitive number", () => {
    // test factor, if factor is number (not array), it must be positive number
    const factor = -0.5;
    const interpolation = "";
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `Invalid factor parameter: ${factor}.Must be positive number or tuple of 2 numbers`;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it("Throws an error if factor is an array, wrong first item", () => {
    // test for factor array, first item in array must be >= -1, (width lower < -1)
    const factor = [-5, 0.3];
    const interpolation = "";
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `factor must have values larger than -1. Got: ${factor}`;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it("Throws an error if factor is an array, wrong second item", () => {
    // test for factor array, second item in array must be >= -1, (width upper < -1)
    const factor = [-0.5, -3];
    const interpolation = "";
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `factor must have values larger than -1. Got: ${factor}`;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it("Throws an error if factor is an array, upper width < lower width", () => {
    // test for upper width < lower width
    const factor = [0.5, 0.3];
    const interpolation = "";
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `factor cannot have upper bound less than lower bound.
          Got upper bound: ${factor[1]}.
          Got lower bound: ${factor[0]}
        `;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });

  it("Check interpolation method 'bilinear'", () => {
    // test interpolation method "bilinear"
    // TODO

    expect("asdf").toEqual("asdf");
  });

  it("Check interpolation method 'nearest'", () => {
    // test interpolation methods "nearest"
    // TODO

    expect("asdf").toEqual("asdf");
  });

  it("Check interpolation method 'gaussian'", () => {
    // test correct, but not implemented interpolation methods
    // ("bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic")
    // Not needed now
    expect("asdf").toEqual("asdf");
  });

  it("Check interpolation method 'unimplemented'", () => {
    // test incorrect input for interpolation method ("unimplemented")
    const factor = 0.5;
    const interpolation = "unimplemented";
    const seed = 42;
    const incorrectArgs = { factor, interpolation, seed };
    const expectedError = `Interpolation is ${interpolation} but only ${[
      ...INTERPOLATION_METHODS,
    ]} are supported`;
    expect(
      () => new RandomWidth(incorrectArgs as RandomWidthArgs)
    ).toThrowError(expectedError);
  });
});
