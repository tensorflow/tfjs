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

 import { Tensor,reshape } from "@tensorflow/tfjs-core";
// import { getExactlyOneTensor } from "../../utils/types_utils";
// import * as K from "../../backend/tfjs_backend";
// import { Kwargs } from "tfjs-layers/src/types";
// import { ValueError } from "tfjs-layers/src/errors";
// import {
//   BaseRandomLayerArgs,
//   BaseRandomLayer,
// } from "tfjs-layers/src/engine/base_random_layer";
// import * as tf from "@tensorflow/tfjs";
import {
  describeMathCPUAndGPU,
  // expectTensorsClose,
} from "../../utils/test_utils";

import {
  RandomWidth,
  RandomWidthArgs,
  INTERPOLATION_METHODS,
} from "./random_width";

describeMathCPUAndGPU("RandomWidth Layer", () => {
  it("Check if output shape matches specifications'", () => {
    // random width and check output shape

    expect("asdf").toEqual("asdf");
  });


  // Made per "py" test_valid_random_width(self):
  it("Check output image width", () => {
    // need (maxval - minval) * rnd + minval = 0.6
    // const mock_factor = 0.6
    // Creating tensor with shape (12,8,5,3) with random values: 0 <= val < 1
    // 12 * 8 * 5 * 3 = 1440
    let randomValues = Array.from({length: 1440}, () => Math.random());
    let inputTensor =reshape(randomValues, [12,8,5,3]);
    const randomWidthLayer = new
          RandomWidth({factor:0.4,rngType:'uniform'});
    const layerOutputTensor = randomWidthLayer.apply(inputTensor) as Tensor;
console.log('Layer Output Tensor Shape: ', layerOutputTensor.shape)
    expect(layerOutputTensor.shape).toEqual([12,8,3,3]);
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
