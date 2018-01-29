/* Copyright 2017 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as dl from 'deeplearn';

import {Cache} from './ModelCache';

const NUM_LAYERS = 4;
const IMAGE_SIZE = 64;

export class FontModel {
  metaData = 'A';
  dimensions = 40;
  range = 0.4;
  charIdMap: {[id: string]: number};
  private variables: {[varName: string]: dl.NDArray};
  private math: dl.NDArrayMath;
  private inferCache = new Cache(this, this.infer);
  private numberOfValidChars = 62;
  private multiplierScalar = dl.Scalar.new(255);

  constructor() {
    // Set up character ID mapping.
    this.charIdMap = {};
    for (let i = 65; i < 91; i++) {
      this.charIdMap[String.fromCharCode(i)] = i - 65;
    }
    for (let i = 97; i < 123; i++) {
      this.charIdMap[String.fromCharCode(i)] = i - 97 + 26;
    }
    for (let i = 48; i < 58; i++) {
      this.charIdMap[String.fromCharCode(i)] = i - 48 + 52;
    }
  }

  load(cb: () => void) {
    const checkpointLoader = new dl.CheckpointLoader(
        'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/fonts/');
    checkpointLoader.getAllVariables().then(vars => {
      this.variables = vars;
      cb();
    });
  }

  get(id: number, args: Array<{}>, priority: number) {
    args.push(this.metaData);
    return new Promise((resolve, reject) => {
      args.push(() => resolve());
      this.inferCache.get(id, args);
    });
  }

  init() {
    this.math = dl.ENV.math;
  }

  infer(args: Array<{}>) {
    const embedding = args[0] as dl.NDArray;
    const ctx = args[1] as CanvasRenderingContext2D;
    const char = args[2] as string;
    const cb = args[3] as () => void;

    const charId = this.charIdMap[char.charAt(0)];
    if (charId == null) {
      throw (new Error('Invalid character id'));
    }

    const adjusted = this.math.scope(keep => {
      const idx = dl.Array1D.new([charId]);
      const onehotVector = dl.oneHot(idx, this.numberOfValidChars).as1D();

      const axis = 0;
      const inputData = embedding.as1D().concat(onehotVector, axis);

      let lastOutput = inputData;

      for (let i = 0; i < NUM_LAYERS; i++) {
        const weights =
            this.variables[`Stack/fully_connected_${i + 1}/weights`] as
            dl.Array2D;
        const biases = this.variables[`Stack/fully_connected_${i + 1}/biases`];

        lastOutput = lastOutput.as2D(-1, weights.shape[0])
                         .matMul(weights)
                         .add(biases)
                         .relu() as dl.Array1D;
      }

      const finalWeights =
          this.variables['fully_connected/weights'] as dl.Array2D;
      const finalBiases =
          this.variables['fully_connected/biases'] as dl.Array2D;

      const finalOutput = lastOutput.as2D(-1, finalWeights.shape[0])
                              .matMul(finalWeights)
                              .add(finalBiases)
                              .sigmoid();

      // Convert the inferred tensor to the proper scaling then draw it.
      return this.multiplierScalar.sub(this.multiplierScalar.mul(finalOutput));
    });

    const d = adjusted.as3D(IMAGE_SIZE, IMAGE_SIZE, 1);

    d.data().then(values => {
      const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);

      let pixelOffset = 0;
      for (let i = 0; i < values.length; i++) {
        const value = values[i];
        imageData.data[pixelOffset++] = value;
        imageData.data[pixelOffset++] = value;
        imageData.data[pixelOffset++] = value;
        imageData.data[pixelOffset++] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
      d.dispose();
      cb();
    });
  }
}
