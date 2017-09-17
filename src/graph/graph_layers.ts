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

// tslint:disable-next-line:max-line-length
import {Initializer, VarianceScalingInitializer, ZerosInitializer} from '../initializers';

import {Graph, Tensor} from './graph';

/**
 * A layers sugar class around the graph that initializes variables
 * automatically for layers.
 */
export class GraphLayers {
  constructor(private g: Graph) {}

  dense(
      name: string, x: Tensor, units: number,
      activation: ((x: Tensor) => Tensor)|null = null, useBias = true,
      kernelInitializer: Initializer = new VarianceScalingInitializer(),
      biasInitializer: Initializer = new ZerosInitializer()) {
    const weights = this.g.variable(
        name + '-weights',
        kernelInitializer.initialize([x.shape[0], units], x.shape[0], units));

    let out = this.g.matmul(x, weights);

    if (useBias) {
      const bias = this.g.variable(
          name + '-bias',
          biasInitializer.initialize([units], x.shape[0], units));
      out = this.g.add(out, bias);
    }

    if (activation != null) {
      out = activation(out);
    }

    return out;
  }
}
