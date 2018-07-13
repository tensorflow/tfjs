/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
// tslint:disable-next-line:max-line-length
import {Constant, ConstantConfig, GlorotNormal, GlorotUniform, HeNormal, Identity, IdentityConfig, Initializer, LeCunNormal, Ones, Orthogonal, OrthogonalConfig, RandomNormal, RandomNormalConfig, RandomUniform, RandomUniformConfig, SeedOnlyInitializerConfig, TruncatedNormal, TruncatedNormalConfig, VarianceScaling, VarianceScalingConfig, Zeros} from './initializers';

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Zeros'
 * }
 */
export function zeros(): Zeros {
  return new Zeros();
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Ones'
 * }
 */
export function ones(): Initializer {
  return new Ones();
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Constant',
 *   configParamIndices: [0]
 * }
 */
export function constant(config: ConstantConfig): Initializer {
  return new Constant(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'RandomUniform',
 *   configParamIndices: [0]
 * }
 */
export function randomUniform(config: RandomUniformConfig): Initializer {
  return new RandomUniform(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'RandomNormal',
 *   configParamIndices: [0]
 * }
 */
export function randomNormal(config: RandomNormalConfig): Initializer {
  return new RandomNormal(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'TruncatedNormal',
 *   configParamIndices: [0]
 * }
 */
export function truncatedNormal(config: TruncatedNormalConfig): Initializer {
  return new TruncatedNormal(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Identity',
 *   configParamIndices: [0]
 * }
 */
export function identity(config: IdentityConfig): Initializer {
  return new Identity(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'VarianceScaling',
 *   configParamIndices: [0]
 * }
 */
export function varianceScaling(config: VarianceScalingConfig): Initializer {
  return new VarianceScaling(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'GlorotUniform',
 *   configParamIndices: [0]
 * }
 */
export function glorotUniform(config: SeedOnlyInitializerConfig): Initializer {
  return new GlorotUniform(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'GlorotNormal',
 *   configParamIndices: [0]
 * }
 */
export function glorotNormal(config: SeedOnlyInitializerConfig): Initializer {
  return new GlorotNormal(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'HeNormal',
 *   configParamIndices: [0]
 * }
 */
export function heNormal(config: SeedOnlyInitializerConfig): Initializer {
  return new HeNormal(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'LeCunNormal',
 *   configParamIndices: [0]
 * }
 */
export function leCunNormal(config: SeedOnlyInitializerConfig): Initializer {
  return new LeCunNormal(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Orthogonal',
 *   configParamIndices: [0]
 * }
 */
export function orthogonal(config: OrthogonalConfig): Initializer {
  return new Orthogonal(config);
}
