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
import {Constant, ConstantArgs, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Identity, IdentityArgs, Initializer, LeCunNormal, LeCunUniform, Ones, Orthogonal, OrthogonalArgs, RandomNormal, RandomNormalArgs, RandomUniform, RandomUniformArgs, SeedOnlyInitializerArgs, TruncatedNormal, TruncatedNormalArgs, VarianceScaling, VarianceScalingArgs, Zeros} from './initializers';

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
 *   useDocsFrom: 'Constant'
 * }
 */
export function constant(args: ConstantArgs): Initializer {
  return new Constant(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'RandomUniform'
 * }
 */
export function randomUniform(args: RandomUniformArgs): Initializer {
  return new RandomUniform(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'RandomNormal'
 * }
 */
export function randomNormal(args: RandomNormalArgs): Initializer {
  return new RandomNormal(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'TruncatedNormal'
 * }
 */
export function truncatedNormal(args: TruncatedNormalArgs): Initializer {
  return new TruncatedNormal(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Identity'
 * }
 */
export function identity(args: IdentityArgs): Initializer {
  return new Identity(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'VarianceScaling'
 * }
 */
export function varianceScaling(config: VarianceScalingArgs): Initializer {
  return new VarianceScaling(config);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'GlorotUniform'
 * }
 */
export function glorotUniform(args: SeedOnlyInitializerArgs): Initializer {
  return new GlorotUniform(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'GlorotNormal'
 * }
 */
export function glorotNormal(args: SeedOnlyInitializerArgs): Initializer {
  return new GlorotNormal(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'HeNormal'
 * }
 */
export function heNormal(args: SeedOnlyInitializerArgs): Initializer {
  return new HeNormal(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'HeUniform'
 * }
 */
export function heUniform(args: SeedOnlyInitializerArgs): Initializer {
  return new HeUniform(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'LeCunNormal'
 * }
 */
export function leCunNormal(args: SeedOnlyInitializerArgs): Initializer {
  return new LeCunNormal(args);
}

/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'LeCunUniform'
 * }
 */
export function leCunUniform(args: SeedOnlyInitializerArgs): Initializer {
  return new LeCunUniform(args);
}


/**
 * @doc {
 *   heading: 'Initializers',
 *   namespace: 'initializers',
 *   useDocsFrom: 'Orthogonal'
 * }
 */
export function orthogonal(args: OrthogonalArgs): Initializer {
  return new Orthogonal(args);
}
