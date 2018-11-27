/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import {Tensor} from '@tensorflow/tfjs-core';

import * as losses from './losses';
import * as metrics from './metrics';

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'binaryAccuracy'
 * }
 */
export function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.binaryAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'binaryCrossentropy'
 * }
 */
export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.binaryCrossentropy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'sparseCategoricalAccuracy'
 * }
 */
export function sparseCategoricalAccuracy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.sparseCategoricalAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'categoricalAccuracy'
 * }
 */
export function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.categoricalAccuracy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'categoricalCrossentropy'
 * }
 */
export function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.categoricalCrossentropy(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'precision'
 * }
 */
export function precision(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.precision(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'recall'
 * }
 */
export function recall(yTrue: Tensor, yPred: Tensor): Tensor {
  return metrics.recall(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'cosineProximity'
 * }
 */
export function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.cosineProximity(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanAbsoluteError'
 * }
 */
export function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsoluteError(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanAbsolutePercentageError'
 * }
 */
export function meanAbsolutePercentageError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

export function MAPE(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

export function mape(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanAbsolutePercentageError(yTrue, yPred);
}

/**
 * @doc {
 *   heading: 'Metrics',
 *   namespace: 'metrics',
 *   useDocsFrom: 'meanSquaredError'
 * }
 */
export function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}

export function MSE(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}

export function mse(yTrue: Tensor, yPred: Tensor): Tensor {
  return losses.meanSquaredError(yTrue, yPred);
}
