/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {SampleWeightMode} from './common';
import {BaseSerialization, PyJson} from './types';

/**
 * Because of the limitations in the current Keras spec, there is no clear
 * definition of what may or may not be the configuration of an optimizer.  Thus
 * this empty interface represents a stopgap in the Keras spec.
 *
 * See internal issue: b/121033602
 */
export type OptimizerConfig<C extends PyJson<Extract<keyof C, string>>> = C;
// See ./types.ts for an explanation of the PyJson type.

/**
 * Configuration of a Keras optimizer, containing both the type of the
 * optimizer and the configuration for the optimizer of that type.
 */
export type OptimizerSerialization<
    N extends string, C extends PyJson<Extract<keyof C, string>>> =
    BaseSerialization<N, C>;
// See ./types.ts for an explanation of the PyJson type.

/**
 * List of all known loss names, along with a string description.
 *
 * Representing this as a class allows both type-checking using the keys and
 * generating an appropriate options array for use in select fields.
 */
export class LossOptions {
  [key: string]: string;
  // tslint:disable:variable-name
  // TODO(soergel): consider whether these can be static, given what
  // convertLossOptions() below needs to do.
  public readonly mean_squared_error = 'Mean Squared Error';
  public readonly mean_absolute_error = 'Mean Absolute Error';
  public readonly mean_absolute_percentage_error =
      'Mean Absolute Percentage Error';
  public readonly mean_squared_logarithmic_error =
      'Mean Squared Logarithmic Error';
  public readonly squared_hinge = 'Squared Hinge';
  public readonly hinge = 'Hinge';
  public readonly categorical_hinge = 'Categorical Hinge';
  public readonly logcosh = 'Logcosh';
  public readonly categorical_crossentropy = 'Categorical Cross Entropy';
  public readonly sparse_categorical_crossentropy =
      'Sparse Categorical Cross Entropy';
  public readonly kullback_leibler_divergence = 'Kullback-Liebler Divergence';
  public readonly poisson = 'Poisson';
  public readonly cosine_proximity = 'Cosine Proximity';
  // tslint:enable:variable-name
}

/**
 * Render the keys and values from the LossOptions class as an array of objects.
 */
function convertLossOptions(): Array<{value: string, label: string}> {
  const options = new LossOptions();
  const result: Array<{value: string, label: string}> = [];
  for (const key in Object.keys(options)) {
    result.push({value: key, label: options[key]});
  }
  return result;
}

/**
 * An array of `{value, label}` pairs describing the valid losses.
 *
 * The `value` is the serializable string constant, and the `label` is a more
 * user-friendly description (e.g. for use in UIs).
 */
export const lossOptions = convertLossOptions();

/**
 * A type representing the strings that are valid loss names.
 */
export type LossKey = keyof LossOptions;

// TODO(soergel): flesh out known metrics options
export type MetricsKey = string;

/**
 * a type for valid values of the `loss_weights` field.
 */
export type LossWeights = number[]|{[key: string]: number};

/**
 * Configuration of the Keras trainer. This includes the configuration to the
 * optimizer, the loss, any metrics to be calculated, etc.
 */
export interface TrainingConfig {
  // tslint:disable-next-line:no-any
  optimizer_config: OptimizerSerialization<string, OptimizerConfig<any>>;
  loss: LossKey|LossKey[]|{[key: string]: LossKey};
  metrics?: MetricsKey[];
  weighted_metrics?: MetricsKey[];
  sample_weight_mode?: SampleWeightMode;
  loss_weights?: LossWeights;
}
