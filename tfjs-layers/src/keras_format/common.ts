/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

// TODO(huan): add layer-specific input shape types (see: https://github.com/tensorflow/tfjs-layers/pull/492)
/** @docalias (null | number)[] */
export type Shape = Array<null | number>;

// The tfjs-core version of DataType must stay synced with this.
export type DataType = 'float32'|'int32'|'bool'|'complex64'|'string';

// TODO(soergel): Move the CamelCase versions back out of keras_format
// e.g. to src/common.ts.  Maybe even duplicate *all* of these to be pedantic?
/** @docinline */
export type DataFormat = 'channelsFirst'|'channelsLast';
export const VALID_DATA_FORMAT_VALUES = ['channelsFirst', 'channelsLast'];

export type InterpolationFormat = 'nearest'|'bilinear';
export const VALID_INTERPOLATION_FORMAT_VALUES = ['nearest', 'bilinear'];
// These constants have a snake vs. camel distinction.
export type DataFormatSerialization = 'channels_first'|'channels_last';

/** @docinline */
export type PaddingMode = 'valid'|'same'|'causal';
export const VALID_PADDING_MODE_VALUES = ['valid', 'same', 'causal'];

/** @docinline */
export type PoolMode = 'max'|'avg';
export const VALID_POOL_MODE_VALUES = ['max', 'avg'];

/** @docinline */
export type BidirectionalMergeMode = 'sum'|'mul'|'concat'|'ave';
export const VALID_BIDIRECTIONAL_MERGE_MODES = ['sum', 'mul', 'concat', 'ave'];

/** @docinline */
export type SampleWeightMode = 'temporal';
export const VALID_SAMPLE_WEIGHT_MODES = ['temporal'];
