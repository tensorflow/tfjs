/**
 * @license
 * Copyright 2023 Google LLC.
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

/**
 *  Cached MHA layer based on `MultiHeadAttention`.
 */

/* Original source: keras_nlp/layers/modeling/cached_multi_head_attention.py */
import { Tensor, Tensor1D, Tensor2D, serialization } from '@tensorflow/tfjs-core';

import { MultiHeadAttention } from '../multihead_attention';
import { NotImplementedError } from '../../../errors';

export declare interface CachedMultiHeadAttentionOptions {
  /**
   * Query `Tensor` of shape `(B, T, dim)`.
   */

  /**
   * Value `Tensor` of shape `(B, S*, dim)`. If `cache` is `null`, `S*`
   * must equal `S` and match the shape of `attentionMask`. If `cache` is
   * not `null`, `S*` can be any length less than `S`, and the computed
   * value will be spliced into `cache` at `cacheUpdateIndex`.
   */
  value: Tensor,

  /**
   * Key `Tensor` of shape `(B, S*, dim)`.  If `cache` is `null`, `S*` must
   * equal `S` and match the shape of `attentionMask`. If `cache` is not `null`,
   * `S*` can be any length less than `S`, and the computed value will be
   * spliced into `cache` at `cacheUpdateIndex`.
   */
  key?: Tensor,

  /**
   * A boolean mask of shape `(B, T, S)`. `attentionMask` prevents
   * attention to certain positions. The boolean mask specifies which
   * query elements can attend to which key elements, 1 indicates
   * attention and 0 indicates no attention. Broadcasting can happen for
   * the missing batch dimensions and the head dimension.
   */
  attentionMask?: Tensor,

  /**
   * A dense float Tensor. The key/value cache, of shape
   * `[B, 2, S, num_heads, key_dims]`, where `S` must agree with the
   * `attentionMask` shape. This argument is intended for use during
   * generation to avoid recomputing intermediate state.
   */
  cache?: Tensor

  /**
   * Integer or Integer `Tensor`. The index at which to update `cache`
   * (usually the index of the current token being processed when running
   * generation). If `cacheUpdateIndex=null` while `cache` is set, the cache
   * will not be updated.
   */
  cacheUpdateIndex?: number|Tensor
}

/**
 * MultiHeadAttention layer with cache support.
 *
 * This layer is suitable for use in autoregressive decoding. It can be use
 * to cache decoder self-attention and cross-attention. The forward pass
 * can happen in one of three modes:
 * - No cache, same as regular multi-head attention.
 * - Static cache (`cache_update_index` is None). In this case, the
 *     cached key/value projections will be used and the input values will
 *     be ignored.
 * - Updated cache (`cache_update_index` is not None). In this case, new
 *     key/value projections are computed using the input, and spliced into
 *     the cache at the specified index.
 *
 * Note that caching is useful only during inference and should not be used
 * during training.
 *
 * We use the notation `B`, `T`, `S` below, where `B` is the batch dimension,
 * `T` is the target sequence length, and `S` in the source sequence length.
 * Note that during generative decoding, `T` is usually 1 (you are
 * generating a target sequence of length one to predict the next token).
 *
 * Returns:
 *     An `(attentionOutput, cache)` tuple. `attentionOutput` is the result
 *     of the computation, of shape `(B, T, dim)`, where `T` is for target
 *     sequence shapes and `dim` is the query input last dimension if
 *     `outputShape` is `null`. Otherwise, the multi-head outputs are
 *     projected to the shape specified by `outputShape`. `cache` is the
 *     updated cache.
 */
export class CachedMultiHeadAttention extends MultiHeadAttention {

  override call(
    query: Tensor, kwargs: CachedMultiHeadAttentionOptions
  ): Tensor|Tensor2D {
    return this.callAndReturnCache(query, kwargs)[0];
  }

  /**
   * Exactly like `call` except also returns the updated cache.
   */
  callAndReturnCache(
    query: Tensor, kwargs: CachedMultiHeadAttentionOptions
  ): [Tensor1D|Tensor2D, Tensor1D|Tensor2D] {
    throw new NotImplementedError(`Not implemented yet.`);
  }
}
serialization.registerClass(CachedMultiHeadAttention);
