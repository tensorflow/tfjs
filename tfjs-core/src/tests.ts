/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

///// DO NOT EDIT: This file is auto-generated by /scripts/enumerate-tests.js

import './browser_util_test';
import './buffer_test';
import './debug_mode_test';
import './device_util_test';
import './engine_test';
import './environment_test';
import './flags_test';
import './globals_test';
import './gradients_test';
import './io/browser_files_test';
import './io/http_test';
import './io/indexed_db_test';
import './io/io_utils_test';
import './io/local_storage_test';
import './io/model_management_test';
import './io/passthrough_test';
import './io/progress_test';
import './io/router_registry_test';
import './io/weights_loader_test';
import './jasmine_util_test';
import './kernel_registry_test';
import './ops/abs_test';
import './ops/acos_test';
import './ops/acosh_test';
import './ops/add_n_test';
import './ops/add_test';
import './ops/all_test';
import './ops/any_test';
import './ops/arg_max_test';
import './ops/arg_min_test';
import './ops/arithmetic_test';
import './ops/asin_test';
import './ops/asinh_test';
import './ops/atan_test';
import './ops/atanh_test';
import './ops/avg_pool_3d_test';
import './ops/avg_pool_test';
import './ops/axis_util_test';
import './ops/basic_lstm_cell_test';
import './ops/batch_to_space_nd_test';
import './ops/batchnorm_test';
import './ops/binary_ops_test';
import './ops/boolean_mask_test';
import './ops/broadcast_to_test';
import './ops/broadcast_util_test';
import './ops/ceil_test';
import './ops/clip_by_value_test';
import './ops/clone_test';
import './ops/compare_ops_test';
import './ops/complex_ops_test';
import './ops/concat_test';
import './ops/concat_util_test';
import './ops/confusion_matrix_test';
import './ops/conv1d_test';
import './ops/conv2d_depthwise_test';
import './ops/conv2d_separable_test';
import './ops/conv2d_test';
import './ops/conv2d_transpose_test';
import './ops/conv3d_test';
import './ops/conv3d_transpose_test';
import './ops/conv_util_test';
import './ops/cos_test';
import './ops/cosh_test';
import './ops/cumsum_test';
import './ops/depth_to_space_test';
import './ops/diag_test';
import './ops/dilation2d_test';
import './ops/dropout_test';
import './ops/dropout_util_test';
import './ops/elu_test';
import './ops/equal_test';
import './ops/erf_test';
import './ops/exp_test';
import './ops/expand_dims_test';
import './ops/expm1_test';
import './ops/eye_test';
import './ops/fill_test';
import './ops/floor_test';
import './ops/from_pixels_test';
import './ops/fused/fused_conv2d_test';
import './ops/fused/fused_depthwise_conv2d_test';
import './ops/fused/fused_mat_mul_test';
import './ops/gather_nd_test';
import './ops/gather_test';
import './ops/greater_equal_test';
import './ops/greater_test';
import './ops/ifft_test';
import './ops/image/crop_and_resize_test';
import './ops/image/flip_left_right_test';
import './ops/image/non_max_suppression_async_test';
import './ops/image/non_max_suppression_test';
import './ops/image/resize_bilinear_test';
import './ops/image/resize_nearest_neighbor_test';
import './ops/image/rotate_with_offset_test';
import './ops/in_top_k_test';
import './ops/is_finite_test';
import './ops/is_inf_test';
import './ops/is_nan_test';
import './ops/leaky_relu_test';
import './ops/less_equal_test';
import './ops/less_test';
import './ops/linalg/band_part_test';
import './ops/linalg/gram_schmidt_test';
import './ops/linalg/qr_test';
import './ops/linspace_test';
import './ops/local_response_normalization_test';
import './ops/log1p_test';
import './ops/log_sigmoid_test';
import './ops/log_softmax_test';
import './ops/log_sum_exp_test';
import './ops/log_test';
import './ops/logical_and_test';
import './ops/logical_not_test';
import './ops/logical_or_test';
import './ops/logical_xor_test';
import './ops/losses/absolute_difference_test';
import './ops/losses/compute_weighted_loss_test';
import './ops/losses/cosine_distance_test';
import './ops/losses/hinge_loss_test';
import './ops/losses/huber_loss_test';
import './ops/losses/log_loss_test';
import './ops/losses/mean_squared_error_test';
import './ops/losses/sigmoid_cross_entropy_test';
import './ops/losses/softmax_cross_entropy_test';
import './ops/mat_mul_test';
import './ops/max_pool_3d_test';
import './ops/max_pool_test';
import './ops/max_pool_with_argmax_test';
import './ops/max_test';
import './ops/mean_test';
import './ops/min_test';
import './ops/mirror_pad_test';
import './ops/moments_test';
import './ops/moving_average_test';
import './ops/multi_rnn_cell_test';
import './ops/multinomial_test';
import './ops/neg_test';
import './ops/norm_test';
import './ops/not_equal_test';
import './ops/one_hot_test';
import './ops/ones_like_test';
import './ops/ones_test';
import './ops/operation_test';
import './ops/pad_test';
import './ops/pool_test';
import './ops/prod_test';
import './ops/rand_test';
import './ops/random_gamma_test';
import './ops/random_normal_test';
import './ops/random_uniform_test';
import './ops/range_test';
import './ops/reciprocal_test';
import './ops/relu6_test';
import './ops/relu_test';
import './ops/reverse_1d_test';
import './ops/reverse_2d_test';
import './ops/reverse_3d_test';
import './ops/reverse_4d_test';
import './ops/reverse_test';
import './ops/round_test';
import './ops/rsqrt_test';
import './ops/scatter_nd_test';
import './ops/selu_test';
import './ops/setdiff1d_async_test';
import './ops/sigmoid_test';
import './ops/sign_test';
import './ops/signal/frame_test';
import './ops/signal/hamming_window_test';
import './ops/signal/hann_window_test';
import './ops/signal/stft_test';
import './ops/sin_test';
import './ops/sinh_test';
import './ops/slice1d_test';
import './ops/slice2d_test';
import './ops/slice3d_test';
import './ops/slice4d_test';
import './ops/slice_test';
import './ops/slice_util_test';
import './ops/softmax_test';
import './ops/softplus_test';
import './ops/space_to_batch_nd_test';
import './ops/sparse_to_dense_test';
import './ops/spectral/fft_test';
import './ops/spectral/irfft_test';
import './ops/spectral/rfft_test';
import './ops/split_test';
import './ops/sqrt_test';
import './ops/square_test';
import './ops/stack_test';
import './ops/step_test';
import './ops/strided_slice_test';
import './ops/sub_test';
import './ops/sum_test';
import './ops/tan_test';
import './ops/tanh_test';
import './ops/tile_test';
import './ops/to_pixels_test';
import './ops/topk_test';
import './ops/transpose_test';
import './ops/truncated_normal_test';
import './ops/unsorted_segment_sum_test';
import './ops/unstack_test';
import './ops/where_async_test';
import './ops/where_test';
import './ops/zeros_like_test';
import './ops/zeros_test';
import './optimizers/adadelta_optimizer_test';
import './optimizers/adagrad_optimizer_test';
import './optimizers/adam_optimizer_test';
import './optimizers/adamax_optimizer_test';
import './optimizers/momentum_optimizer_test';
import './optimizers/optimizer_test';
import './optimizers/rmsprop_optimizer_test';
import './optimizers/sgd_optimizer_test';
import './platforms/platform_browser_test';
import './platforms/platform_node_test';
import './profiler_test';
import './public/chained_ops/register_all_chained_ops_test';
import './serialization_test';
import './tape_test';
import './tensor_test';
import './tensor_util_test';
import './test_util_test';
import './types_test';
import './util_test';
import './variable_test';
import './version_test';
import './worker_node_test';
import './worker_test';
