#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<37,17> conv1d_result_t;
typedef ap_fixed<16,6> conv1d_weight_t;
typedef ap_fixed<16,6> conv1d_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> conv1d_relu_table_t;
typedef ap_fixed<33,13> batch_normalization_result_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef ap_fixed<33,13> layer5_t;
typedef ap_fixed<57,27> conv1d_1_result_t;
typedef ap_fixed<16,6> conv1d_1_weight_t;
typedef ap_fixed<16,6> conv1d_1_bias_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> conv1d_1_relu_table_t;
typedef ap_fixed<33,13> batch_normalization_1_result_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;
typedef ap_fixed<33,13> layer9_t;
typedef ap_fixed<57,27> conv1d_2_result_t;
typedef ap_fixed<16,6> conv1d_2_weight_t;
typedef ap_fixed<16,6> conv1d_2_bias_t;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> conv1d_2_relu_table_t;
typedef ap_fixed<33,13> batch_normalization_2_result_t;
typedef ap_fixed<16,6> batch_normalization_2_scale_t;
typedef ap_fixed<16,6> batch_normalization_2_bias_t;
typedef ap_fixed<16,6> layer13_t;
typedef ap_fixed<39,19> dense_result_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer14_index;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<18,8> dense_relu_table_t;
typedef ap_fixed<38,18> dense_1_result_t;
typedef ap_fixed<16,6> dense_1_weight_t;
typedef ap_fixed<16,6> dense_1_bias_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> dense_1_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> dense_1_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> dense_1_softmax_inv_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> dense_1_softmax_inv_inp_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
