#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[1024*2],
    result_t layer17_out[6]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer17_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1d_weight_t, 224>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1d_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<batch_normalization_scale_t, 16>(s4, "s4.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv1d_1_weight_t, 2560>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv1d_1_bias_t, 32>(b6, "b6.txt");
        nnet::load_weights_from_txt<batch_normalization_1_scale_t, 32>(s8, "s8.txt");
        nnet::load_weights_from_txt<batch_normalization_1_bias_t, 32>(b8, "b8.txt");
        nnet::load_weights_from_txt<conv1d_2_weight_t, 6144>(w10, "w10.txt");
        nnet::load_weights_from_txt<conv1d_2_bias_t, 64>(b10, "b10.txt");
        nnet::load_weights_from_txt<batch_normalization_2_scale_t, 64>(s12, "s12.txt");
        nnet::load_weights_from_txt<batch_normalization_2_bias_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<dense_weight_t, 2048>(w14, "w14.txt");
        nnet::load_weights_from_txt<dense_bias_t, 32>(b14, "b14.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 192>(w16, "w16.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 6>(b16, "b16.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv1d_result_t layer2_out[1024*16];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[1024*16];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    batch_normalization_result_t layer4_out[1024*16];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[256*16];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    conv1d_1_result_t layer6_out[256*32];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[256*32];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    batch_normalization_1_result_t layer8_out[256*32];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[64*32];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    conv1d_2_result_t layer10_out[64*64];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    layer11_t layer11_out[64*64];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0

    batch_normalization_2_result_t layer12_out[64*64];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    layer13_t layer13_out[64];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    dense_result_t layer14_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    layer15_t layer15_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    dense_1_result_t layer16_out[6];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    nnet::conv_1d_cl<input_t, conv1d_result_t, config2>(input_1, layer2_out, w2, b2); // conv1d

    nnet::relu<conv1d_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv1d_relu

    nnet::normalize<layer3_t, batch_normalization_result_t, config4>(layer3_out, layer4_out, s4, b4); // batch_normalization

    nnet::pooling1d_cl<batch_normalization_result_t, layer5_t, config5>(layer4_out, layer5_out); // max_pooling1d

    nnet::conv_1d_cl<layer5_t, conv1d_1_result_t, config6>(layer5_out, layer6_out, w6, b6); // conv1d_1

    nnet::relu<conv1d_1_result_t, layer7_t, relu_config7>(layer6_out, layer7_out); // conv1d_1_relu

    nnet::normalize<layer7_t, batch_normalization_1_result_t, config8>(layer7_out, layer8_out, s8, b8); // batch_normalization_1

    nnet::pooling1d_cl<batch_normalization_1_result_t, layer9_t, config9>(layer8_out, layer9_out); // max_pooling1d_1

    nnet::conv_1d_cl<layer9_t, conv1d_2_result_t, config10>(layer9_out, layer10_out, w10, b10); // conv1d_2

    nnet::relu<conv1d_2_result_t, layer11_t, relu_config11>(layer10_out, layer11_out); // conv1d_2_relu

    nnet::normalize<layer11_t, batch_normalization_2_result_t, config12>(layer11_out, layer12_out, s12, b12); // batch_normalization_2

    nnet::global_pooling1d_cl<batch_normalization_2_result_t, layer13_t, config13>(layer12_out, layer13_out); // global_average_pooling1d

    nnet::dense<layer13_t, dense_result_t, config14>(layer13_out, layer14_out, w14, b14); // dense

    nnet::relu<dense_result_t, layer15_t, relu_config15>(layer14_out, layer15_out); // dense_relu

    nnet::dense<layer15_t, dense_1_result_t, config16>(layer15_out, layer16_out, w16, b16); // dense_1

    nnet::softmax<dense_1_result_t, result_t, softmax_config17>(layer16_out, layer17_out); // dense_1_softmax

}

