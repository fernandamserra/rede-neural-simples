//
//  rede-neural.hpp
//  rede-neural-simples-cpp
//
//  Created by Fernanda Serra on 02/08/18.
//  Copyright © 2018 Fernanda Serra. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include "NeuronLayer.hpp"
class NeuralNetwork
{
    public :
        double learning_rate = 0.5;
        int num_inputs;
        NeuronLayer hidden_layer, output_layer;
        NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, std::vector<double> hidden_layer_weights, double hidden_layer_bias, std::vector<double> output_layer_weights, double output_layer_bias);
        void init_weights_from_inputs_to_hidden_layer_neurons(std::vector<double> hidden_layer_weights);
        void init_weights_from_hidden_layer_neurons_to_output_layer_neurons(std::vector<double> output_layer_weights);
        void inspect();
        std::vector<double> feed_forward(std::vector<double> inputs);
        void train(std::vector<double> training_inputs, std::vector<double> training_outputs);
        double calculate_total_error(std::vector<std::vector<double> > training_sets);
    
};

#endif /* NeuralNetwork_hpp */
