//
//  NeuralNetwork.cpp
//  Back-Propagation
//
//  Created by Fernanda Serra on 02/08/18.
//  Copyright © 2018 Fernanda Serra. All rights reserved.
//
#include "NeuralNetwork.hpp"
#include "NeuronLayer.hpp"
#include <stdio.h>
#include <stdlib.h>
#include<time.h>
using namespace std;
NeuralNetwork::NeuralNetwork(int num_inputs,
                               int num_hidden,
                               int num_outputs,
                               vector<double> hidden_layer_weights,
                               double hidden_layer_bias,
                               vector<double> output_layer_weights,
                               double output_layer_bias)
{
    this->num_inputs = num_inputs;
    this->hidden_layer = NeuronLayer();
    hidden_layer.setBias(hidden_layer_bias);
    hidden_layer.criaNeuronios(num_hidden);
    this->output_layer = NeuronLayer();
    output_layer.setBias(output_layer_bias);
    output_layer.criaNeuronios(num_outputs);
    this->init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights);
    this->init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights);
}



void NeuralNetwork :: init_weights_from_inputs_to_hidden_layer_neurons(vector<double> hidden_layer_weights)
{
    double weight_num = 0;
    for(int i = 0; i < (hidden_layer.neurons).size();i++)
    {
        for(int j = 0; j < num_inputs; j++)
        {
            if(hidden_layer_weights.empty())
            {
                hidden_layer.neurons[i].weights.push_back(rand());
            }else
            {
                hidden_layer.neurons[i].weights.push_back(hidden_layer_weights[weight_num]);
            }
            weight_num += 1;
        }
    }
}

void NeuralNetwork :: init_weights_from_hidden_layer_neurons_to_output_layer_neurons(std::vector<double> output_layer_weights) {
    double weight_num = 0;
    for(int i = 0; i < (output_layer.neurons).size();i++)
    {
        for(int j = 0; j < (hidden_layer.neurons).size(); j++)
        {
            if(output_layer_weights.empty())
            {
                output_layer.neurons[i].weights.push_back(rand());
            }else
            {
                output_layer.neurons[i].weights.push_back(output_layer_weights[weight_num]);
            }
            weight_num += 1;
        }
    }
}

void NeuralNetwork :: inspect()
{
    printf("------");
    printf("* Inputs: ");
    for (int i = 0; i < num_inputs; i++){
        printf("%lf ", hidden_layer.neurons[0].inputs[i]);
    }
    printf("------");
    printf("Hidden Layer");
    hidden_layer.inspect();
    printf("------");
    printf("* Output Layer");
    output_layer.inspect();
    printf("------");
}
std::vector<double> NeuralNetwork :: feed_forward(vector<double>inputs)
{
    vector<double>hidden_layer_outputs = hidden_layer.feed_forward(inputs);
    return output_layer.feed_forward(hidden_layer_outputs);
}

void NeuralNetwork :: train(vector<double> training_inputs, vector<double> training_outputs)
{
    feed_forward(training_inputs);
    vector<double> pd_errors_wrt_output_neuron_total_net_input (output_layer.neurons.size(),0);
    for(int i = 0 ; i < output_layer.neurons.size(); i++)
    {
        pd_errors_wrt_output_neuron_total_net_input[i] = output_layer.neurons[i].calculate_pd_error_wrt_total_net_input(training_outputs[i]);
    }
    vector<double> pd_errors_wrt_hidden_neuron_total_net_input (hidden_layer.neurons.size(),0);
    for(int i = 0; i < (hidden_layer.neurons).size(); i++)
    {
        double d_error_wrt_hidden_neuron_output = 0;
        for(int j = 0; j < (output_layer.neurons).size(); j++)
        {
            d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[j] * output_layer.neurons[j].weights[i];
        }
        pd_errors_wrt_hidden_neuron_total_net_input[i] = d_error_wrt_hidden_neuron_output * hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_input();
    }
    for(int i = 0; i < (output_layer.neurons).size(); i++)
    {
        for(int j = 0; j < (output_layer.neurons[i].weights).size(); j++)
        {
            double pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[i] * output_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j);
            output_layer.neurons[i].weights[j] -= learning_rate * pd_error_wrt_weight;
            
        }
    }
    for(int i = 0; i< (hidden_layer.neurons).size(); i++)
    {
        for(int j = 0; j < (hidden_layer.neurons[i].weights).size(); j++)
        {
            
            double pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[i] * hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j);
            hidden_layer.neurons[i].weights[j] -= learning_rate * pd_error_wrt_weight;
        }
    }
}

double NeuralNetwork::calculate_total_error(std::vector<std::vector<double> > training_sets)
{
    double total_error = 0;
    vector<double> training_inputs, training_outputs;
        training_inputs = training_sets[0];
        training_outputs = training_sets[1];
        feed_forward(training_inputs);
        for(int j = 0; j < (training_outputs).size(); j++)
        {
            total_error += output_layer.neurons[j].calculate_error(training_outputs[j]);
        }
    return total_error;
}

