//
//  NeuronLayer.cpp
//  Back-Propagation
//
//  Created by Fernanda Serra on 02/08/18.
//  Copyright © 2018 Fernanda Serra. All rights reserved.
//

#include "NeuronLayer.hpp"
#include <stdlib.h>
#include <time.h>
#include <iostream>

NeuronLayer::NeuronLayer()
{
    
}

void NeuronLayer::criaNeuronios(int num_neurons)
{
    neurons = std::vector<Neuron>();
    for (int i = 0; i < num_neurons; i++)
    {
        neurons.push_back(Neuron(this->bias));
    }
}

void NeuronLayer::setBias(double bias)
{
    // NÃO PREPARADO PARA CRIAR VIÉS ALEATÓRIO
    this->bias = bias;
}

NeuronLayer::NeuronLayer(int num_neurons, double bias)
{
    // CHECAR A COISA DO NONE, QUE É QUANDO NAO É PASSADO UM VIÉS
    if (!bias) 
    {
        srand((unsigned int)time(NULL));
        this->bias = (double) ((rand()%10)/10);
    } else 
    {
        this->bias = bias;
    }
    neurons = std::vector<Neuron>();
    for (int i = 0; i < num_neurons; i++)
    {
        neurons.push_back(Neuron(this->bias));
    }
}

void NeuronLayer::inspect(){
    std::cout << "Neurons: " << neurons.size() << "\n";
    for (int n = 0; n < neurons.size(); n++){
        std::cout << "Neuron " << n << "\n";
        for (int w = 0; w < neurons[n].weights.size(); w++) {
            std::cout << " Weight: " << neurons[n].weights[w];
        }
        std::cout << "  Bias: " << bias << "\n";
    }
}

std::vector<double> NeuronLayer::feed_forward(std::vector<double> inputs){
    std::vector<double> outputs = std::vector<double>();
    int i=0;
    for (Neuron neuron : neurons) {
        neurons[i].output = neuron.calculate_output(inputs);
        neurons[i].inputs = inputs;
        outputs.push_back(neurons[i].output);
        i++;
    }
    return outputs;
}

std::vector<double> NeuronLayer::get_outputs(){
    std::vector<double> outputs = std::vector<double>();
    for (Neuron neuron : neurons){
        outputs.push_back(neuron.output);
    }
    return outputs;
}









