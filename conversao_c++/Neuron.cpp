//
//  Neuron.cpp
//  Back-Propagation
//
//  Created by Fernanda Serra on 02/08/18.
//  Copyright © 2018 Fernanda Serra. All rights reserved.
//
#include <math.h>
#include "Neuron.hpp"
using namespace std;

Neuron::Neuron(double b)
{
    bias = b;
    weights = vector<double>();
}

double Neuron::calculate_output(vector<double> inputs)
{
    this->inputs = inputs;
    this->output = this->squash(this->calculate_total_net_input());
    return output;
}

double Neuron::calculate_total_net_input()
{
    double total = 0.0;
    for (int i=0; i<inputs.size();i++)
    {
        total += inputs[i] * weights[i];
    }
    return total + bias;
}

double Neuron::squash(double total_net_input)
{
    return 1 / (1 + pow(M_E,-total_net_input));
}

double Neuron::calculate_pd_error_wrt_total_net_input(double target_output)
{
    return calculate_pd_error_wrt_output(target_output) * calculate_pd_total_net_input_wrt_input();
}

double Neuron::calculate_error(double target_output)
{
    return 0.5 * pow((target_output - output),2);
}

double Neuron::calculate_pd_error_wrt_output(double target_output)
{
    return (output - target_output);
}

double Neuron::calculate_pd_total_net_input_wrt_input()
{
    return output * (1 - output);
}

double Neuron::calculate_pd_total_net_input_wrt_weight(int index)
{
    return this->inputs[index];
}
