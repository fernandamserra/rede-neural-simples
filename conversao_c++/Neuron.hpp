//
//  Neuron.hpp
//  Back-Propagation
//
//  Created by Fernanda Serra on 02/08/18.
//  Copyright © 2018 Fernanda Serra. All rights reserved.
//

#ifndef Neuron_hpp
#define Neuron_hpp

#include <stdio.h>
#include <vector>
using namespace std;

class Neuron
{
    public:
        double bias, output;
        vector<double> weights;
        vector<double> inputs;
        Neuron(double bias);
        double calculate_output(vector<double> inputs);
        double squash(double total_net_input);
        double calculate_total_net_input();
        double calculate_error(double target_output);
        double calculate_pd_error_wrt_total_net_input(double target_output);
        double calculate_pd_error_wrt_output(double target_output);
        double calculate_pd_total_net_input_wrt_input();
        double calculate_pd_total_net_input_wrt_weight(int index);
};

#endif /* Neuron_hpp */
