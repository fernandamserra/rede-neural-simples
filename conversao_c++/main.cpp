//
//  main.cpp
//  Back-Propagation
//
//  Created by Nabson Paiva on 02/08/18.
//  Copyright © 2018 Nabson Paiva. All rights reserved.
//

#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"
using namespace std;
int main(int argc, const char * argv[]) {
    vector<double> escondida (4,0);
    escondida[0] = (0.15);
    escondida[1] = (0.2);
    escondida[2] = (0.25);
    escondida[3] = (0.3);
    vector<double> saida;
    saida.push_back(0.4);
    saida.push_back(0.45);
    saida.push_back(0.5);
    saida.push_back(0.55);
    NeuralNetwork *nn = new NeuralNetwork(2, 2, 2, escondida ,0.35, saida, 0.6);
    vector<double> entrada;
    entrada.push_back(0.05);
    entrada.push_back(0.1);
    vector<double> esperado;
    esperado.push_back(0.01);
    esperado.push_back(0.99);
    vector<vector<double> > treino;
    treino.push_back(entrada);
    treino.push_back(esperado);
    for (int i = 0; i < 10000; i++) {
        nn->train(entrada, esperado);
        cout.precision(9);
        cout.setf(ios::fixed);
        cout << i << "  " << nn->calculate_total_error(treino) << "\n	";
    }
    return 0;
}
