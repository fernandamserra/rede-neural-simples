import random
import math

#
# Shorthand:
#   "pd_" as a variable prefix means "partial derivative"
#   "d_" as a variable prefix means "derivative"
#   "_wrt_" is shorthand for "with respect to"
#   "w_ho" and "w_ih" are the index of weights from hidden to output layer neurons and input to hidden layer neurons respectively
#
# Comment references:
#
# [1] Wikipedia article on Backpropagation
#   http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
# [2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#   https://class.coursera.org/neuralnets-2012-001/lecture/39
# [3] The Back Propagation Algorithm
#   https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

class NeuralNetwork:
    # Taxa de aprendizado padronizada no tutorial
    LEARNING_RATE = 0.5
    
    # Cria uma nova rede
    """ Parâmetros: n de entradas, n de neurônios na hl, n de neurônios na saída, vetor com os pesos da hl, viés da hl, vetor com os pesos dos neurônios da saída e viés da saída """
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        #cria a variável com o número de entradas
        self.num_inputs = num_inputs
        
        #Cria uma nova camada escondida passando o número de neurônios e o viés
        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        #Cria a camada de saída passando o número de neurônios e o viés
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        #Inicializa os pesos das conexões entre a entrada e a camada escondida
        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        #inicializa os pesos das conexões entre a camada escondida e a saída
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # Cria os pesos das conexões entre a entrada e a camada escondida
    # Recebe um vetor com os pesos da camada escondida
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        #Contador pra percorrer o vetor de pesos, caso não seja vazio
        weight_num = 0
        # repetição para fazer o mesmo a todos os neurônios
        for h in range(len(self.hidden_layer.neurons)):
            # Repetição para fazer o mesmo a todas as entradas
            for i in range(self.num_inputs):
                # caso o vetor de pesos não tenha sido inicializado
                if not hidden_layer_weights:
                    # adicinoa pesos aleatórios para cada neurônio da camada escondida
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    # adiciona os pesos do vetor passado ao objeto da camada escondida
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                #incrementa contador
                weight_num += 1
    
    #inicializa os pesos das conexões entre a camada escondida e a saída
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        #Contador pra percorrer o vetor de pesos, caso não seja vazio
        weight_num = 0
        # repetição para fazer o mesmo a todos os neurônios
        for o in range(len(self.output_layer.neurons)):
            # repetição para fazer o mesmo a todos os neurônios da camada escondida
            for h in range(len(self.hidden_layer.neurons)):
                # caso o vetor de pesos não tenha sido inicializado
                if not output_layer_weights:
                    # adicinoa pesos aleatórios para cada neurônio da camada de saída
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    # adiciona os pesos do vetor passado ao objeto da camada de saída
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                #incrementa contador
                weight_num += 1

    #Imprime os dados da rede neural
    def inspect(self):
        print('------')
        #Imprime todas as entradas em um vetor
        # Exemplo:
        # >>> a = [8,9,2]
        # >>> print('* Inputs: {}'.format(a))
        # * Inputs: [8, 9, 2]
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        # Método que imprime os dados da camada escondida
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        # Método que imprime os dados da camada de saída
        self.output_layer.inspect()
        print('------')

    # Realiza a etapa de "ida"
    # Recebe um vetor co as entradas
    def feed_forward(self, inputs):
        # Cria o vetor que receberá a saída da camada escondida
        # Chama o método feed_foward da classe NuronLayer que gera o vetor saída da camada
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        # Repete o processo de feed foward para a camada de saída e retorna o vetor contendo a saída final de cada neurônio dessa última camada
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Realiza o treino com Online Learning, ou seja, atualiza o peso após cada caso de treinamento
    # Recebe um vetor para as entradas e outro para as saídas do treinamento
    def train(self, training_inputs, training_outputs):
        # Realiza o feed foward da rede com as entradas passadas por parâmetro
        self.feed_forward(training_inputs)

        # 1. CALCULA O QUANTO UMA MUDANÇA NOS PESOS DA CAMADA DE SAÍDA AFETA NA SAÍDA TOTAL
        # DADO PELA DERIVADA DO ERRO EM FUNÇÃO DO PESO, EM QUE SERÁ USADA REGRA DA CADEIA PARA OBTER O RESULTADO
        # Inicializa um vetor com a mesma quantidade de neurônios da camada de saída com todos os valores zerados
        # Exemplo:
        # >>> a = [0] * 5
        # >>> a
        # [0, 0, 0, 0, 0]
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        # Repete o processo para todos os neurônios da camada de saída
        for o in range(len(self.output_layer.neurons)):
            # CHAMA O MÉTODO QUE REALIZA A REGRA DA CADEIA E CALCULA O QUANTO CADA NEURÔNIO PESOU PARA A RESPOSTA FINAL (ERRO)
            # ∂E/∂zⱼ - DERIVADA DO ERRO EM RELAÇÃO AO PESO DO NEURÔNIO
            # O vetor guarda o quanto a mudança em cada peso afetará a saída total
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. CALCULA O QUANTO UMA MUDANÇA NOS PESOS DA CAMADA ESCONDIDA AFETA NA SAÍDA TOTAL
        # DADA PELA DERIVADA DO ERRO EM FUNÇÃO DO PESO, EM QUE SERÁ USADA REGRA DA CADEIA PARA OBTER O RESULTADO
        # Inicializa um vetor com a mesma quantidade de neurônios da camada escondida com todos os valores zerados
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        # Repete o processo para todos os neurônios da camada escondida
        for h in range(len(self.hidden_layer.neurons)):
            
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ - DERIVADA DO ERRO COM RESPEITO AO PESO DOS NEURÔNIOS
            # Cria um acumulador
            d_error_wrt_hidden_neuron_output = 0
            # Repete o processo para todos os neurônios da camada de saída
            for o in range(len(self.output_layer.neurons)):
                # acumulador recebe o quanto cada peso da camada de saída afeta a saída multiplicado pelo peso de cada respectivo neurônio
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
            
            # CALCULA A DERIVADA FINAL DOS PESOS DA CAMADA ESCONDIDA USANDO A REGRA DA CADEIA
            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. ATUALIZA OS PESOS DA CAMADA DE SAÍDA
        # Repete para todos os neurônios da camada de saída
        for o in range(len(self.output_layer.neurons)):
            # Repete para todas as redes ligadas ao neurônio do loop atual
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                
                # CALCULA O RESULTADO FINAL DA DERIVADA PARCIAL DO ERRO COM RESPEITO AOS RESPECTIVOS PESOS
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)
                
                # APLICA A TACA DE APRENDIZAGEM AOS PESOS PARA ATUALIZÁ-LOS
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                
                # CALCULA O RESULTADO FINAL DA DERIVADA PARCIAL DO ERRO COM RESPEITO AOS RESPECTIVOS PESOS
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)
                
                # APLICA A TACA DE APRENDIZAGEM AOS PESOS PARA ATUALIZÁ-LOS
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    # CALCULA O ERRO TOTAL, QUE É O SOMATÓRIO DE TODOS OS VALORES DE ERROS DOS PESOS
    # RECEBE UMA MATRIZ COM TODOS OS CASOS DE TREINAMENTO
    def calculate_total_error(self, training_sets):
        # ACUMULADOR
        total_error = 0
        # REPETE PARA TODOS OS CASOS DE ENTRADA
        for t in range(len(training_sets)):
            # ATRIBUI A RESPECTIVA LINHA DO SET DE TREINAMENTO AOS DE ENTRADA E SAÍDA
            training_inputs, training_outputs = training_sets[t]
            # REALIZA O FEED FOWARD NOS DE ENTRADA
            self.feed_forward(training_inputs)
            # PARA TODOS OS VALORES DE ENTRADA, REPETE:
            for o in range(len(training_outputs)):
                #ACUMULA A TAXA DE ERRO TOTAL COM BASE NA DE CADA PESO
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        # RETORNA O ERRO TOTAL
        return total_error

#Classe das camadas
class NeuronLayer:
    #construtor que recebe o número de neurônios e o viés da camada
    def __init__(self, num_neurons, bias):

        # Todos os neurônios de uma camada compartilham do mesmo viés
        self.bias = bias if bias else random.random()
        
        #inicializa um vetor para guardar os neurônios
        self.neurons = []
        #cria os num_neurons da camada
        for i in range(num_neurons):
            #adiciona um objeto da classe Neuron ao vetor de neurônios, todos com o mesmo viés e pesos vazios
            self.neurons.append(Neuron(self.bias))

            
    #Imprime os dados de uma camada
    def inspect(self):
        #Imprime o número de neurônios
        print('Neurons:', len(self.neurons))
        #Repete para cada neurônio
        for n in range(len(self.neurons)):
            #Imprime o índice do neurônio no vetor (Começa de 0)
            print(' Neuron', n)
            #Repete para cada peso relacionado ao neurônio, consequentemente, para cada conexão a ele
            for w in range(len(self.neurons[n].weights)):
                # Imprime todos os pesos das conexões do neurônio em questão
                print('  Weight:', self.neurons[n].weights[w])
            #Imprime o viés de cada neurônio da camada
            print('  Bias:', self.bias)
    
    # Método que calcula as saídas de cada neurônio da camada
    # Recebe as entradas como parâmetro
    def feed_forward(self, inputs):
        # Inicializa vetor para a saída dessa camada
        outputs = []
        # Repete o processo para todos os neurônios
        for neuron in self.neurons:
            # Para cada neurônio da camada chama o método que calcula sua respectiva saída para todas as entradas
            # Adiciona o resultado ao vetor de saída da camada
            outputs.append(neuron.calculate_output(inputs))
        # Retorna um vetor contendo a saída de todos os neurônios da camada
        return outputs

    # Retorna o vetor de saída da camada
    # Deve ser chamada depois de terem sido calculadas as saídas de cada neurônio
    def get_outputs(self):
        # Cria um vetor para receber as saídas
        outputs = []
        # Repete para cada neurônio da camada
        for neuron in self.neurons:
            # Adiciona a saída do neurônio ao vetor
            outputs.append(neuron.output)
        # Retorna o vetor preenchido com a saída de cada neurônio
        return outputs

#Classe dos neurônios
class Neuron:
    #Construtor
    def __init__(self, bias):
        #inicialização do viés com o passado
        self.bias = bias
        #inicializa o vetor de pesos
        self.weights = []
        
    # Calcula a saída do neurônio para cada entrada contida no vetor passado por parâmetro
    def calculate_output(self, inputs):
        # Inicializa o vetor de entrada com o parâmetro para ser usado pelo método calculate_total_net_input depois
        self.inputs = inputs
        # Output recebe a saída do neurônio após ter gerado a entrada total líquida e a compactado
        self.output = self.squash(self.calculate_total_net_input())
        # Retorna a saída do neurônio
        return self.output

    # Calcula a entrada total líquida do neurônio dadas todas as entradas juntas
    def calculate_total_net_input(self):
        # Cria acumulador
        total = 0
        # Repete para o número total de entradas
        for i in range(len(self.inputs)):
            # acumula a soma de todas as entradas multiplicadas por seus respectivos pesos
            total += self.inputs[i] * self.weights[i]
        # retorna o valor total de saída acrescido do viés
        return total + self.bias
    
    # Aplica a função logística para compactar/esmagar a saída do neurônio
    # Após ter a entrada total líquida esmagada, o resultado será a saída do neurônio
    # The result is sometimes referred to as 'net' [2] or 'net' [1]
    def squash(self, total_net_input):
        # 1 sobre 1 mais o número de euler (e) elevado à entrada total líquida multiplicada por -1
        return 1 / (1 + math.exp(-total_net_input))

    # USA A REGRA DA CADEIA PRA DEFINIR O QUANTO UMA MUDANÇA NO PESO DA REDE DESTE NEURÔNIO PESA NA SAÍDA TOTAL
    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    # CALCULATE_PD_ERROR_WRT_OUTPUT RETORNA A DERIVADA DO ERRO EM FUNÇÃO À SAÍDA DO NEURÔNIO
    # CALCULATE_PD_TOTAL_NET_INPUT_WRT_INPUT CALCULA A DERIVADA DA SAÍDA DO NEURÔNIO EM FUNÇÃO DA REDE
    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();

    # Calcula o erro de saída do neurônio com a Função de Erro Quadrática (Mean Square Error Method):
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2
    
    # CALCULA A DERIVADA PARCIAL DO ERRO COM RESPEITO À SAÍDA ATUAL - QUANTO O ERRO MUDA EM RELAÇÃO À SAÍDA DESSE NEURÔNIO?
    # ANALISA O QUANTO A SAÍDA DO NEURÔNIO PESOU NA RESPOSTA FINAL (ERRO)
    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # = -(target output - actual output)
    #
    # The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    # = actual output - target output
    #
    # Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    #
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    # = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    # DERIVO A SAÍDA DO NEURÔNIO (DADA PELA FUNÇÃO LOGÍSTICA DE ESMAGAMENTO) EM RELAÇÃO À ENTRADA LÍQUIDA TOTAL DELE
    # The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    # yⱼ = φ = 1 / (1 + e^(-zⱼ))
    # Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    #
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    # CALCULA A DERIVADA PARCIAL DA ENTRADA LÍQUIDA TOTAL EM RELAÇÃO AO PESO
    # APÓS DERIVAR, TEM-SE QUE O RESULTADO É O VALOR DE ENTRADA DO RESPECTIVO PESO
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    #
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]

###

# Blog post example:

nn = NeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
for i in range(10000):
    nn.train([0.05, 0.1], [0.01, 0.99])
    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
