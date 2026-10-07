#include<iostream>
#include<vector>
#include<random>

class Neuron {
    private:
        std::vector<double> weights;
        double bias;
    public:
        Neuron(int inputCount){
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<double> distribution(-1.0, 1.0);
            for(int i = 0; i < inputCount; i++){
                weights.push_back(distribution(generator));
            }
            bias = distribution(generator);
        }

        void train(const std::vector<double>& inputs, double target, double learningRate) {
            double prediction = forward(inputs);
            double error = prediction - target;
            for(int i = 0; i < inputs.size(); i++){
                weights[i] -= learningRate * error * inputs[i];
            }
            bias -= learningRate * error;
        }

    double forward(const std::vector<double> & inputs){
        double sum = bias;
        for(int i = 0; i < inputs.size(); i++){
            sum += inputs[i] * weights[i];
        }
        return sum;
    }
};

int main(){
    Neuron neuron(1);
    std::vector<double> inputs = { 1.0, 2.0, 3.0, 4.0 };
    std::vector<double> targets = { 0.0, 0.0, 1.0, 1.0 };
    double learningRate = 0.01;
    
    //train
    for(int epoch = 0; epoch < 50; epoch++){
        for(int i = 0; i < inputs.size(); i++){
            std::vector<double> input = { inputs[i] };
            neuron.train(input, targets[i], learningRate);
        }
    }
    
    //predict
    for(int i = 0; i < inputs.size(); i++){
        std::vector<double> input = { inputs[i] };
        double prediction = neuron.forward(input);
        std::cout << "Input: " << inputs[i] << " Prediction: " << prediction << " Target: " << targets[i] << std::endl;
    }


    return 0;
}