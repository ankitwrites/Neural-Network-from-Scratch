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
    double input = 4.0;
    double target = 1.0;
    double learningRate = 0.01;
    std::vector<double> inputs = { input };
    double prediction = neuron.forward(inputs);
    double error = prediction - target;
    std::cout << "--------Before Training--------" << std::endl;
    std::cout << "Prediction: " << prediction << '\n';
    std::cout << "Error: " << error << '\n';
    std::cout << "--------After Training---------" << std::endl;
    neuron.train(inputs, target, learningRate);
    double newPrediction = neuron.forward(inputs);
    double newError = newPrediction - target;
    std::cout << "Prediction: " << newPrediction << '\n';
    std::cout << "Error: " << newError << '\n';
    return 0;
}