#include<iostream>
#include<vector>
#include<random>
#include<cmath>

//define sigmoid(x) = 1/(1 + e^(-x))
double sigmoid(double x){
    return 1.0/(1.0 + std::exp(-x));
}

//classify
int classify(double prediction){
    if(prediction >= 0.5){
        return 1;
    } 
    return 0;
}

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
                //derivative of sigmoid is sigmoid'(x) = prediction * (1 - prediction)
                weights[i] -= learningRate * error * inputs[i] * prediction * (1 - prediction);
            }
            bias -= learningRate * error * prediction * (1 - prediction);
        }

    double forward(const std::vector<double> & inputs){
        double sum = bias;
        for(int i = 0; i < inputs.size(); i++){
            sum += inputs[i] * weights[i];
        }
        return sigmoid(sum);
    }
};

int main(){
    Neuron neuron(1);
    std::vector<double> inputs = { 1.0, 2.0, 3.0, 4.0 };
    std::vector<double> targets = { 0.0, 0.0, 1.0, 1.0 };
    double learningRate = 0.01;
    
    //train
    for(int epoch = 0; epoch < 10000; epoch++){
        for(int i = 0; i < inputs.size(); i++){
            std::vector<double> input = { inputs[i] };
            neuron.train(input, targets[i], learningRate);
        }
    }
    
    //predict
    for(int i = 0; i < inputs.size(); i++){
        std::vector<double> input = { inputs[i] };
        double prediction = neuron.forward(input);
        double result = classify(prediction);
        std::cout << "Input: " << inputs[i] << " Prediction: " << prediction << " Result: " << result << " Target: " << targets[i] << std::endl;
    }


    return 0;
}