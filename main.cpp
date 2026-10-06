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
    double forward(const std::vector<double> & inputs){
        double sum = bias;
        for(int i = 0; i < inputs.size(); i++){
            sum += inputs[i] * weights[i];
        }
        return sum;
    }
};

int main(){
    Neuron neuron(2);
    std::vector<double> inputs = { 2.0, 3.0 };
    double output = neuron.forward(inputs);
    std::cout << output << '\n';
    return 0;
}