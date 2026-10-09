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

        double lastSum;
        double lastOutput;

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

        double getWeight(int index) const {
            return weights[index];
        }

        /*void train(const std::vector<double>& inputs, double target, double learningRate) {
            double prediction = forward(inputs);
            double error = prediction - target;
            for(int i = 0; i < inputs.size(); i++){
                //derivative of sigmoid is sigmoid'(x) = prediction * (1 - prediction)
                weights[i] -= learningRate * error * inputs[i] * prediction * (1 - prediction);
            }
            bias -= learningRate * error * prediction * (1 - prediction);
        }*/

        double forward(const std::vector<double> & inputs){
            double sum = bias;
            for(int i = 0; i < inputs.size(); i++){
                sum += inputs[i] * weights[i];
            }
            lastSum = sum;
            lastOutput = sigmoid(sum);
            return lastOutput;
        }

        double calculateGradient(double target){

            double error = lastOutput - target;

            double sigmoidDerivative = lastOutput * (1 - lastOutput);

            return error * sigmoidDerivative;

        }

        void updateWeights(const std::vector<double>& inputs, double gradient, double learningRate){
            for(int i = 0; i < weights.size(); i++){
                weights[i] -= learningRate * gradient * inputs[i];
            }
            bias -= learningRate * gradient;
        }

        double calculateHiddenGradient(double outputGradient, double outputWeight){
            return lastOutput * (1.0 - lastOutput) * outputGradient * outputWeight;
        }
};

class Layer {

    private:
        std::vector<Neuron> neurons;
    
    public:
        Layer(int inputCount, int neuronCount){

            for(int i = 0; i < neuronCount; i++){
                neurons.emplace_back(inputCount);
            }

        }

        std::vector<double> forward(const std::vector<double>& inputs){
            std::vector<double> outputs;

            for(int i = 0; i < neurons.size(); i++){
                outputs.push_back(neurons[i].forward(inputs));
            }

            return outputs;
        }

        double getNeuronWeight(int index) const {
            return neurons[0].getWeight(index);
        }

        /*void train(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate){
            for(int i = 0; i  < neurons.size(); i++){

                neurons[i].train(inputs, targets[i], learningRate);

            }
        }*/

        double calculateGradient(double target){
            return neurons[0].calculateGradient(target);
        }

        void updateWeights(const std::vector<double>& inputs, double gradient, double learningRate){
            neurons[0].updateWeights(inputs, gradient, learningRate);
        }

        std::vector<double> calculateHiddenGradients(double outputGradient, const Layer& outputLayer){
            std::vector<double> gradients;
            for(int i = 0; i < neurons.size(); i++){
                double outputWeight = outputLayer.getNeuronWeight(i);
                double gradient = neurons[i].calculateHiddenGradient(outputGradient, outputWeight);
                gradients.push_back(gradient);
            }
            return gradients;
        }

        void updateHiddenWeights(const std::vector<double>& inputs, const std::vector<double>& gradients, double learningRate){
            for(int i = 0; i < neurons.size(); i++){
                neurons[i].updateWeights(inputs, gradients[i], learningRate);
            }
        }

};

int main(){
    /*Neuron neuron(2);
    std::vector<std::vector<double>> inputs = { {1.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0} };
    std::vector<double> targets = { 0.0, 0.0, 1.0, 1.0 };
    double learningRate = 0.01;
    
    //train
    for(int epoch = 0; epoch < 10000; epoch++){
        for(int i = 0; i < inputs.size(); i++){
            neuron.train(inputs[i], targets[i], learningRate);
        }
    }
    
    //predict
    for(int i = 0; i < inputs.size(); i++){
        double prediction = neuron.forward(inputs[i]);
        double result = classify(prediction);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] Prediction: " << prediction << " Result: " << result << " Target: " << targets[i] << std::endl;
    }*/

    Layer layer(2, 3);
    Layer outputLayer(3, 1);

    double learningRate = 0.01;

    std::vector<double> inputs = { 2.0, 3.0 };
    double target = 1.0;

    /*std::vector<double> targets = { 1.0, 0.0, 1.0};

    for(int epoch = 0; epoch < 10000; epoch++){
        layer.train(inputs, targets, learningRate);
    }*/

    /*for(int epoch = 0; epoch < 1000; epoch++){
        std::vector<double> hiddenOutputs = layer.forward(inputs);

        std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);
    

        double gradient = outputLayer.calculateGradient(target);

        std::cout << "Prediction: " << finalOutput[0] << " Gradient: " << gradient << '\n';

        outputLayer.updateWeights(hiddenOutputs, gradient, learningRate);
    }*/

    for(int epoch = 0; epoch < 10000; epoch++){
        std::vector<double> hiddenOutputs = layer.forward(inputs);

        std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);

        double outputGradient = outputLayer.calculateGradient(target);

        std::vector<double> hiddenGradients = layer.calculateHiddenGradients(outputGradient, outputLayer);

        layer.updateHiddenWeights(inputs, hiddenGradients, learningRate);

        outputLayer.updateWeights(hiddenOutputs, outputGradient, learningRate);
    }

    std::vector<double> hiddenOutputs = layer.forward(inputs);

    std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);

    std::cout << "Prediction: " << finalOutput[0] << '\n';

    std::cout << "Target: " << target << '\n';


    return 0;
}