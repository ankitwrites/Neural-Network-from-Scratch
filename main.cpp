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

//loss
double calculateLoss(double prediction, double target) {
    double error = prediction - target;
    return 0.5 * error * error;
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

void trainNetwork(Layer& layer, Layer& outputLayer, const std::vector<std::vector<double>>& trainingInputs, const std::vector<double>& targets, int epochs, double learningRate){
    for(int epoch = 0; epoch < epochs; epoch++){
        for(int i = 0; i < trainingInputs.size(); i++){


            std::vector<double> hiddenOutputs = layer.forward(trainingInputs[i]);

            std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);

            double outputGradient = outputLayer.calculateGradient(targets[i]);

            std::vector<double> hiddenGradients = layer.calculateHiddenGradients(outputGradient, outputLayer);

            layer.updateHiddenWeights(trainingInputs[i], hiddenGradients, learningRate);

            outputLayer.updateWeights(hiddenOutputs, outputGradient, learningRate);


            //loss after every 1,000 epochs
            /*if((epoch + 1) % 1000 == 0){
                double totalLoss = 0.0;
                for(int i = 0; i < trainingInputs.size(); i++){
                    std::vector<double> hiddenOutputs = layer.forward(trainingInputs[i]);

                    std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);

                    totalLoss += calculateLoss(finalOutput[0], targets[i]);
                }
                double averageLoss = totalLoss / trainingInputs.size();
                //std::cout << "epoch : " << epoch + 1 << " average Loss: " << averageLoss << "\n";
            }*/
        }
    }
}

double predict(Layer& layer, Layer& outputLayer, const std::vector<double>& inputs){
    std::vector<double> hiddenOutputs = layer.forward(inputs);

    std::vector<double> finalOutput = outputLayer.forward(hiddenOutputs);

    return finalOutput[0];
}

int main(){

    Layer layer(2, 3);
    Layer outputLayer(3, 1);

    double learningRate = 0.01;

    std::vector<std::vector<double>> trainingInputs = { {1.0, 1.0}, {1.0, 2.0}, {2.0, 3.0}, {3.0, 3.0} };
    std::vector<double> targets = { 0.0, 0.0, 1.0, 1.0 };

    std::vector<std::vector<double>> testInputs = { {1.0, 3.0}, {3.0, 2.0} };
    std::vector<double> testTargets = { 0.0, 1.0 };

    std::cout << "Training-------------------" << "\n";

    trainNetwork(layer, outputLayer, trainingInputs, targets, 10000, learningRate);

    int correct = 0;

    for(int i = 0; i < trainingInputs.size(); i++){
        double prediction = predict(layer, outputLayer, trainingInputs[i]);

        int predictedClass = classify(prediction);

        if(predictedClass == static_cast<int>(targets[i])){
            correct++;
        }

        std::cout << "Input: [" << trainingInputs[i][0] << ", " << trainingInputs[i][1] << "]" << " Prediction: " << prediction << " Predicted Class: " << predictedClass << " target: " << targets[i] << "\n";
    }

    double accuracy = 100.0 * correct / trainingInputs.size();
    std::cout << "Accuracy: " << accuracy << "%\n";

    // test dataset

    std::cout << "Test-------------------" << "\n";

    int testCorrect = 0;

    for(int i = 0; i < testInputs.size(); i++){
        double prediction = predict(layer, outputLayer, testInputs[i]);

        int predictedClass = classify(prediction);

        if(predictedClass == static_cast<int>(testTargets[i])){
            testCorrect++;
        }

        std::cout << "Input: [" << testInputs[i][0] << ", " << testInputs[i][1] << "]" << " Prediction: " << prediction << " Predicted Class: " << predictedClass << " target: " << testTargets[i] << "\n";

    }

    double testAccuracy = 100.0 * testCorrect / testInputs.size();
    std::cout << "Accuracy: " << testAccuracy << "%\n";


    return 0;
}