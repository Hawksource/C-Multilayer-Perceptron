#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

using namespace std;
//classes
class neuron
{
    double weight;
public:
    double Accumulate;
    void setWeight(double);
    double getWeight();
    neuron();
};

typedef vector<neuron> layer;

class network
{
    vector<unsigned> topology;
    int numLayers;
    vector<layer> Layers; //defined as a vector of vector of neurons
                        // Layers[layer number][neuron number]
    double relu(double);
public:
    double cost;
    network(vector<unsigned>);
    void predict(vector<double>);
    void printNet();


};

//main
int main()
{
    srand(time(NULL));
    vector<unsigned> t;
    t.push_back(3);
    t.push_back(2);
    t.push_back(5);

    vector<double> inp;
    inp.push_back(1);
    inp.push_back(2);
    inp.push_back(3);

    network n = network(t);
    n.printNet();
    n.predict(inp);
    n.printNet();
    return 0;
}


//class functions
neuron::neuron()
{
    int sign = rand()%2;
    double temp = rand();
    while(temp>1) temp/=10.0;
    if(sign) temp*=-1;
    weight = temp;
    Accumulate = 0;
    cout << "constructed neuron\n";
}
void neuron::setWeight(double Weight)
{
    weight = Weight;
}
double neuron::getWeight()
{
    return weight;
}

network::network(vector<unsigned> Topology)
{
    topology = Topology;
    cost = 0;
    numLayers = topology.size();
    //pushes back #numLayers into the network, and after each pushback,
    //populates the layer with the desired # of neurons + a bias
    for(int i = 0;i<numLayers;i++)
    {
        Layers.push_back(layer());
        for(unsigned j = 0;j<topology[i];j++) Layers.back().push_back(neuron()); //<= to add bias
    }
    cout << "Network constructed\n";
}

void network::predict(vector<double> input)
{
    //dimension check
    if(input.size()!=topology[0])
    {
        cout << "Input dimensions do not match, expected " << topology[0];
        cout << " got " << input.size() << endl;
        return;
    }
    //putting input into the network
    for(unsigned i = 0;i<Layers[0].size();i++)
    {
        Layers[0][i].Accumulate = input[i];
    }
    //
    for(int i = 0;i<numLayers-1;i++)//for every layer
    {
        cout << "Feed forward layer\n";
        for(unsigned j = 0;j<Layers[i].size();j++) //for every neuron in the current layer
        {
            cout << "neuron in layer\n";
            double feedWeight = relu(Layers[i][j].getWeight());
            for(unsigned k = 0;k<Layers[i+1].size();k++) //for every neuron in the 2nd layer
            {
                //Add the current neuron's accumulate multiplied by its weight to
                //all neurons in the next layer's accumulate
                Layers[i+1][k].Accumulate += feedWeight*Layers[i][k].Accumulate;
            }
        }
    }
    cout << "Prediction: ";
    for(unsigned i = 0;i<Layers[numLayers-1].size();i++)
    {
        cout << Layers[numLayers-1][i].Accumulate << "\t";
    }
    cout << endl << endl;
}
double network::relu(double in)
{
    if(in<0) return 0;
    else return in;
}
void network::printNet()
{
    for(int i = 0;i<numLayers;i++)
    {
        cout << "Layer " << i << ":\n";
        for(unsigned j = 0;j<Layers[i].size();j++)
        {
            cout << "Weight: " << Layers[i][j].getWeight();
            cout << " | Value: " << Layers[i][j].Accumulate << endl << endl;
        }
        cout << "\n";
    }
}