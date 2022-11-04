#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>

using namespace std;
//classes
class neuron
{
    double weight;
    int Scale;
public:
    double Accumulate;
    const void setWeight(double);
    double getWeight();
    neuron(int);
    neuron(double, int);
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
    int Scale;
    bool operator== (network);
    bool operator< (network);
    bool operator> (network);
    double cost;
    network(vector<unsigned>, int);
    network(network&, double); // 'copy & slightly change' constructor
    void predict(vector<double>);
    void predict(vector<double>, vector<double>);
    void printNet();


};

class population
{
    //functions in general order of being called while training
    void runData(vector<vector<double> >); //runs dataset through all networks in pop
                                            //vector of vectors(for multiple data points)
    void Sort(); //sorts networks by their cost
    void cull(); //kills 90% of the worst networks
    void repopulate(); //repopulates the 90% from offshoots of the 10%
public:
    void train(vector<double>, int); //takes a dataset and # of epochs
    network best(); //returns best network, pop[0] once sorted.
    population(int,vector<unsigned>);
    vector<network> pop; //holds networks
};


//main
int main()
{
    srand(time(NULL));
    vector<unsigned> t;
    t.push_back(3);
    t.push_back(2);
    t.push_back(1);

    vector<double> inp;
    inp.push_back(1);
    inp.push_back(2);
    inp.push_back(3);

    vector<double> gt;
    gt.push_back(5);

    network n = network(t, 1);
    network cn = network(n, 10);
    n.printNet();
    cn.printNet();
    return 0;
}


//class functions
neuron::neuron(int scale)
{
    Scale = scale;
    int sign = rand()%2;
    double temp = rand();
    if(sign) temp*=-1;
    weight = (temp*(Scale/(double)RAND_MAX));
    Accumulate = 0;
}
neuron::neuron(double Weight, int scale)
{
    Scale = scale;
    weight = Weight;
    if (Weight>Scale) Weight= Scale;//might cause issues in the future with loading
                                    //in a pretrained model
    Accumulate = 0;
}
const void neuron::setWeight(double Weight)
{
    weight = Weight;
    while(weight>Scale) weight-=Scale; //might cause issues in the future with loading
                                        //in a pretrained model
}
double neuron::getWeight()
{
    return weight;
}

network::network(vector<unsigned> Topology, int scale)
{
    Scale = scale;
    topology = Topology;
    cost = 0;
    numLayers = topology.size();
    //pushes back #numLayers into the network, and after each pushback,
    //populates the layer with the desired # of neurons + a bias
    for(int i = 0;i<numLayers;i++)
    {
        Layers.push_back(layer());
        for(unsigned j = 0;j<topology[i];j++) Layers.back().push_back(neuron(Scale)); //<= to add bias
    }
    cout << "Network constructed\n";
}

network::network(network &target, double learningRate)
{
    //takes learning rate (a number > 1; a percent, not decimal) and for each neuron
    //in the old network, creates a new one within +-learningRate% of the old.
    //so learningRate = 1 basically copies the old netork

    Scale = target.Scale;
    if(learningRate<1)
    {
        cout << "Error: Learning rate(percent) is less than 1.\n\n";
    }
    topology = target.topology;
    cost = 0;
    numLayers = topology.size();
    for(int i = 0;i<numLayers;i++)
    {

        Layers.push_back(layer());

        for(unsigned j = 0;j<topology[i];j++)
            {
            int increaseOrDecrease = rand()%2;
                double temp = rand()%(int)learningRate;
                temp/=100;
                if(increaseOrDecrease) temp = 1+temp;
                else temp = 1-temp;
                double tw = target.Layers[i][j].getWeight();
                Layers.back().push_back(neuron(tw*temp, Scale)); //<= to add bias
            }
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
        for(unsigned j = 0;j<Layers[i].size();j++) //for every neuron in the current layer
        {
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
    for(int i = 0;i<numLayers;i++)
    {
        for(unsigned j = 0;j<Layers[i].size();j++)
        {
            Layers[i][j].Accumulate = 0;
        }
    }
}

void network::predict(vector<double> input,vector<double> groundTruth)
{
    //dimension check
    if(input.size()!=topology[0])
    {
        cout << "Input dimensions do not match, expected " << topology[0];
        cout << " got " << input.size() << endl;
        return;
    }
    if(groundTruth.size()!=Layers[numLayers-1].size())
    {
        cout << "Output dimensions do not match, expected " << Layers[numLayers-1].size();
        cout << " got " << groundTruth.size() << endl;
        return;
    }
    //putting input into the network
    for(unsigned i = 0;i<Layers[0].size();i++)
    {
        Layers[0][i].Accumulate = input[i];
    }
    //
    for(int i = 0;i<numLayers-1;i++)//for every layer(excluding output layer)
    {
        for(unsigned j = 0;j<Layers[i].size();j++) //for every neuron in the current layer
        {
            double feedWeight = relu(Layers[i][j].getWeight());
            for(unsigned k = 0;k<Layers[i+1].size();k++) //for every neuron in the 2nd layer
            {
                //Add the current neuron's accumulate multiplied by its weight to
                //all neurons in the next layer's accumulate
                Layers[i+1][k].Accumulate += feedWeight*Layers[i][k].Accumulate;
            }
        }
    }
    //cost is the euclidean distance from the prediction to the ground truth
    double costSquared = 0;
   for(unsigned i = 0;i<groundTruth.size();i++)
   {
       costSquared +=(groundTruth[i]-Layers[numLayers-1][i].Accumulate)*(groundTruth[i]-Layers[numLayers-1][i].Accumulate);
   }
   cost+=sqrt(costSquared);
    for(int i = 0;i<numLayers;i++)
    {
        for(unsigned j = 0;j<Layers[i].size();j++)
        {
            Layers[i][j].Accumulate = 0;
        }
    }
}

double network::relu(double in)
{
    if(in<0) return 0;
    else return in;
}
bool network::operator== (network A)
{
    if(this->cost==A.cost) return true;
    else return false;
}
bool network::operator< (network A)
{
    if(this->cost<A.cost) return true;
    else return false;
}
bool network::operator> (network A)
{
    if(this->cost>A.cost) return true;
    else return false;
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
