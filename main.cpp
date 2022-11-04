#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>

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
    unsigned inDim();
    unsigned outDim();


};

class population
{
    //functions in general order of being called while training
    void runData(vector<vector<double> >); //runs dataset through all networks in pop
                                            //vector of vectors(for multiple data points)
    void Sort(); //sorts networks by their cost
    void cull(); //kills 90% of the worst networks
    void repopulate(); //repopulates the 90% from offshoots of the 10%
    int Size;
public:
    double learningRate;
    void train(vector<vector<double> >, int); //takes a dataset and # of epochs
    network best(); //returns best network, pop[0] once sorted.
    population(int, int, double, vector<unsigned>);
    vector<network> pop; //holds networks
};


//main
int main()
{
    srand(time(NULL));
    vector<unsigned> t;
    t.push_back(2);
    t.push_back(9);
    t.push_back(1);

    vector<vector<double> > ds;
    for(unsigned i = 0;i<500;i++)
    {
        vector<double> temp;
        int i1 = rand()%100;
        int i2 = rand()%100;
        temp.push_back(i1);
        temp.push_back(i2);
        temp.push_back(i1+i2);
        ds.push_back(temp);
    }

    vector<double> testIn = {9,10};

    population p = population(100,1,60,t);
    p.train(ds, 1000);
    p.best().predict(testIn);
    p.best().printNet();
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
unsigned network::inDim()
{
    return topology[0];
}
unsigned network::outDim()
{
    return topology.back();
}

population::population(int SIZE, int scale, double LearningRate, vector<unsigned> Topology)
{
    learningRate = LearningRate;
    Size = SIZE;
    for(int i = 0;i<SIZE;i++)
    {
        pop.push_back(network(Topology,scale));
    }
}
network population::best()
{
    return pop[0];
}

void population::runData(vector<vector<double> > inData)
{
    for(unsigned i = 0;i<inData.size();i++)//for each data entry
    {
        vector<double> in;
        vector<double> gt;
        for(unsigned j = 0;j<pop[0].outDim();j++)
        {
            double le = inData[i].back();
            gt.push_back(le);
            inData[i].pop_back();
        }
        reverse(gt.begin(),gt.end());
        for(unsigned j = 0;j<pop[0].inDim();j++)
        {
            double le = inData[i].back();
            in.push_back(le);
            inData[i].pop_back();
        }
        reverse(in.begin(),in.end());
        for(unsigned j = 0;j<pop.size();j++)//for each network in the population
        {
            pop[j].predict(in,gt);
        }
    }
}
void population::Sort()
{
    sort(pop.begin(),pop.end());
}
void population::cull()
{
    for(int i = 0;i<(int)(pop.size()*.9);i++)
    {
        //cout << "killed cost " << pop.back().cost << endl;
        pop.pop_back();
    }
}
void population::repopulate()
{
    int bestRange = pop.size();
    int neededNets = Size-bestRange;
    for(int i = 0;i<neededNets;i++)
    {
        int chooseIndex = rand()%bestRange;
        network temp  = network(pop[chooseIndex], learningRate);
        pop.push_back(temp);
    }
}
void population::train(vector<vector<double> > dataset, int epochs)
{
    for(unsigned i = 0;i<pop.size();i++) pop[i].cost = 0;
    for(int i = 0;i< epochs;i++)
    {
        runData(dataset);//runs all of the data, getting a total cost for each network
        Sort(); //sorts
        cull();//kills off bad networks
        repopulate();//replaces victims
    }
}
