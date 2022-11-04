#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <fstream>

using namespace std;
//classes
class neuron
{
    double weight;
    int Scale;
public:
    double Accumulate;
    void setWeight(double);
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
    void exportNet(string);
    void loadNet(string);


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
    //making a topology to pass to a network population
    srand(time(NULL));
    vector<unsigned> t;
    t.push_back(2);
    t.push_back(9);
    t.push_back(3);
    t.push_back(1);

    //making a dataset to train on
    vector<vector<double> > ds;
    for(unsigned i = 0;i<500;i++)
    {
        vector<double> temp;
        int i1 = rand()%10;
        int i2 = rand()%10;
        temp.push_back(i1);
        temp.push_back(i2);
        temp.push_back(i1+i2);
        ds.push_back(temp);
    }

    //vector to pass to get a prediction
    vector<double> testIn = {1,3};

    //creating a population to evolve
    population p = population(100,1,90,t); //(SIZE, scale, learning rate, topology)

    p.train(ds, 10000); // (dataset, epoch#)
    p.best().predict(testIn); //(datapoint)
    p.best().exportNet("Adder"); //(filename) {no file extension}

    /*
    network n1 = network(t,1);
    network n2 = network(t,1);
    n2.loadNet("Adder");
    cout << "Untrained:";
    n1.predict(testIn);
     cout << "Trained:";
    n2.predict(testIn);
    */
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
    Accumulate = 0;
}
void neuron::setWeight(double Weight)
{
    weight = Weight;
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

void network::exportNet(string filename)
{
    //creating output file
    ofstream out;
    filename+=".txt";
    out.open(filename);
    for(unsigned i = 0;i<topology.size();i++)
    {
        out << topology[i];
        if(i+1<=topology.size()) out << "|";
    }

    out << "[";
    for(int i = 0;i<numLayers;i++) //for every layer
    {
        for(unsigned j = 0;j<Layers[i].size();j++)//for every node in the layer
        {
            out << Layers[i][j].getWeight();
            if(j+1<=Layers[i].size()) out << "|";
        }
    }
    out << "]";

    out.close();
}

void network::loadNet(string filename)
{
    //looks at the desired text file. Reads first few digits with the delimiter '|' to get the topology. Continues and makes a list of the weights.
    //Reverses the order of the weights and starts adding them to the network and popping them off the list.
    topology.clear();
    Layers.clear();
    ifstream in;
    filename+=".txt";
    in.open(filename);
    if(!in.is_open())
    {
        cout << "File not found, unable to load model.\n";
        return;
    }
    string currentNumber;
    char s;
    bool onWeights = false;
    vector<double> weightList;
    while(in >> s)
    {
        if(!onWeights)
        {
            if(s=='[')
            {
                onWeights = true;
                currentNumber = "";
            }
            else if(s=='|')
            {
                topology.push_back((unsigned)stoi(currentNumber));
                currentNumber = "";

            }
            else
            {
                currentNumber+=s;
            }
            continue;
        }
        if(s==']')
            {
                break;
            }
        else if(s=='|')
        {
            double w = stod(currentNumber);
            weightList.push_back(w);
            currentNumber = "";
        }
        else
        {
            currentNumber+=s;
        }
    }
    reverse(weightList.begin(), weightList.end());
    for(unsigned i = 0;i<topology.size();i++)
    {
        vector<neuron> tempV;
        for(unsigned j = 0;j<topology[i];j++)
        {
            neuron tempN = neuron(weightList.back(),1);
            weightList.pop_back();
            tempV.push_back(tempN);
        }
        Layers.push_back(tempV);
    }
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
    //built in sort, would like to replace by a self made one eventually.
    sort(pop.begin(),pop.end());
}
void population::cull()
{
    //calculates how many the bottom 90% encompass. pops that amount off.
    for(int i = 0;i<(int)(pop.size()*.9);i++)
    {
        pop.pop_back();
    }
}
void population::repopulate()
{
    //chooses a random network from the best 10%, creates a deviation of that network. Repeat until population is refilled.
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
        system("CLS");
        cout << "Progress: %";
        cout << (i/(double)epochs)*100;
        runData(dataset);//runs all of the data, getting a total cost for each network
        Sort(); //sorts
        cull();//kills off bad networks
        repopulate();//replaces victims
    }
    system("CLS");
    cout << "Progress: %100";
    cout << endl;
}
