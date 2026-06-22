#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;

int main() {

    vector<double> x;
    vector<double> y;

    // =====================
    // Read CSV File
    // =====================
    ifstream file("data.csv");

    string line;

    getline(file, line); // Skip header

    while(getline(file, line)) {

        stringstream ss(line);

        string crab, clam;

        getline(ss, crab, ',');
        getline(ss, clam, ',');

        x.push_back(stod(crab));
        y.push_back(stod(clam));
    }

    file.close();

    // =====================
    // Initialize
    // =====================
    double m = 0.0;
    double c = 0.0;

    double learning_rate = 0.001;

    int epochs = 50000;

    int n = x.size();

    // =====================
    // Training
    // =====================
    for(int epoch = 0; epoch < epochs; epoch++) {

        double cost = 0;
        double dm = 0;
        double dc = 0;

        for(int i = 0; i < n; i++) {

            double prediction = m * x[i] + c;

            cost += pow(prediction - y[i], 2);

            dm += (prediction - y[i]) * x[i];

            dc += (prediction - y[i]);
        }

        cost = cost / (2 * n);

        dm = dm / n;
        dc = dc / n;

        // Update Parameters
        m = m - learning_rate * dm;
        c = c - learning_rate * dc;

        if(epoch % 5000 == 0) {

            cout << "Epoch: " << epoch << endl;
            cout << "Cost : " << cost << endl;
            cout << "m    : " << m << endl;
            cout << "c    : " << c << endl;
            cout << "------------------------" << endl;
        }
    }

    // =====================
    // Final Model
    // =====================
    cout << "\nTraining Complete\n";

    cout << "Final Slope (m): " << m << endl;
    cout << "Final Intercept (c): " << c << endl;

    // Prediction Example
    double crabs = 4;

    double prediction = m * crabs + c;

    cout << "\nPrediction for 3 crabs: "
         << prediction << endl;

    return 0;
}