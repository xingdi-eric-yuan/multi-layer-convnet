#pragma once
#include "general_settings.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace cv;

void read_MNIST_data(vector<Mat>&, vector<Mat>&, Mat&, Mat&);

void read_Mnist(string, vector<Mat> &);

void read_Mnist_Label(string, Mat &);

void readData(vector<Mat> &, Mat &, string, string, int);

void preProcessing(vector<Mat>&, vector<Mat>&);

int ReverseInt (int);

Mat concat(const vector<Mat> &);