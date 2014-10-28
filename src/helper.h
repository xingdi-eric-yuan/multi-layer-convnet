#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

// int <==> string
string i2str(int);
int str2i(string);

void unconcatenateMat(const vector<Mat>&, vector<vector<Mat> >*, int);
Mat concatenateMat(const vector<vector<Mat> >&);
Mat concatenateMat(const vector<Mat>&, int );
double getLearningRate(const Mat&);
