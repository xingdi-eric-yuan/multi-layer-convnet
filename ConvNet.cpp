// ConvNet.cpp
// Version 2.0
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// A Convolutional Neural Networks hand writing classifier.
// You can set the amount of Conv Layers and Full Connected Layers.
// Output layer is softmax regression
//
// To run this code, you should have OpenCV in your computer.
// Have fun with it ^v^
// 
// I'm using mac os so if you're using other OS, just change 
// these "#include"s into your style. Make sure you included
// OpenCV stuff, math.h, f/io/s stream, and unordered_map.

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace cv;
using namespace std;
// Gradient Checking
#define G_CHECKING 1
// Conv2 parameter
#define CONV_FULL 0
#define CONV_SAME 1
#define CONV_VALID 2
// Pooling methods
#define POOL_MAX 0
#define POOL_MEAN 1
#define POOL_STOCHASTIC 2
// get Key type
#define KEY_CONV 0
#define KEY_POOL 1
#define KEY_DELTA 2
#define KEY_UP_DELTA 3
// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

#define ATD at<double>
#define elif else if
int NumHiddenNeurons = 200;
int NumHiddenLayers = 2;
int nclasses = 10;
int NumConvLayers = 2;

vector<int> KernelSize;
vector<int> KernelAmount;
vector<int> PoolingDim;

int batch;
int Pooling_Methed = POOL_MAX;
int nonlin = NL_RELU;

typedef struct ConvKernel{
    Mat W;
    double b;
    Mat Wgrad;
    double bgrad;
}ConvK;

typedef struct ConvLayer{
    vector<ConvK> layer;
    int kernelAmount;
}Cvl;

typedef struct Network{
    Mat W;
    Mat b;
    Mat Wgrad;
    Mat bgrad;
}Ntw;

typedef struct SoftmaxRegession{
    Mat Weight;
    Mat Wgrad;
    Mat b;
    Mat bgrad;
    double cost;
}SMR;

// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s=ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

void
unconcatenateMat(vector<Mat> &src, vector<vector<Mat> > &dst, int vsize){
    for(int i=0; i<src.size() / vsize; i++){
        vector<Mat> tmp;
        for(int j=0; j<vsize; j++){
            tmp.push_back(src[i * vsize + j]);
        }
        dst.push_back(tmp);
    }
}

Mat 
concatenateMat(vector<vector<Mat> > &vec){

    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    int height = vec[0].size() * subFeatures;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);

    for(int i=0; i<vec.size(); i++){
        for(int j=0; j<vec[i].size(); j++){
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat 
concatenateMat(vector<Mat> &vec, int matcols){
    vector<vector<Mat> > temp;
    unconcatenateMat(vec, temp, vec.size() / matcols);
    return concatenateMat(temp);
}

int 
ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void 
read_Mnist(string filename, vector<Mat> &vec){
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i){
            Mat tpmat = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tpmat.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tpmat);
        }
    }
}

void 
read_Mnist_Label(string filename, Mat &mat)
{
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            mat.ATD(0, i) = (double)temp;
        }
    }
}

Mat 
sigmoid(Mat &M){
    Mat temp;
    exp(-M, temp);
    return 1.0 / (temp + 1.0);
}

Mat 
dsigmoid(Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat
ReLU(Mat& M){
    Mat res(M);
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            if(M.ATD(i, j) < 0.0) res.ATD(i, j) = 0.0;
        }
    }
    return res;
}

Mat
dReLU(Mat& M){
    Mat res = Mat::zeros(M.rows, M.cols, CV_64FC1);
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            if(M.ATD(i, j) > 0.0) res.ATD(i, j) = 1.0;
        }
    }
    return res;
}

Mat 
Tanh(Mat &M){
    Mat res(M);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            res.ATD(i, j) = tanh(M.ATD(i, j));
        }
    }
    return res;
}

Mat
dTanh(Mat &M){
    Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
    Mat temp;
    pow(M, 2.0, temp);
    res -= temp;
    return res;
}

Mat 
nonLinearity(Mat &M){
    if(nonlin == NL_RELU){
        return ReLU(M);
    }elif(nonlin == NL_TANH){
        return Tanh(M);
    }else{
        return sigmoid(M);
    }
}

Mat 
dnonLinearity(Mat &M){
    if(nonlin == NL_RELU){
        return dReLU(M);
    }elif(nonlin == NL_TANH){
        return dTanh(M);
    }else{
        return dsigmoid(M);
    }
}

// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}

// A Matlab/Octave style 2-d convolution function.
// from http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/
Mat 
conv2(Mat &img, Mat &kernel, int convtype) {
    Mat dest;
    Mat source = img;
    if(CONV_FULL == convtype) {
        source = Mat();
        int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
        copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2, (additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
    }
    Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
    int borderMode = BORDER_CONSTANT;
    Mat fkernal;
    flip(kernel, fkernal, -1);
    filter2D(source, dest, img.depth(), fkernal, anchor, 0, borderMode);

    if(CONV_VALID == convtype) {
        dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
                   .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
    }
    return dest;
}

// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat
kron(Mat &a, Mat &b){

    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC1);
    for(int i=0; i<a.rows; i++){
        for(int j=0; j<a.cols; j++){
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.ATD(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}

Point 
findLoc(Mat &prob, int m){
    Mat temp, idx;
    Point res = Point(0, 0);
    prob.reshape(0, 1).copyTo(temp); 
    sortIdx(temp, idx, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
    int i = idx.at<int>(0, m);
    res.x = i % prob.rows;
    res.y = i / prob.rows;
    return res;
}

Mat
Pooling(Mat &M, int pVert, int pHori, int poolingMethod, vector<Point> &locat, bool isTest){
    int remX = M.cols % pHori;
    int remY = M.rows % pVert;
    Mat newM;
    if(remX == 0 && remY == 0) M.copyTo(newM);
    else{
        Rect roi = Rect(remX, remY, M.cols - remX, M.rows - remY);
        M(roi).copyTo(newM);
    }
    Mat res = Mat::zeros(newM.rows / pVert, newM.cols / pHori, CV_64FC1);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            Mat temp;
            Rect roi = Rect(j * pHori, i * pVert, pHori, pVert);
            newM(roi).copyTo(temp);
            double val;
            // for Max Pooling
            if(POOL_MAX == poolingMethod){ 
                double minVal; 
                double maxVal; 
                Point minLoc; 
                Point maxLoc;
                minMaxLoc( temp, &minVal, &maxVal, &minLoc, &maxLoc );
                val = maxVal;
                locat.push_back(Point(maxLoc.x + j * pHori, maxLoc.y + i * pVert));
            }elif(POOL_MEAN == poolingMethod){
                // Mean Pooling
                val = sum(temp)[0] / (pVert * pHori);
            }elif(POOL_STOCHASTIC == poolingMethod){
                // Stochastic Pooling
                double sumval = sum(temp)[0];
                Mat prob = temp / sumval;
                if(isTest){
                    val = sum(prob.mul(temp))[0];
                }else{
                    int ran = rand() % (temp.rows * temp.cols);
                    Point loc = findLoc(prob, ran);
                    val = temp.ATD(loc.y, loc.x);
                    locat.push_back(Point(loc.x + j * pHori, loc.y + i * pVert));
                }
            }
            res.ATD(i, j) = val;
        }
    }
    return res;
}

Mat 
UnPooling(Mat &M, int pVert, int pHori, int poolingMethod, vector<Point> &locat){
    Mat res;
    if(POOL_MEAN == poolingMethod){
        Mat one = Mat::ones(pVert, pHori, CV_64FC1);
        res = kron(M, one) / (pVert * pHori);
    }elif(POOL_MAX == poolingMethod || POOL_STOCHASTIC == poolingMethod){
        res = Mat::zeros(M.rows * pVert, M.cols * pHori, CV_64FC1);
        for(int i=0; i<M.rows; i++){
            for(int j=0; j<M.cols; j++){
                res.ATD(locat[i * M.cols + j].y, locat[i * M.cols + j].x) = M.ATD(i, j);
            }
        }
    }
    return res;
}

double 
getLearningRate(Mat &data){
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;
    int nsamples = data.cols;
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;
    SVD uwvT = SVD(Sigma);
    return 0.9 / uwvT.w.ATD(0, 0);
}

void
weightRandomInit(ConvK &convk, int width){

    double epsilon = 0.1;
    convk.W = Mat::ones(width, width, CV_64FC1);
    double *pData; 
    for(int i = 0; i<convk.W.rows; i++){
        pData = convk.W.ptr<double>(i);
        for(int j=0; j<convk.W.cols; j++){
            pData[j] = randu<double>();        
        }
    }
    convk.W = convk.W * (2 * epsilon) - epsilon;
    convk.b = 0;
    convk.Wgrad = Mat::zeros(width, width, CV_64FC1);
    convk.bgrad = 0;
}

void
weightRandomInit(Ntw &ntw, int inputsize, int hiddensize){

    double epsilon = sqrt(6) / sqrt(hiddensize + inputsize + 1);
    double *pData;
    ntw.W = Mat::ones(hiddensize, inputsize, CV_64FC1);
    for(int i=0; i<hiddensize; i++){
        pData = ntw.W.ptr<double>(i);
        for(int j=0; j<inputsize; j++){
            pData[j] = randu<double>();
        }
    }
    ntw.W = ntw.W * (2 * epsilon) - epsilon;
    ntw.b = Mat::zeros(hiddensize, 1, CV_64FC1);
    ntw.Wgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.bgrad = Mat::zeros(hiddensize, 1, CV_64FC1);
}

void 
weightRandomInit(SMR &smr, int nclasses, int nfeatures){
    double epsilon = 0.01;
    smr.Weight = Mat::ones(nclasses, nfeatures, CV_64FC1);
    double *pData; 
    for(int i = 0; i<smr.Weight.rows; i++){
        pData = smr.Weight.ptr<double>(i);
        for(int j=0; j<smr.Weight.cols; j++){
            pData[j] = randu<double>();        
        }
    }
    smr.Weight = smr.Weight * (2 * epsilon) - epsilon;
    smr.b = Mat::zeros(nclasses, 1, CV_64FC1);
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.bgrad = Mat::zeros(nclasses, 1, CV_64FC1);
}

void
ConvNetInitPrarms(vector<Cvl> &ConvLayers, vector<Ntw> &HiddenLayers, SMR &smr, int imgDim, int nsamples){

    // Init Conv layers
    for(int i=0; i<NumConvLayers; i++){
        Cvl tpcvl;
        for(int j=0; j<KernelAmount[i]; j++){
            ConvK tmpConvK;
            weightRandomInit(tmpConvK, KernelSize[i]);
            tpcvl.layer.push_back(tmpConvK);
        }
        tpcvl.kernelAmount = KernelAmount[i];
        ConvLayers.push_back(tpcvl);
    }

    // Init Hidden layers
    int outDim = imgDim;
    for(int i=0; i<NumConvLayers; i++){
        outDim = outDim - KernelSize[i] + 1;
        outDim = outDim / PoolingDim[i];
    }
    int hiddenfeatures = pow(outDim, 2);
    for(int i=0; i<ConvLayers.size(); i++){
        hiddenfeatures *= ConvLayers[i].kernelAmount;
    }
    Ntw tpntw;
    weightRandomInit(tpntw, hiddenfeatures, NumHiddenNeurons);
    HiddenLayers.push_back(tpntw);
    for(int i=1; i<NumHiddenLayers; i++){
        Ntw tpntw2;
        weightRandomInit(tpntw2, NumHiddenNeurons, NumHiddenNeurons);
        HiddenLayers.push_back(tpntw2);
    }
    // Init Softmax layer
    weightRandomInit(smr, nclasses, NumHiddenNeurons);
}

Mat
getNetworkActivation(Ntw &ntw, Mat &data){
    Mat acti;
    acti = ntw.W * data + repeat(ntw.b, 1, data.cols);
    acti = sigmoid(acti);
    return acti;
}

void 
convAndPooling(vector<Mat> &x, vector<Cvl> &CLayers, 
                unordered_map<string, Mat> &map, 
                unordered_map<string, vector<Point> > &loc){
    // Conv & Pooling
    int nsamples = x.size();
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vec;
        for(int cl = 0; cl < CLayers.size(); cl ++){
            int pdim = PoolingDim[cl];
            if(cl == 0){
                for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = conv2(x[m], temp, CONV_VALID);
                    tmpconv += CLayers[cl].layer[k].b;
                    tmpconv = nonLinearity(tmpconv);
                    map[s2] = tmpconv;
                    vector<Point> PoolingLoc;
                    tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, false);
                    string s3 = s2 + "P";
                    map[s3] = tmpconv;
                    loc[s3] = PoolingLoc;
                    vec.push_back(s3);
                }
            }else{
                vector<string> tmpvec;
                for(int tp = 0; tp < vec.size(); tp ++){
                    for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                        Mat tmpconv = conv2(map[vec[tp]], temp, CONV_VALID);
                        tmpconv += CLayers[cl].layer[k].b;
                        tmpconv = nonLinearity(tmpconv);
                        map[s2] = tmpconv;
                        vector<Point> PoolingLoc;
                        tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, false);
                        string s3 = s2 + "P";
                        map[s3] = tmpconv;
                        loc[s3] = PoolingLoc;
                        tmpvec.push_back(s3);
                    }
                }
                swap(vec, tmpvec);
                tmpvec.clear();
            }
        }    
        vec.clear();   
    }
}

void 
convAndPooling(vector<Mat> &x, vector<Cvl> &CLayers, 
                unordered_map<string, Mat> &map){
    // Conv & Pooling
    int nsamples = x.size();
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vec;
        for(int cl = 0; cl < CLayers.size(); cl ++){
            int pdim = PoolingDim[cl];
            if(cl == 0){
                for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
                    string s2 = s1 + "C0K" + i2str(k);
                    Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                    Mat tmpconv = conv2(x[m], temp, CONV_VALID);
                    tmpconv += CLayers[cl].layer[k].b;
                    tmpconv = nonLinearity(tmpconv);
                    map[s2] = tmpconv;
                    vector<Point> PoolingLoc;
                    tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, true);
                    string s3 = s2 + "P";
                    map[s3] = tmpconv;
                    vec.push_back(s3);
                }
            }else{
                vector<string> tmpvec;
                for(int tp = 0; tp < vec.size(); tp ++){
                    for(int k = 0; k < CLayers[cl].kernelAmount; k ++){
                        string s2 = vec[tp] + "C" + i2str(cl) + "K" + i2str(k);
                        Mat temp = rot90(CLayers[cl].layer[k].W, 2);
                        Mat tmpconv = conv2(map[vec[tp]], temp, CONV_VALID);
                        tmpconv += CLayers[cl].layer[k].b;
                        tmpconv = nonLinearity(tmpconv);
                        map[s2] = tmpconv;
                        vector<Point> PoolingLoc;
                        tmpconv = Pooling(tmpconv, pdim, pdim, Pooling_Methed, PoolingLoc, true);
                        string s3 = s2 + "P";
                        map[s3] = tmpconv;
                        tmpvec.push_back(s3);
                    }
                }
                swap(vec, tmpvec);
                tmpvec.clear();
            }
        }    
        vec.clear();   
    }
}

void
hashDelta(Mat &src, unordered_map<string, Mat> &map, vector<Cvl> &CLayers){
    int nsamples = src.cols;
    for(int m = 0; m < nsamples; m ++){
        string s1 = "X" + i2str(m);
        vector<string> vecstr;
        for(int i = 0; i < CLayers.size(); i ++){
            if(i == 0){
                string s2 = s1 + "C0";
                for(int k = 0; k < CLayers[i].kernelAmount; k ++){
                    string s3 = s2 + "K" + i2str(k) + "P";
                    if(i == CLayers.size() - 1){
                        s3 += "D";
                    }
                    vecstr.push_back(s3);
                }
            }else{
                vector<string> vec2;
                for(int j = 0; j < vecstr.size(); j ++){
                    string s2 = vecstr[j] + "C" + i2str(i);
                    for(int k = 0; k < CLayers[i].kernelAmount; k ++){
                        string s3 = s2 + "K" + i2str(k) + "P";
                        if(i == CLayers.size() - 1){
                            s3 += "D";
                        }
                        vec2.push_back(s3);
                    }
                }
                swap(vecstr, vec2);
                vec2.clear();
            }
        }
        int sqDim = src.rows / vecstr.size();
        int Dim = sqrt(sqDim);
        for(int i=0; i<vecstr.size(); i++){
            Rect roi = Rect(m, i * sqDim, 1, sqDim);
            Mat temp;
            src(roi).copyTo(temp);
            Mat img = temp.reshape(0, Dim);
            map[vecstr[i]] = img;
        }  
    }
}

vector<string>
getLayerNKernelNKey(vector<Cvl> &CLayers, int nsamples, int nLayer, int nKernel, int keyType){
    vector<string> vecstr;
    for(int i=0; i<nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j=0; j<=nLayer; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            if(j == nLayer){
                int k = nKernel;
                string s3 = s2 + "K" + i2str(k);
                if(keyType == KEY_POOL) s3 += "P";
                elif(keyType == KEY_DELTA) s3 += "PD";
                elif(keyType == KEY_UP_DELTA) s3 += "PUD";
                tmpvecstr.push_back(s3);
            }else{
                for(int k=0; k<CLayers[j].kernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    s3 += "P";
                    tmpvecstr.push_back(s3);
                }
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

vector<string>
getLayerNKernelNKey(vector<Cvl> &CLayers, int nsamples, int nLayer, int kernelLayer, int nKernel, int keyType){
    if(nLayer == kernelLayer) return getLayerNKernelNKey(CLayers, nsamples, nLayer, nKernel, keyType);
    vector<string> vecstr;
    for(int i=0; i<nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j=0; j<=nLayer; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            if(j == nLayer){
                for(int k=0; k<CLayers[j].kernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    if(keyType == KEY_POOL) s3 += "P";
                    elif(keyType == KEY_DELTA) s3 += "PD";
                    elif(keyType == KEY_UP_DELTA) s3 += "PUD";
                    tmpvecstr.push_back(s3);
                }
            }elif(j == kernelLayer){
                int k = nKernel;
                string s3 = s2 + "K" + i2str(k) + "P";
                tmpvecstr.push_back(s3);
            }else{
                for(int k=0; k<CLayers[j].kernelAmount; k++){
                    string s3 = s2 + "K" + i2str(k);
                    s3 += "P";
                    tmpvecstr.push_back(s3);
                }
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

vector<string>
getLayerNKey(vector<Cvl> &CLayers, int nsamples, int n, int keyType){
    vector<string> vecstr;
    for(int i=0; i<nsamples; i++){
        string s1 = "X" + i2str(i);
        vecstr.push_back(s1);
    }
    for(int j=0; j<=n; j++){
        vector<string> tmpvecstr;
        for(int i=0; i<vecstr.size(); i++){
            string s2 = vecstr[i] + "C" + i2str(j);
            for(int k=0; k<CLayers[j].kernelAmount; k++){
                string s3 = s2 + "K" + i2str(k);
                if(j != n){
                    s3 += "P";
                }else{
                    if(keyType == KEY_POOL){
                        s3 += "P";
                    }elif(keyType == KEY_DELTA){
                        s3 += "PD";
                    }elif(keyType == KEY_UP_DELTA){
                        s3 += "PUD";
                    }
                }
                tmpvecstr.push_back(s3);
            }
        }
        swap(vecstr, tmpvecstr);
        tmpvecstr.clear();
    }
    return vecstr;
}

int 
getSampleNum(string str){
    int i = 1;
    while(str[i] >='0' && str[i] <= '9'){
        ++ i;
    }
    string sub = str.substr(1, i - 1);
    return str2i(sub);
}

int 
getCurrentKernel(string str){
    int i = str.length() - 1;
    while(str[i] !='K'){
        -- i;
    }
    int start = i + 1;
    i = start;
    while(str[i] <= '9' && str[i] >= '0'){
        ++ i;
    }
    string sub = str.substr(start, i - start);
    return str2i(sub);
}

string
getPreviousLayerKey(string str, int keyType){
    int i = str.length() - 1; 
    while(str[i] != 'C'){
        -- i;
    }
    if(keyType == KEY_CONV){
        return str.substr(0, i - 1);
    }elif(keyType == KEY_POOL){
        return str.substr(0, i);
    }elif(keyType == KEY_DELTA){
        return str.substr(0, i) + "D";
    }else{
        return str.substr(0, i) + "UD";
    }
}

void
getNetworkCost(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Ntw> &hLayers, SMR &smr, double lambda){

    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    unordered_map<string, vector<Point> > locmap;
    convAndPooling(x, CLayers, cpmap, locmap);

    vector<Mat> P;
    vector<string> vecstr = getLayerNKey(CLayers, nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i=0; i<vecstr.size(); i++){
        P.push_back(cpmap[vecstr[i]]);
    }
    Mat convolvedX = concatenateMat(P, nsamples);
    P.clear();

    // full connected layers
    vector<Mat> acti;
    acti.push_back(convolvedX);
    for(int i=1; i<=NumHiddenLayers; i++){
        Mat tmpacti = hLayers[i - 1].W * acti[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
        acti.push_back(sigmoid(tmpacti));
    }

    Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M -= repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);

    // softmax regression
    Mat groundTruth = Mat::zeros(nclasses, nsamples, CV_64FC1);
    for(int i=0; i<nsamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }
    Mat logP;
    log(p, logP);
    logP = logP.mul(groundTruth);
    smr.cost = - sum(logP)[0] / nsamples;
    pow(smr.Weight, 2.0, tmp);
    smr.cost += sum(tmp)[0] * lambda / 2;
    for(int cl=0; cl<CLayers.size(); cl++){
        for(int i=0; i<CLayers[cl].kernelAmount; i++){
            pow(CLayers[cl].layer[i].W, 2.0, tmp);
            smr.cost += sum(tmp)[0] * lambda / 2;
        }
    }

    // bp - softmax
    tmp = (groundTruth - p) * acti[acti.size() - 1].t();
    tmp /= -nsamples;
    smr.Wgrad = tmp + lambda * smr.Weight;
    reduce((groundTruth - p), tmp, 1, CV_REDUCE_SUM);
    smr.bgrad = tmp / -nsamples;

    // bp - full connected
    vector<Mat> delta(acti.size());
    delta[delta.size() -1] = -smr.Weight.t() * (groundTruth - p);
    delta[delta.size() -1] = delta[delta.size() -1].mul(dsigmoid(acti[acti.size() - 1]));
    for(int i = delta.size() - 2; i >= 0; i--){
        delta[i] = hLayers[i].W.t() * delta[i + 1];
        if(i > 0) delta[i] = delta[i].mul(dsigmoid(acti[i]));
    }
    for(int i=NumHiddenLayers - 1; i >=0; i--){
        hLayers[i].Wgrad = delta[i + 1] * acti[i].t();
        hLayers[i].Wgrad /= nsamples;
        reduce(delta[i + 1], tmp, 1, CV_REDUCE_SUM);
        hLayers[i].bgrad = tmp / nsamples;
    }

    //bp - Conv layer
    hashDelta(delta[0], cpmap, CLayers);
    for(int cl = CLayers.size() - 1; cl >= 0; cl --){
        int pDim = PoolingDim[cl];
        vector<string> deltaKey = getLayerNKey(CLayers, nsamples, cl, KEY_DELTA);

        for(int k = 0; k < deltaKey.size(); k ++){
            string locstr = deltaKey[k].substr(0, deltaKey[k].length() - 1);
            string convstr = deltaKey[k].substr(0, deltaKey[k].length() - 2);
            Mat upDelta = UnPooling(cpmap[deltaKey[k]], pDim, pDim, Pooling_Methed, locmap[locstr]);
            upDelta = upDelta.mul(dnonLinearity(cpmap[convstr]));

            string upDstr = locstr + "UD";
            cpmap[upDstr] = upDelta;
        }
        if(cl > 0){
            for(int k = 0; k < CLayers[cl - 1].kernelAmount; k ++){
                vector<string> prev = getLayerNKernelNKey(CLayers, nsamples, cl, cl - 1, k, KEY_UP_DELTA);
                for(int i = 0; i < prev.size(); i++){
                    string strd = getPreviousLayerKey(prev[i], KEY_DELTA);
                    unordered_map<string, Mat>::iterator got = cpmap.find(strd);
                    if(got == cpmap.end()){
                        string psize = getPreviousLayerKey(prev[i], KEY_POOL);
                        Mat zero = Mat::zeros(cpmap[psize].rows, cpmap[psize].cols, CV_64FC1);
                        cpmap[strd] = zero;
                    }
                    int currentKernel = getCurrentKernel(prev[i]);
                    cpmap[strd] += conv2(cpmap[prev[i]], CLayers[cl].layer[currentKernel].W, CONV_FULL);
                }
            }
        }
        for(int j = 0; j < CLayers[cl].kernelAmount; j ++){
            Mat tpgradW = Mat::zeros(KernelSize[cl], KernelSize[cl], CV_64FC1);
            double tpgradb = 0.0;
            vector<string> convKey = getLayerNKernelNKey(CLayers, nsamples, cl, j, KEY_UP_DELTA);
            for(int m = 0; m < convKey.size(); m ++){
                Mat temp = rot90(cpmap[convKey[m]], 2);
                if(cl == 0){
                    tpgradW += conv2(x[getSampleNum(convKey[m])], temp, CONV_VALID);
                }else{
                    string strprev = getPreviousLayerKey(convKey[m], KEY_POOL);
                    tpgradW += conv2(cpmap[strprev], temp, CONV_VALID);
                }
                tpgradb += sum(cpmap[convKey[m]])[0];
            }
            CLayers[cl].layer[j].Wgrad = tpgradW / nsamples + lambda * CLayers[cl].layer[j].W;
            CLayers[cl].layer[j].bgrad = tpgradb / nsamples;
        }
    }
    // deconstruct
    cpmap.clear();
    locmap.clear();
    acti.clear();
    delta.clear();
}

void
gradientChecking(vector<Cvl> &CLayers, vector<Ntw> &hLayers, SMR &smr, vector<Mat> &x, Mat &y, double lambda){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
    int a = 0;
    int b = 1;
    Mat grad(CLayers[a].layer[b].Wgrad);
    cout<<"test network !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<CLayers[a].layer[b].W.rows; i++){
        for(int j=0; j<CLayers[a].layer[b].W.cols; j++){
            double memo = CLayers[a].layer[b].W.ATD(i, j);
            CLayers[a].layer[b].W.ATD(i, j) = memo + epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value1 = smr.cost;
            CLayers[a].layer[b].W.ATD(i, j) = memo - epsilon;
            getNetworkCost(x, y, CLayers, hLayers, smr, lambda);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<grad.ATD(i, j) / tp<<endl;
            CLayers[a].layer[b].W.ATD(i, j) = memo;
        }
    }
}

void
trainNetwork(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Ntw> &HiddenLayers, SMR &smr, double lambda){

    if (G_CHECKING){
        gradientChecking(CLayers, HiddenLayers, smr, x, y, lambda);
    }else{
        cout<<"Network Learning................"<<endl;
        // define the velocity vectors.
        Mat v_smr_W = Mat::zeros(smr.Weight.rows, smr.Weight.cols, CV_64FC1);
        Mat v_smr_b = Mat::zeros(smr.b.rows, smr.b.cols, CV_64FC1);
        vector<Mat> v_hl_W;
        vector<Mat> v_hl_b;
        for(int i = 0; i < HiddenLayers.size(); i ++){
            Mat tempW = Mat::zeros(HiddenLayers[i].W.rows, HiddenLayers[i].W.cols, CV_64FC1);
            Mat tempb = Mat::zeros(HiddenLayers[i].b.rows, HiddenLayers[i].b.cols, CV_64FC1);
            v_hl_W.push_back(tempW);
            v_hl_b.push_back(tempb);
        }
        vector<vector<Mat> > v_cvl_W;
        vector<vector<double> > v_cvl_b;
        for(int cl = 0; cl < CLayers.size(); cl++){
            vector<Mat> tmpvecW;
            vector<double> tmpvecb;
            for(int i = 0; i < CLayers[cl].kernelAmount; i ++){
                Mat tempW = Mat::zeros(CLayers[cl].layer[i].W.rows, CLayers[cl].layer[i].W.cols, CV_64FC1);
                double tempb = 0.0;
                tmpvecW.push_back(tempW);
                tmpvecb.push_back(tempb);
            }
            v_cvl_W.push_back(tmpvecW);
            v_cvl_b.push_back(tmpvecb);
        }

        int epochs = 3;
        double lrate = 0.1;
        int iterPerEpo = 250;
        double Momentum = 0.5;
        string s = "costvalue.txt";
        FILE *pOut = fopen(s.c_str(), "w+");
        for(int epo = 0; epo < epochs; epo++){
            for(int k = 0; k < iterPerEpo; k++){

                if(k > 50) Momentum = 0.95;

                int randomNum = ((long)rand() + (long)rand()) % (x.size() - batch);
                vector<Mat> batchX;
                for(int i=0; i<batch; i++){
                    batchX.push_back(x[i + randomNum]);
                }
                Rect roi = Rect(randomNum, 0, batch, y.rows);
                Mat batchY = y(roi);

                getNetworkCost(batchX, batchY, CLayers, HiddenLayers, smr, lambda);

                fprintf(pOut, "%lf\n", smr.cost);
                cout<<"epochs: "<<epo<<", learning step: "<<k<<", Cost function value = "<<smr.cost<<endl;

                v_smr_W = v_smr_W * Momentum + lrate * smr.Wgrad;
                v_smr_b = v_smr_b * Momentum + lrate * smr.b;
                smr.Weight -= v_smr_W;
                smr.b -= v_smr_b;
                for(int i=0; i<HiddenLayers.size(); i++){
                    v_hl_W[i] = v_hl_W[i] * Momentum + lrate * HiddenLayers[i].Wgrad;
                    v_hl_b[i] = v_hl_b[i] * Momentum + lrate * HiddenLayers[i].bgrad;
                    HiddenLayers[i].W -= v_hl_W[i];
                    HiddenLayers[i].b -= v_hl_b[i];
                }
                for(int cl = 0; cl < CLayers.size(); cl++){
                    for(int i=0; i<CLayers[cl].kernelAmount; i++){
                        v_cvl_W[cl][i] = v_cvl_W[cl][i] * Momentum + lrate * CLayers[cl].layer[i].Wgrad;
                        v_cvl_b[cl][i] = v_cvl_b[cl][i] * Momentum + lrate * CLayers[cl].layer[i].bgrad;
                        CLayers[cl].layer[i].W -= v_cvl_W[cl][i];
                        CLayers[cl].layer[i].b -= v_cvl_b[cl][i];
                    }
                }
            }
            lrate *= 0.5;
        }
        fclose(pOut);       
    }
}

void
readData(vector<Mat> &x, Mat &y, string xpath, string ypath, int number_of_images){
    //read MNIST iamge into OpenCV Mat vector
    read_Mnist(xpath, x);
    for(int i=0; i<x.size(); i++){
        x[i].convertTo(x[i], CV_64FC1, 1.0/255, 0);
    }
    //read MNIST label into double vector
    y = Mat::zeros(1, number_of_images, CV_64FC1);
    read_Mnist_Label(ypath, y);
}

Mat 
resultProdict(vector<Mat> &x, vector<Cvl> &CLayers, vector<Ntw> &hLayers, SMR &smr, double lambda){

    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    convAndPooling(x, CLayers, cpmap);

    vector<Mat> P;
    vector<string> vecstr = getLayerNKey(CLayers, nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i=0; i<vecstr.size(); i++){
        P.push_back(cpmap[vecstr[i]]);
    }
    Mat convolvedX = concatenateMat(P, nsamples);
    P.clear();

    // full connected layers
    vector<Mat> acti;
    acti.push_back(convolvedX);
    for(int i=1; i<=NumHiddenLayers; i++){
        Mat tmpacti = hLayers[i - 1].W * acti[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
        acti.push_back(sigmoid(tmpacti));
    }

    Mat M = smr.Weight * acti[acti.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat tmp;
    reduce(M, tmp, 0, CV_REDUCE_MAX);
    M -= repeat(tmp, M.rows, 1);
    Mat p;
    exp(M, p);
    reduce(p, tmp, 0, CV_REDUCE_SUM);
    divide(p, repeat(tmp, p.rows, 1), p);
    log(p, tmp);

    Mat result = Mat::ones(1, tmp.cols, CV_64FC1);
    for(int i=0; i<tmp.cols; i++){
        double maxele = tmp.ATD(0, i);
        int which = 0;
        for(int j=1; j<tmp.rows; j++){
            if(tmp.ATD(j, i) > maxele){
                maxele = tmp.ATD(j, i);
                which = j;
            }
        }
        result.ATD(0, i) = which;
    }
    // deconstruct
    cpmap.clear();
    acti.clear();
    return result;
}

void
saveWeight(Mat &M, string s){
    s += ".txt";
    FILE *pOut = fopen(s.c_str(), "w+");
    for(int i=0; i<M.rows; i++){
        for(int j=0; j<M.cols; j++){
            fprintf(pOut, "%lf", M.ATD(i, j));
            if(j == M.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}

int 
main(int argc, char** argv)
{

    long start, end;
    start = clock();

    vector<Mat> trainX;
    vector<Mat> testX;
    Mat trainY, testY;
    readData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 60000);
    readData(testX, testY, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", 10000);

    cout<<"Read trainX successfully, including "<<trainX[0].cols * trainX[0].rows<<" features and "<<trainX.size()<<" samples."<<endl;
    cout<<"Read trainY successfully, including "<<trainY.cols<<" samples"<<endl;
    cout<<"Read testX successfully, including "<<testX[0].cols * testX[0].rows<<" features and "<<testX.size()<<" samples."<<endl;
    cout<<"Read testY successfully, including "<<testY.cols<<" samples"<<endl;

    int imgDim = trainX[0].rows;
    int nsamples = trainX.size();
    vector<Cvl> ConvLayers;
    vector<Ntw> HiddenLayers;
    SMR smr;
    KernelSize.push_back(5);
    KernelSize.push_back(7);
    KernelAmount.push_back(4);
    KernelAmount.push_back(8);
    PoolingDim.push_back(2);
    PoolingDim.push_back(2);

    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim, nsamples);
    // Train network using Back Propogation
    batch = 256;
    trainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, 3e-3);

    if(! G_CHECKING){
        // Test use test set
        Mat result = resultProdict(testX, ConvLayers, HiddenLayers, smr, 3e-3);
        Mat err(testY);
        err -= result;
        int correct = err.cols;
        for(int i=0; i<err.cols; i++){
            if(err.ATD(0, i) != 0) --correct;
        }
        cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    }    
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}