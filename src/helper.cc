#include "helper.h"

using namespace cv;
using namespace std;


// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s = ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

void
unconcatenateMat(const vector<Mat> &src, vector<vector<Mat> > *dst, int vsize){
    for(int i = 0; i < src.size() / vsize; i++){
        vector<Mat> tmp;
        for(int j = 0; j< vsize; j++){
            Mat img;
            src[i * vsize + j].copyTo(img);
            tmp.push_back(img);
        }
        dst -> push_back(tmp);
    }
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
concatenateMat(const vector<vector<Mat> > &vec){
    int subFeatures = vec[0][0].rows * vec[0][0].cols;
    int height = vec[0].size() * subFeatures;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        for(int j = 0; j < vec[i].size(); j++){
            Rect roi = Rect(i, j * subFeatures, 1, subFeatures);
            Mat subView = res(roi);
            Mat ptmat = vec[i][j].reshape(0, subFeatures);
            ptmat.copyTo(subView);
        }
    }
    return res;
}

Mat 
concatenateMat(const vector<Mat> &vec, int matcols){
    vector<vector<Mat> > temp;
    unconcatenateMat(vec, &temp, vec.size() / matcols);
    return concatenateMat(temp);
}

double 
getLearningRate(const Mat &data){
    // see Yann LeCun's Efficient BackProp, 5.1 A Little Theory
    int nfeatures = data.rows;
    int nsamples = data.cols;
    //covariance matrix = x * x' ./ nfeatures;
    Mat Sigma = data * data.t() / nsamples;
    SVD uwvT = SVD(Sigma);
    Sigma.release();
    return 0.9 / uwvT.w.ATD(0, 0);
}


