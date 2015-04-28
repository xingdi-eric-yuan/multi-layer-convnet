#include "read_data.h"

using namespace cv;
using namespace std;

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
read_MNIST_data(vector<Mat> &trainX, vector<Mat> &testX, Mat &trainY, Mat &testY){
    


    readData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 60000);
    readData(testX, testY, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", 10000);
    preProcessing(trainX, testX);
    dataEnlarge(trainX, trainY);

    cout<<"****************************************************************************"<<endl
        <<"**                        READ DATASET COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;
    cout<<"The training data has "<<trainX.size()<<" images, each images has "<<trainX[0].cols<<" columns and "<<trainX[0].rows<<" rows."<<endl;
    cout<<"The testing data has "<<testX.size()<<" images, each images has "<<testX[0].cols<<" columns and "<<testX[0].rows<<" rows."<<endl;
    cout<<"There are "<<trainY.cols<<" training labels and "<<testY.cols<<" testing labels."<<endl<<endl;
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
        file.close();
    }
}

void
readData(vector<Mat> &x, Mat &y, string xpath, string ypath, int number_of_images){

    //read MNIST iamge into OpenCV Mat vector
    read_Mnist(xpath, x);
    for(int i = 0; i < x.size(); i++){
        x[i].convertTo(x[i], CV_64FC1, 1.0/255, 0);
    }
    //read MNIST label into double vector
    y = Mat::zeros(1, number_of_images, CV_64FC1);
    read_Mnist_Label(ypath, y);
}

Mat 
concat(const vector<Mat> &vec){
    
    int height = vec[0].rows * vec[0].cols;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        Rect roi = Rect(i, 0, 1, height);
        Mat subView = res(roi);
        Mat ptmat = vec[i].reshape(0, height);
        ptmat.copyTo(subView);
    }
    return res;
}

void
preProcessing(vector<Mat> &trainX, vector<Mat> &testX){
    /*
    for(int i = 0; i < trainX.size(); i++){
        trainX[i].convertTo(trainX[i], CV_64FC1, 1.0/255, 0);
    }
    for(int i = 0; i < testX.size(); i++){
        testX[i].convertTo(testX[i], CV_64FC1, 1.0/255, 0);
    }
    */
    // first convert vec of mat into a single mat
    Mat tmp = concat(trainX);
    Mat tmp2 = concat(testX);
    Mat alldata = Mat::zeros(tmp.rows, tmp.cols + tmp2.cols, CV_64FC1);
    
    tmp.copyTo(alldata(Rect(0, 0, tmp.cols, tmp.rows)));
    tmp2.copyTo(alldata(Rect(tmp.cols, 0, tmp2.cols, tmp.rows)));

    Scalar mean;
    Scalar stddev;
    meanStdDev (alldata, mean, stddev);

    for(int i = 0; i < trainX.size(); i++){
        divide(trainX[i] - mean[0], stddev[0], trainX[i]);
    }
    for(int i = 0; i < testX.size(); i++){
        divide(testX[i] - mean[0], stddev[0], testX[i]);
    }
    tmp.release();
    tmp2.release();
    alldata.release();
}





