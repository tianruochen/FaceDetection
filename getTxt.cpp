#include <opencv2\opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;
using namespace cv;

int main()
{
    Directory dir;
    string basePath = "face";
    string exten = "*";
    bool addPath = true;
    vector<string> fileNames = dir.GetListFiles(basePath, exten, addPath);
    cout << fileNames.size() << endl;
    ofstream outData("train.txt");
    for (int i = 0; i < fileNames.size(); i++)
    {
        cout << fileNames[i] << endl;
        outData << fileNames[i] << " 1" << endl;
    }
    outData.close();
    system("pause");
    return 0;
}