#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <fstream>
#include "hist.hpp"
#include <String>
#include "string.h"
#include "evaluate.h"

using namespace cv;
using namespace std;
#define SHOW_ 0

string ORG_FILE_PATH;
string HE_FILE_PATH;
string BBHE_FILE_PATH;
string DSIHE_FILE_PATH;
string HEwVED_FILE_PATH;
string RMSHE_FILE_PATH;
string RSIHE_FILE_PATH;
string EDSHE_FILE_PATH;
string CURRENT_PATH;
//写在这里成全局变量完全是因为后面测试读写测试数据路径方便、、垃圾

int main(int argc, char* argv[]) {
	ofstream outFile;

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);
	string fileName = "seed.tif";
	if (argc == 2) {
		fileName = string(argv[1]);
	}

	string filePath = fileName.substr(0, fileName.find('.', 0));
	filePath = "md " + filePath + "Test";
	system(filePath.data());

	Mat srcPic = imread(fileName, cv::ImreadModes::IMREAD_GRAYSCALE);
	Mat dst;
	std::cout << fileName << std::endl;
	/**/
	ORG_FILE_PATH = filePath + "\\ORG";
	system(ORG_FILE_PATH.data());
	ORG_FILE_PATH = ORG_FILE_PATH.substr(3);//创建完目录要删掉前面三个字符
	CURRENT_PATH = ORG_FILE_PATH;
	hist::calcHistOutFile(srcPic, ORG_FILE_PATH + "\\Org_Hist.txt");
	imwrite(ORG_FILE_PATH + "\\ORG_PNG.png", srcPic, compression_params);

	outFile.open(CURRENT_PATH + "\\评估表.txt");
	outFile << "%原始图灰度数量；平均亮度" << endl;
	outFile << calcGrayNum(srcPic) << " " <<hist::calcAvgGrayLumin(srcPic) << endl;
	outFile.close();

	//vector<int> hist;
	//calcHist(srcPic, hist);
	
	//******************************************************************************************************************
	HE_FILE_PATH = filePath + "\\HE";
	system(HE_FILE_PATH.data());
	HE_FILE_PATH = HE_FILE_PATH.substr(3);
	CURRENT_PATH = HE_FILE_PATH;
	hist::histEqual(srcPic, dst);
	hist::calcHistOutFile(dst, HE_FILE_PATH + "\\HE_Hist.txt");
	imwrite(HE_FILE_PATH + "\\HE_PNG.png", dst, compression_params);
	cout << "HE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

	outFile.open(CURRENT_PATH + "\\评估表.txt");
	outFile << "%参数-灰度数量-熵-平均亮度-DEN-GMSD" << endl;//GMSD目前还没写2018年11月5日11:03:41
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) <<" " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("HE看看");
	imshow("HE看看", dst);
#endif
	//******************************************************************************************************************
	//******************************************************************************************************************
	BBHE_FILE_PATH = filePath + "\\BBHE";
	system(BBHE_FILE_PATH.data());
	BBHE_FILE_PATH = BBHE_FILE_PATH.substr(3);
	CURRENT_PATH = BBHE_FILE_PATH;
	hist::BBHE(srcPic, dst);
	hist::calcHistOutFile(dst, BBHE_FILE_PATH + "\\BBHE_Hist.txt");
	imwrite(BBHE_FILE_PATH + "\\BBHE_PNG.png", dst, compression_params);
	cout << "BBHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

	outFile.open(CURRENT_PATH + "\\评估表.txt");
	outFile << "%参数-灰度数量-熵-平均亮度-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("BBHE看看");
	imshow("BBHE看看", dst);
#endif
	//******************************************************************************************************************
	//******************************************************************************************************************
	DSIHE_FILE_PATH = filePath + "\\DSIHE";
	system(DSIHE_FILE_PATH.data());
	DSIHE_FILE_PATH = DSIHE_FILE_PATH.substr(3);
	CURRENT_PATH = DSIHE_FILE_PATH;
	hist::DSIHE(srcPic, dst);
	hist::calcHistOutFile(dst, DSIHE_FILE_PATH + "\\DSIHE_Hist.txt");
	imwrite(DSIHE_FILE_PATH + "\\DSIHE_PNG.png", dst, compression_params);
	cout << "DSIHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

	outFile.open(CURRENT_PATH + "\\评估表.txt");
	outFile << "% 参数-灰度数量-熵-平均亮度-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("DSIHEHE看看");
	imshow("DSIHEHE看看", dst);
#endif
	
	//******************************************************************************************************************
	//******************************************************************************************************************
	HEwVED_FILE_PATH = filePath + "\\HEwVED";
	system(HEwVED_FILE_PATH.data());
	HEwVED_FILE_PATH = HEwVED_FILE_PATH.substr(3);
	CURRENT_PATH = HEwVED_FILE_PATH;
	outFile.open(CURRENT_PATH + "\\评估表_参数-灰度数量-熵-平均亮度-DEN-GMSD.txt");
	for (int i = 0; i < 256; i++) {
		hist::HEwVED(srcPic, dst, i / 255.0);
		hist::calcHistOutFile(dst, HEwVED_FILE_PATH + "\\HEwVED_Hist_param_" + to_string(i) + ".txt");
		imwrite(HEwVED_FILE_PATH + "\\HEwVED_Hist_param_" + to_string(i) + ".png", dst, compression_params);
		cout << "HEwVED Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
		
#if SHOW_
		namedWindow("HEwVED看看");
		imshow("HEwVED看看", dst);
		waitKey(100);
#endif

	}
	outFile.close();
	//******************************************************************************************************************
	//******************************************************************************************************************

	RMSHE_FILE_PATH = filePath + "\\RMSHE";
	system(RMSHE_FILE_PATH.data());
	RMSHE_FILE_PATH = RMSHE_FILE_PATH.substr(3);
	CURRENT_PATH = RMSHE_FILE_PATH;
	outFile.open(CURRENT_PATH + "\\评估表_参数-灰度数量-熵-平均亮度-DEN-GMSD.txt");
	for (int i = 0; i < 9; i++) {
		hist::RMSHE(srcPic, dst, i);
		hist::calcHistOutFile(dst, RMSHE_FILE_PATH + "\\RMSHE_Hist_param_"+ to_string(i) + ".txt");
		imwrite(RMSHE_FILE_PATH + "\\RMSHE_Hist_param_" + to_string(i) + ".png", dst, compression_params);
		cout << "RMSHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;

#if SHOW_
		namedWindow("RMSHE看看");
		imshow("RMSHE看看", dst);
#endif
	}
	outFile.close();
	//******************************************************************************************************************
	//******************************************************************************************************************
	
	RSIHE_FILE_PATH = filePath + "\\RSIHE";
	system(RSIHE_FILE_PATH.data());
	RSIHE_FILE_PATH = RSIHE_FILE_PATH.substr(3);
	CURRENT_PATH = RSIHE_FILE_PATH;
	outFile.open(CURRENT_PATH + "\\评估表_参数-灰度数量-熵-平均亮度-DEN-GMSD.txt");
	for (int i = 0; i < 9; i++) {
		hist::RSIHE(srcPic, dst, i);
		hist::calcHistOutFile(dst, RSIHE_FILE_PATH + "\\RSIHE_Hist_param_" + to_string(i) + ".txt");
		imwrite(RSIHE_FILE_PATH + "\\RSIHE_Hist_param_" + to_string(i) + ".png", dst, compression_params);

		cout << "RSIHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
		
#if SHOW_
		namedWindow("RSIHE看看");
		imshow("RSIHE看看", dst);
#endif
	}
	outFile.close();
	
	//******************************************************************************************************************
	//******************************************************************************************************************
	
	EDSHE_FILE_PATH = filePath + "\\EDSHE";
	system(EDSHE_FILE_PATH.data());
	EDSHE_FILE_PATH = EDSHE_FILE_PATH.substr(3);
	CURRENT_PATH = EDSHE_FILE_PATH;
	hist::EDSHE(srcPic, dst);
	hist::calcHistOutFile(dst, "./" + EDSHE_FILE_PATH + "\\EDSHEHist.txt");
	imwrite(EDSHE_FILE_PATH + "\\EDSHE_Hist.png", dst, compression_params);
	cout << "EDSHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

	outFile.open(CURRENT_PATH + "\\评估表.txt");
	outFile << "%参数-灰度数量-熵-平均亮度-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("EDSHE看看");
	imshow("EDSHE看看", dst);
	waitKey(0);
#endif
	//******************************************************************************************************************
}

