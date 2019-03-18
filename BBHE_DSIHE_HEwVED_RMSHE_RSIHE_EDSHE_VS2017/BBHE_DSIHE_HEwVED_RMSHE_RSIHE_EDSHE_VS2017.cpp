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
//д�������ȫ�ֱ�����ȫ����Ϊ������Զ�д��������·�����㡢������

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
	ORG_FILE_PATH = ORG_FILE_PATH.substr(3);//������Ŀ¼Ҫɾ��ǰ�������ַ�
	CURRENT_PATH = ORG_FILE_PATH;
	hist::calcHistOutFile(srcPic, ORG_FILE_PATH + "\\Org_Hist.txt");
	imwrite(ORG_FILE_PATH + "\\ORG_PNG.png", srcPic, compression_params);

	outFile.open(CURRENT_PATH + "\\������.txt");
	outFile << "%ԭʼͼ�Ҷ�������ƽ������" << endl;
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

	outFile.open(CURRENT_PATH + "\\������.txt");
	outFile << "%����-�Ҷ�����-��-ƽ������-DEN-GMSD" << endl;//GMSDĿǰ��ûд2018��11��5��11:03:41
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) <<" " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("HE����");
	imshow("HE����", dst);
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

	outFile.open(CURRENT_PATH + "\\������.txt");
	outFile << "%����-�Ҷ�����-��-ƽ������-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("BBHE����");
	imshow("BBHE����", dst);
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

	outFile.open(CURRENT_PATH + "\\������.txt");
	outFile << "% ����-�Ҷ�����-��-ƽ������-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("DSIHEHE����");
	imshow("DSIHEHE����", dst);
#endif
	
	//******************************************************************************************************************
	//******************************************************************************************************************
	HEwVED_FILE_PATH = filePath + "\\HEwVED";
	system(HEwVED_FILE_PATH.data());
	HEwVED_FILE_PATH = HEwVED_FILE_PATH.substr(3);
	CURRENT_PATH = HEwVED_FILE_PATH;
	outFile.open(CURRENT_PATH + "\\������_����-�Ҷ�����-��-ƽ������-DEN-GMSD.txt");
	for (int i = 0; i < 256; i++) {
		hist::HEwVED(srcPic, dst, i / 255.0);
		hist::calcHistOutFile(dst, HEwVED_FILE_PATH + "\\HEwVED_Hist_param_" + to_string(i) + ".txt");
		imwrite(HEwVED_FILE_PATH + "\\HEwVED_Hist_param_" + to_string(i) + ".png", dst, compression_params);
		cout << "HEwVED Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
		
#if SHOW_
		namedWindow("HEwVED����");
		imshow("HEwVED����", dst);
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
	outFile.open(CURRENT_PATH + "\\������_����-�Ҷ�����-��-ƽ������-DEN-GMSD.txt");
	for (int i = 0; i < 9; i++) {
		hist::RMSHE(srcPic, dst, i);
		hist::calcHistOutFile(dst, RMSHE_FILE_PATH + "\\RMSHE_Hist_param_"+ to_string(i) + ".txt");
		imwrite(RMSHE_FILE_PATH + "\\RMSHE_Hist_param_" + to_string(i) + ".png", dst, compression_params);
		cout << "RMSHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;

#if SHOW_
		namedWindow("RMSHE����");
		imshow("RMSHE����", dst);
#endif
	}
	outFile.close();
	//******************************************************************************************************************
	//******************************************************************************************************************
	
	RSIHE_FILE_PATH = filePath + "\\RSIHE";
	system(RSIHE_FILE_PATH.data());
	RSIHE_FILE_PATH = RSIHE_FILE_PATH.substr(3);
	CURRENT_PATH = RSIHE_FILE_PATH;
	outFile.open(CURRENT_PATH + "\\������_����-�Ҷ�����-��-ƽ������-DEN-GMSD.txt");
	for (int i = 0; i < 9; i++) {
		hist::RSIHE(srcPic, dst, i);
		hist::calcHistOutFile(dst, RSIHE_FILE_PATH + "\\RSIHE_Hist_param_" + to_string(i) + ".txt");
		imwrite(RSIHE_FILE_PATH + "\\RSIHE_Hist_param_" + to_string(i) + ".png", dst, compression_params);

		cout << "RSIHE Src avg:" << hist::calcAvgGrayLumin(srcPic) << " " << hist::calcAvgGrayLumin(dst) << endl;

		outFile << i << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
		
#if SHOW_
		namedWindow("RSIHE����");
		imshow("RSIHE����", dst);
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

	outFile.open(CURRENT_PATH + "\\������.txt");
	outFile << "%����-�Ҷ�����-��-ƽ������-DEN-GMSD" << endl;
	outFile << 0 << " " << calcGrayNum(dst) << " " << hist::calcEntropy(dst) << " " << hist::calcAvgGrayLumin(dst) << " " << DEN(srcPic, dst) << endl;
	outFile.close();

#if SHOW_
	namedWindow("EDSHE����");
	imshow("EDSHE����", dst);
	waitKey(0);
#endif
	//******************************************************************************************************************
}

