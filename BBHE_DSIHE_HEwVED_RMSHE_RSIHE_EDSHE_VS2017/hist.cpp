#include "hist.hpp"
#include <numeric>
#include <iomanip>
/*
	2018年10月28日15:13:50
	这里其实做了很多多余的工作，需要重构，但是，
	写着玩的，知道就行了，哪那么多事。	--	Quasimo
*/
#define _PRINT_OUT_ 1
int _WRITE_REMAP_TABLE_ = 1;

extern string ORG_FILE_PATH;
extern string HE_FILE_PATH;
extern string BBHE_FILE_PATH;
extern string DSIHE_FILE_PATH;
extern string HEwVED_FILE_PATH;
extern string RMSHE_FILE_PATH;
extern string RSIHE_FILE_PATH;
extern string EDSHE_FILE_PATH;
extern string CURRENT_PATH;


namespace hist{


void calcHist(Mat src, vector<int>& hist) {
	//
	vector<int> hist_t(256, 0);
	hist.clear();
	hist.resize(hist_t.size());
	copy(hist_t.begin(), hist_t.end(), hist.begin());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			hist[src.at<uchar>(i, j)]++;
		}
	}
	return;
}

void calcHistOutFile(Mat src, string filename) {
	vector<int> hist;
	calcHistOutFile(src, hist, filename);
}
void calcHistOutFile(Mat src, vector<int>& hist, string filename) {
	calcHist(src, hist);

	ofstream outFile(filename);
	for (int i = 0; i < hist.size(); i++) {
		outFile << i << " " << hist[i] << endl;
	}
	outFile.close();

}
void histEqual(Mat src, Mat& dst) {
	//
	if (src.type() == CV_8UC1){
		vector<int> hist;
		vector<int> accum(256, 0);
		vector<double> pAccum(256, 0.0);
		calcHist(src, hist);
#if _PRINT_OUT_
		for (int i = 0; i < hist.size(); i++) {
			std::cout << hist[i] << " ";
		}
		std::cout << "\n" << std::endl;
#endif
		int accum_n = 0;
		for (int i = 0; i < hist.size(); ++i) {
			hist[i] = hist[i] + accum_n;
			accum_n = hist[i];
		}
	#if _PRINT_OUT_
		for (int i = 0; i < hist.size(); i++) {
			std::cout << hist[i] << " ";
		}
		std::cout << "\n" << std::endl;
	#endif
		for (int i = 0; i < hist.size(); i++) {
			pAccum[i] = hist[i]*1.0 / (src.cols * src.rows);
			hist[i] = (int)((256 - 1) * pAccum[i]);
		}
	#if _PRINT_OUT_
		for (int i = 0; i < hist.size(); i++) {
			std::cout << pAccum[i] << " ";
		}
		std::cout << "\n" << std::endl;
	#endif
	#if _PRINT_OUT_
		for (int i = 0; i < hist.size(); i++) {
			std::cout << i << ":" << hist[i] << " ";
		}
		std::cout << "\n" << std::endl;
	#endif
		if(_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "HE_REMAP_TABLE.txt", hist);
		remapMatByMapTable(hist, src, dst);
		return;
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		histEqual(matHSVs[2], matHSVs[2]);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
}

void equalization(vector<int> & vec, int base) {
	vector<int> accum(vec);
	vector<double> pAccum(vec.size(), 0);
	double sumP = accumulate(accum.begin(), accum.end(), 0);
#if _PRINT_OUT_
	for (int i = 0; i < accum.size(); i++) {
		std::cout << accum[i] << " ";
	}
	std::cout << std::endl;
#endif
	for (int i = 1; i < accum.size(); i++) {
		accum[i] += accum[i - 1];
	}
#if _PRINT_OUT_
	for (int i = 0; i < accum.size(); i++) {
		std::cout << accum[i] << " ";
	}
	std::cout << std::endl;
#endif
	for (int i = 0; i < accum.size(); i++) {
		pAccum[i] = accum[i] / sumP;
		vec[i] = (int)((vec.size() - 1) * pAccum[i] + base);
	}
}
void equalization(vector<int> & vec) {
	equalization(vec, 0);
}
void histEqual(vector<Vec2i>& hist) {
	//第一个通道是原始亮度，第二个通道是像素数
	vector<Vec2d> accum(hist.size(), Vec2d(0, 0));
	for (int i = 0; i < hist.size(); i++) {
		Vec2d t;
		t[0] = hist[i][0];
		t[1] = hist[i][1];
		accum[i] = t;
	}
	for (int i = 1; i < accum.size(); i++) {
		accum[i][1] += accum[i - 1][1];
	}
	for (int i = 0; i < accum.size(); i++) {
		accum[i][1] /= (accum.back())[1];
	}
	for (int i = 0; i < accum.size(); i++) {
		accum[i][1] = (int)(accum[i][1] * (accum.size() - 1) + accum[0][0]);
	}
#if 0
	for (int i = 0; i < accum.size(); i++) {
		std::cout << accum[i][0] << " " << accum[i][1] << std::endl;
	}
#endif
	hist.clear();
	hist.insert(hist.begin(), accum.begin(), accum.end());
}
void remapMatByMapTable(const vector<int>& hist, Mat src, Mat& dst) {
	//这里只考虑灰度图？
	if (!src.data) {
		//传入图为空
		return;
	}
	dst = cv::Mat(src.size(), src.type(), cv::Scalar::all(0));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = hist[src.at<uchar>(i, j)];
		}
	}
}

double calcAvgGrayLumin(Mat src) {
	//计算平均亮度
	//输入的图像为灰度图
	double accum = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			accum += src.ptr<uchar>(i)[j];
		}
	}
	return accum / (src.rows * src.cols);
}


void BBHE(Mat src, Mat& dst) {
	//
	if (src.type() == CV_8UC1) {
		double avgLumin = calcAvgGrayLumin(src);
		vector<int> histVec;
		calcHist(src, histVec);
		vector<int> lowPart, highPart;
		for (int i = 0; i < histVec.size(); i++) {
			if (i <= avgLumin) {
				lowPart.push_back(histVec[i]);
			} else {
				highPart.push_back(histVec[i]);
			}
		}
		equalization(lowPart);
		equalization(highPart, (int)avgLumin);
		histVec.clear();
		histVec.insert(histVec.end(), lowPart.begin(), lowPart.end());
		histVec.insert(histVec.end(), highPart.begin(), highPart.end());
		if(_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "BBHE_REMAP_TABLE.txt", histVec);

		remapMatByMapTable(histVec, src, dst);
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		BBHE(matHSVs[2], matHSVs[2]);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
	
}
double calcMidGrayLumin(Mat src) {
	//
	vector<int> hist;
	calcHist(src, hist);
	double sum = accumulate(hist.begin(), hist.end(), 0);
	double tmp = 0;
	int i = 0;
	for (; i < hist.size(); i++) {
		tmp += hist[i];
		if (tmp > sum / 2) {
			break;
		}
	}
	tmp = 0;
	int j = (int)(hist.size() - 1);
	for (; j > 0; j--) {
		tmp += hist[j];
		if (tmp > sum / 2) {
			break;
		}
	}
	return (i + j) / 2.0;
}
void DSIHE(Mat src, Mat& dst) {
	if (src.type() == CV_8UC1) {
		double midLumin = calcMidGrayLumin(src);
		vector<int> histVec;
		calcHist(src, histVec);
		vector<int> lowPart, highPart;
		for (int i = 0; i < histVec.size(); i++) {
			if (i <= midLumin) {
				lowPart.push_back(histVec[i]);
			} else {
				highPart.push_back(histVec[i]);
			}
		}
		equalization(lowPart);
		equalization(highPart, (int)midLumin);
		histVec.clear();
		histVec.insert(histVec.end(), lowPart.begin(), lowPart.end());
		histVec.insert(histVec.end(), highPart.begin(), highPart.end());
		if(_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "DSIHE_REMAP_TABLE.txt", histVec);

		remapMatByMapTable(histVec, src, dst);
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		DSIHE(matHSVs[2], matHSVs[2]);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
}

void HEwVED(Mat src, Mat& dst, double a) {
	_WRITE_REMAP_TABLE_ = 0;	//这种方法本质上不算重映射灰度	//于是取消输出映射表
	if (src.type() == CV_8UC1) {
		Mat tmp;
		histEqual(src, tmp);
		dst = (1 - a) * src + a * tmp;
		dst.convertTo(dst, CV_8UC1);
		return;
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		HEwVED(matHSVs[2], matHSVs[2], a);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
	_WRITE_REMAP_TABLE_ = 1;
}
void HEwVED(Mat src, Mat& dst, int a) {
	HEwVED(src, dst, a / 255.0);
}

void splitInto2ByHalfArea(vector<Vec2i>& hist, vector<Vec2i>& left, vector<Vec2i>& rght) {
	//第一个通道是原始亮度，第二个通道是像素数
	double sum = 0;
	for (int i = 0; i < hist.size(); i++) {
		sum += hist[i][1];
	}
	left.clear();
	rght.clear();
	int i = 0, j = 0, tmp = 0;
	for (; i < hist.size(); ++i) {
		tmp += hist[i][1];
		if (tmp >= sum / 2.0) {
			break;
		}
	}
	tmp = 0;
	for (j = (int)(hist.size() - 1);  j > 0 ;  j--) {
		tmp += hist[j][1];
		if (tmp >= sum / 2.0) {
			break;
		}
	}
	int mid = (int)((i + j) / 2.0);
	for (int k = 0; k < hist.size(); k++) {
		if (k < mid) {
			left.push_back(hist[k]);
		} else {
			rght.push_back(hist[k]);
		}
	}

}

void splitInto2ByAvgLumin(vector<Vec2i>& hist, vector<Vec2i>& left, vector<Vec2i>& rght) {
	//
	double avglumin = 0, sum_ = 0, pixelNum = 0;
	//vector<Vec2i> left_, rght_;
	for (int i = 0; i < hist.size(); i++) {
		sum_ += hist[i][0] * hist[i][1];
		pixelNum += hist[i][1];
	}
	avglumin = sum_ / pixelNum;
	left.clear();
	rght.clear();
	for (int i = 0; i < hist.size(); i++) {
		if (hist[i][0] <= avglumin) {
			left.push_back(hist[i]);
		} else {
			rght.push_back(hist[i]);
		}
	}
}

void rmshe_iter(vector<Vec2i>& hist, int iterNum) {
	//
	if (iterNum == 0) {
		//
		histEqual(hist);
		return;
	}
	vector<Vec2i> left, rght;
	splitInto2ByAvgLumin(hist, left, rght);
	rmshe_iter(left, iterNum - 1);
	rmshe_iter(rght, iterNum - 1);
	hist.clear();
	hist.insert(hist.end(), left.begin(), left.end());
	hist.insert(hist.end(), rght.begin(), rght.end());
}
void rsihe_iter(vector<Vec2i>& hist, int iterNum) {
	//
	if (iterNum == 0) {
		//
		histEqual(hist);
		return;
	}
	vector<Vec2i> left, rght;
	splitInto2ByHalfArea(hist, left, rght);
	rsihe_iter(left, iterNum - 1);
	rsihe_iter(rght, iterNum - 1);
	hist.clear();
	hist.insert(hist.end(), left.begin(), left.end());
	hist.insert(hist.end(), rght.begin(), rght.end());
}
void RMSHE(Mat src, Mat& dst, int iterator_s) {
	//
	if (src.type() == CV_8UC1){
		vector<Vec2i> hist2i;
		vector<int> histVec;
		calcHist(src, histVec);
		//第一个通道是原始亮度，第二个通道是像素数
		for (int i = 0; i < histVec.size(); i++) {
			Vec2i t;
			t[0] = i;
			t[1] = histVec[i];
			hist2i.push_back(t);
		}
	
	#if 0
		for (int i = 0; i < hist2i.size(); i++) {
			std::cout << hist2i[i][0] << " " << hist2i[i][1] << std::endl;
		}
	#endif

		rmshe_iter(hist2i, iterator_s);
		vector<int> hist;
		for (int i = 0; i < hist2i.size(); i++) {
			//
			hist.push_back(hist2i[i][1]);
		}
		_WRITE_REMAP_TABLE_ = 1;
		if(_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "RMSHE_REMAP_TABLE_param_" + to_string(iterator_s) + ".txt", hist);
		remapMatByMapTable(hist, src, dst);
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		RMSHE(matHSVs[2], matHSVs[2], iterator_s);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
}
void RSIHE(Mat src, Mat& dst, int iterator_s) {
	//
	if (src.type() == CV_8UC1){
		vector<Vec2i> hist2i;
		vector<int> histVec;
		calcHist(src, histVec);
		//第一个通道是原始亮度，第二个通道是像素数
		for (int i = 0; i < histVec.size(); i++) {
			Vec2i t;
			t[0] = i;
			t[1] = histVec[i];
			hist2i.push_back(t);
		}

	#if 0
		for (int i = 0; i < hist2i.size(); i++) {
			std::cout << hist2i[i][0] << " " << hist2i[i][1] << std::endl;
		}
	#endif

		rsihe_iter(hist2i, iterator_s);
	#if 1
		for (int i = 0; i < hist2i.size(); i++) {
			std::cout << hist2i[i][0] << " " << hist2i[i][1] << std::endl;
		}
	#endif
		vector<int> hist;
		for (int i = 0; i < hist2i.size(); i++) {
			//
			hist.push_back(hist2i[i][1]);
		}
		_WRITE_REMAP_TABLE_ = 1;
		if (_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "RSIHE_REMAP_TABLE_param_" + to_string(iterator_s) + ".txt", hist);
		remapMatByMapTable(hist, src, dst);
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		RSIHE(matHSVs[2], matHSVs[2], iterator_s);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
}
void EDSHE(Mat src, Mat& dst) {
	if (src.type() == CV_8UC1){
		vector<Vec2i> hist_;
		calcHist(src, hist_);
	//	std::cout << "\nEntropy: " << calcEntropy(hist) << std::endl;
		vector<Vec10d> hist(hist_.size(), Vec10d(0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0));
		vector<vector<Vec10d>> hists;
		for (int i = 0; i < hist_.size(); i++) {
			hist[i][0] = hist_[i][0];
			hist[i][1] = hist_[i][1];
			hist[i][2] = hist[i][1] / (src.cols * src.rows * 1.0);
		//	std::cout << i << " :\t" << hist[i][0] << " " << hist[i][1] << " " << hist[i][2] << std::endl;
		}
		//system("pause");
		//第一通道为原始亮度、第二通道为像素数、第三通道为概率密度、第四通道为factor_i、第五通道为树结构同一区域标记；
		//每增加一层，乘以2，左侧加0，右侧加1.初始为1；
	#if 0
		for (int i = 0; i < hist.size(); i++) {
			std::cout << i << " :" << std::setw(9) << hist[i][0] << "\t" << std::setw(9) << hist[i][1] << "\t" << std::setw(9) << hist[i][2] << "\t" << std::setw(9) << hist[i][FACTOR] << "\t" << std::setw(9) << hist[i][RANGE] << "\t" << std::setw(9) << hist[i][SUB_TREE_FLAG] << "\tRANGE:" << std::setw(9) << hist[i][MIN_RANGE] << "\t" << std::setw(9) << hist[i][MAX_RANGE] << "\t" << std::setw(9) << hist[i][MAPED_GRAY] << std::endl;
		}
	#endif
		edshe_iter(hist, 1);
		calcRange(hist);

		splitVec2VecsBySUB_TREE_FLAG(hist, hists);
		for (int i = 0; i < hists.size(); i++) {
			subTreeHE(hists[i]);
		}
		mergeVecs2VecBySUB_TREE_FLAG(hists, hist);
	#if 1
		for (int i = 0; i < hist.size(); i++) {
			std::cout << i << " :" << std::setw(9) << hist[i][0] << "\t" << std::setw(9) << hist[i][1] << "\t" << std::setw(9) << hist[i][2] << "\t" << std::setw(9) << hist[i][FACTOR] << "\t" << std::setw(9) << hist[i][RANGE] << "\t" << std::setw(9) << hist[i][SUB_TREE_FLAG] << "\tRANGE:" << std::setw(9) << hist[i][MIN_RANGE] << "\t" << std::setw(9) << hist[i][MAX_RANGE] << "\t" << std::setw(9) << hist[i][MAPED_GRAY] << std::endl;
		}
	#endif
		vector<int> map_hist(256, 0);
		for (int i = 0; i < hist.size(); i++) {
			map_hist[i] = (int)(hist[i][MAPED_GRAY]);
	#if 0
			std::cout << i << " " << map_hist[i] << std::endl;
	#endif
		}
		_WRITE_REMAP_TABLE_ = 1;
		if (_WRITE_REMAP_TABLE_)
			writeRemapTable(CURRENT_PATH, "EDSHE_REMAP_TABLE.txt", map_hist);
		remapMatByMapTable(map_hist, src, dst);
		//dst = src.clone();
	} else if (src.type() == CV_8UC3) {
		//彩色模型
		Mat matHSV, matHSVs[3];
		cvtColor(src, matHSV, COLOR_BGR2HSV);
		split(matHSV, matHSVs);
		EDSHE(matHSVs[2], matHSVs[2]);
		merge(matHSVs, 3, matHSV);
		cvtColor(matHSV, dst, COLOR_HSV2BGR);
	}
}

void writeRemapTable(string path, string tableName, const vector<int>& hist) {
	ofstream outFile(path + "\\" + tableName);
	for (int i = 0; i < hist.size(); i++) {
		outFile << i << " " << hist[i] << endl;
	}
	outFile.close();	//突然发现这里很像上面写的输出直方图，但是还有一点点不一样，其实写成通用的就是输出vector到文本
}
double calcEntropy(const Mat& src) {
	//根据图像计算熵
	vector<Vec2d> hist;
	calcHist(src, hist);
	return calcEntropy(hist);
}
}