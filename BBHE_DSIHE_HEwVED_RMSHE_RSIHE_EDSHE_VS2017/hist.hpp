#ifndef _HIST__
#define _HIST__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <fstream>
using namespace cv;
using namespace std;

/*
	不强调,但是这里的操作均是针对灰度图像，所以不做类型检验

*/
namespace hist {

	typedef Vec<double, 8> Vec8d;
	typedef Vec<double, 9> Vec9d;
	typedef Vec<double, 10> Vec10d;
	typedef Vec<double, 11> Vec11d;
#define ORG_GRAY 0
#define PIXELS 1
#define PROBABILITY 2
#define FACTOR	3
#define SUB_TREE_FLAG 4
#define RANGE 5
#define MIN_RANGE 6
#define MAX_RANGE 7
#define MAPED_GRAY 8
#define SUB_TREE_PROB 9
//计算直方图
void calcHist(Mat src, vector<int>& hist);
template<typename T>	//T是Vec2i/Vec2d/Vec2f....
void calcHist(Mat src, vector<T>& hist) {
	vector<int> histVec;
	calcHist(src, histVec);
	hist.clear();
	for (int i = 0; i < histVec.size(); i++) {
		T t;
		t[0] = i;
		t[1] = histVec[i];
		hist.push_back(t);
	}
}
template<typename T>
int countZero(vector<T>& vec) {
	int count = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i][1] == 0) count++;
	}
	return count;
}
template<typename T>
int countNonZero(vector<T>& vec) {
	int count = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i][1] != 0) count++;
	}
	return count;
}
template<typename T>
void calcRange(vector<T>& vec) {
	//计算动态范围、、EDSHE
	//输入为计算过factor的vec
	double sum_factor = 0.0;
	int subTreeFlag = -1;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i][SUB_TREE_FLAG] != subTreeFlag) {
			subTreeFlag = (int)(vec[i][SUB_TREE_FLAG]);
			sum_factor += vec[i][FACTOR];
		}
	}
	subTreeFlag = -1;
	for (int i = 0; i < vec.size(); i++) {
		
		vec[i][RANGE] = (256 - 1)*vec[i][FACTOR] / sum_factor;
	}
	subTreeFlag = -1;
	double minRange = 0, maxRange = 0;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i][SUB_TREE_FLAG] != subTreeFlag) {
			subTreeFlag = (int)(vec[i][SUB_TREE_FLAG]);
			maxRange += vec[i][RANGE];
		}
		vec[i][MAX_RANGE] = maxRange;
		vec[i][MIN_RANGE] = maxRange - vec[i][RANGE];
	}
}
//输出直方图到文件
void calcHistOutFile(Mat src, string filename);
//计算直方图并输出到文件
void calcHistOutFile(Mat src, vector<int>& hist, string filename);
//初始的直方图均衡算法
void histEqual(Mat src, Mat& dst);
void histEqual(vector<Vec2i>& hist);
//根据映射表重映射图像
void remapMatByMapTable(const vector<int>& hist, Mat src, Mat& dst);
//计算图像的平均亮度
double calcAvgGrayLumin(Mat src);
//计算图像的亮度中值
double calcMidGrayLumin(Mat src);
double calcEntropy(const Mat& src);
//依据直方图计算熵
template<typename T>
double calcEntropy(const vector<T> hist) {

	double sumOfPixels = 0.0, result = 0.0;
	if (hist.size() == 256) {
		vector<T> pHist(hist.size(), 0);
		
		for (int i = 0; i < hist.size(); i++) {
			sumOfPixels += hist[i][1];
			pHist[i][0] = hist[i][0];
		}
		for (int i = 0; i < hist.size(); i++) {
			pHist[i][1] = hist[i][1] / sumOfPixels;
		}
		for (int i = 0; i < pHist.size(); i++) {
			result -= pHist[i][1] == 0 ? 0 : pHist[i][1] * log2(pHist[i][1]);
		}
	} else {
		for (int i = 0; i < hist.size(); i++) {
			result -= hist[i][2] == 0? 0 : hist[i][2] * log2(hist[i][2]);//吐槽：文章里写的公式，log都不标注底是多少，10？e？查来查去又在MATLAB里面测试最后发现是2
		}
	}
	
	return result;
}
//依据直方图计算熵//偷个懒，idx是熵刚好为一半时候的索引
//第一通道为原始亮度、第二通道为像素数、第三通道为概率密度、第四通道为factor_i、第五通道为树结构同一区域标记；
//每增加一层，乘以2，左侧加0，右侧加1.初始为1；//没用Vec5d因为没有5d有6d，多就多吧，万一用到了//还真的。。我很吊
//后来有直接写成了模板,至少2个通道，而且要求浮点类型，但是不做类型安全检测，因为懒
template<typename T>
double calcEntropy(const vector<T> hist, int& idx) {
//	int breakCon = (int)hist[0][0];
	double sumOfPixels = 0.0, result = 0.0, halfEntropy = 0;
	if (hist.size() == 256) {
		vector<T> pHist(hist.size(), 0);

		for (int i = 0; i < hist.size(); i++) {
			sumOfPixels += hist[i][1];
			pHist[i][0] = hist[i][0];
		}
		for (int i = 0; i < hist.size(); i++) {
			pHist[i][1] = hist[i][1] / sumOfPixels;
		}
		for (int i = 0; i < pHist.size(); i++) {
			result -= pHist[i][1] == 0? 0 : pHist[i][1] * log2(pHist[i][1]);
		}
	} else {
		for (int i = 0; i < hist.size(); i++) {
			result -= hist[i][2] == 0? 0 : hist[i][2] * log2(hist[i][2]);//吐槽：文章里写的公式，log都不标注底是多少，10？e？查来查去又在MATLAB里面测试最后发现是2
		}
	}
	for (int i = 0; i < hist.size(); i++) {
		halfEntropy -= hist[i][PROBABILITY] == 0? 0 :hist[i][PROBABILITY] * log2(hist[i][PROBABILITY]);
		if (halfEntropy >= (result / 2.0)) {
			idx = i;
			break;
		}
	}
	if ((idx == -1 || idx == hist.size() - 1) && hist.size() > 2)
		idx = (int)(hist.size() - 2);
	return result;
}
//根据平均亮度划分
void splitInto2ByAvgLumin(vector<Vec2i>& hist, vector<Vec2i>& left, vector<Vec2i>& rght);
//根据概率（像素数）划分
void splitInto2ByHalfArea(vector<Vec2i>& hist, vector<Vec2i>& left, vector<Vec2i>& rght);
//根据熵划分//因为不确定Vector里面究竟放个多少通道数据，所以写成模板。。Orz		//后面发现我这么做的是对的，还很骚
template<typename T>
void splitInto2ByEntropy(vector<T>& hist, vector<T>& left, vector<T>& rght) {
	//
	int idx = -1;
	calcEntropy(hist, idx);
	left.clear();
	rght.clear();
	for (int i = 0; i < hist.size(); i++) {
		if (i <= idx) {
			left.push_back(hist[i]);
		} else {
			rght.push_back(hist[i]);
		}
	}
	return;
}
template<typename T>
void splitVec2VecsBySUB_TREE_FLAG(vector<T>& vec, vector<vector<T>>& vecs) {
	//把vector按照子树划分开，然后分别直方图均衡，最后合并得到映射表，当然，合并的操作不在这里，在下面
	vecs.clear();
	vecs.clear();
	int subTreeFlag = (int)(vec[0][SUB_TREE_FLAG]);
	vector<T> vecTmp;
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i][SUB_TREE_FLAG]  == subTreeFlag) {
			vecTmp.push_back(vec[i]);
		} else {
			subTreeFlag = vec[i][SUB_TREE_FLAG];
			vecs.push_back(vecTmp);
			vecTmp.clear();
			vecTmp.push_back(vec[i]);
		}
		if (i == vec.size() - 1) {
			vecs.push_back(vecTmp);
		}
	}
}
template<typename T>
void mergeVecs2VecBySUB_TREE_FLAG(vector<vector<T>>& vecs, vector<T>& vec) {
	//就是上面说的合并操作
	vec.clear();
	for (int i = 0; i < vecs.size(); i++) {
		vec.insert(vec.end(), vecs[i].begin(), vecs[i].end());
	}
}
template<typename T>
void subTreeHE(vector<T>& vec) {
	//对子树进行直方图均衡化
	if (vec.size() < 2) {
		vec[0][MAPED_GRAY] = vec[0][MAX_RANGE];
	} else if (vec.size() >= 2 ) {
		double sumPixels = 0.0;
		for (int i = 0; i < vec.size(); i++) {
			sumPixels += vec[i][PIXELS];
		}
		for (int i = 0; i < vec.size(); i++) {
			vec[i][SUB_TREE_PROB] = vec[i][PIXELS] / sumPixels;
		}
		for (int i = 1; i < vec.size(); i++) {
			vec[i][SUB_TREE_PROB] += vec[i - 1][SUB_TREE_PROB];
		}
		for (int i = 0; i < vec.size(); i++) {
			vec[i][MAPED_GRAY] = vec[i][MIN_RANGE] + vec[i][RANGE] * vec[i][SUB_TREE_PROB];
		}
	} else {
		//讲道理不应该进入这里
		std::cout << "SOMETHING WRONG!!!!!!!!!" << std::endl;
	}

}
void rmshe_iter(vector<Vec2i>& hist, int iterNum);
void rsihe_iter(vector<Vec2i>& hist, int iterNum);
template<typename T>
int findMinAvailable(vector<T>& hist) {
	if (hist.size() == 1) {
		return 0;
	}//凭手感写的，这一段可以不要的
	for (int i = 0; i < hist.size(); i++) {
		if (hist[i][PIXELS] != 0) return i;
	}
	return -1;
}
template<typename T>
int findMaxAvailable(vector<T>& hist) {
	if (hist.size() == 1) {
		return 0;
	}//凭手感写的，这一段可以不要的
	for (int i = (int)(hist.size() - 1); i >= 0; i--) {
		if (hist[i][PIXELS] != 0) return i;
	}
	return -1;
}
template<typename T>
void edshe_iter(vector<T>& hist, int iter) {
	//摊手
	vector<T> left, rght;
	
	int span = findMaxAvailable(hist) - findMinAvailable(hist);
	if ( span >= 2) {
		splitInto2ByEntropy(hist, left, rght);
		edshe_iter(left, iter * 2);
		edshe_iter(rght, iter * 2 + 1);

		hist.clear();
		hist.insert(hist.end(), left.begin(), left.end());
		hist.insert(hist.end(), rght.begin(), rght.end());
	} else {
		//其实这里究竟是该在上面写还是下面写，文章里没有描述的很清楚。。印度人。。stop bibi show me the code
		double entropy = calcEntropy(hist);
		double nonZeroCount = countNonZero(hist);
		double absTop = abs(2 * nonZeroCount - hist.size()), absDown = pow(nonZeroCount, 1.5);
		double factor = span * entropy * (absTop / absDown);
		for (int i = 0; i < hist.size(); i++) {
			hist[i][FACTOR] = factor;
			hist[i][SUB_TREE_FLAG] = iter;
			//std::cout << i << " :\t" << hist[i][0] << " " << hist[i][1] << " " << hist[i][2] << " " << hist[i][3] << " " << hist[i][4] << " " << hist[i][5] << std::endl;
		}
	}
}

//Contrast enhancement using brightness preserving bi-histogram equalization	3
void BBHE(Mat src, Mat& dst);
//Image enhancement based on equal area dualistic sub-image histogram equalization method	  4
void DSIHE(Mat src, Mat& dst);
//2010-Modified histogram equalization for image contrast enhancement	9
void HEwVED(Mat src, Mat& dst, double a);
void HEwVED(Mat src, Mat& dst, int a);
//Contrast enhancement using recursive mean-separate histogram equalization for scalable brightness preservation	6
void RMSHE(Mat src, Mat& dst, int iterator_s);
//Recursive sub-image histogram equalization applied to gray scale images	7
void RSIHE(Mat src, Mat& dst, int iterator_s);
//2016-Contrast Enhancement using Entropy based Dynamic Sub-Histogram Equalization
void EDSHE(Mat src, Mat& dst);

//下面写点测试数据的函数
//路径、文件名、数据表
void writeRemapTable(string path, string tableName, const vector<int>& hist);
}
#endif