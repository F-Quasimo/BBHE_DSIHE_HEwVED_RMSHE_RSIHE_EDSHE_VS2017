#include "evaluate.h"

double DEN(const Mat& src, const Mat& dst) {
	//详见文章DEN是一种评价标准，参考2016-Contrast Enhancement using Entropy based Dynamic Sub-Histogram Equalization （25），取值在0~1，高的值表示更好的对比度。
	return 1 / (1 + (log2(256) - DE_(dst) / log2(256) - DE_(src)));
}
double DE_(const Mat& src) {
	//同上
	vector<Vec2d> accum(256, Vec2d(0, 0));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			accum[src.at<uchar>(i, j)][0]++;
		}//这部分垃圾，其实这部分可以直接用calcHist算出来，但是算了，这就是为什么最后总是需要重构重构。。或者写的时候就要想好重用的原因。模板无敌
	}
	for (int i = 0; i < accum.size(); i++) {
		accum[i][1] = accum[i][0] / (src.rows * src.cols);
	}
	double ans = 0;
	for (int i = 0; i < accum.size(); i++) {
		ans -= accum[i][1] == 0 ? 0 : accum[i][1] * log2(accum[i][0]);
	}
	return ans;
}

int calcGrayNum(const Mat& src) {
	vector<int> hist;
	hist::calcHist(src, hist);
	int count = 0;
	for (int i = 0; i < hist.size(); i++) {
		if (hist[i] != 0) {
			count++;
		}
	}
	return count;
}
