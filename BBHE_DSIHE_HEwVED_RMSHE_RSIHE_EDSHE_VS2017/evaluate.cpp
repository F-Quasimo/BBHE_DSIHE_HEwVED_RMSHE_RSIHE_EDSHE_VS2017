#include "evaluate.h"

double DEN(const Mat& src, const Mat& dst) {
	//�������DEN��һ�����۱�׼���ο�2016-Contrast Enhancement using Entropy based Dynamic Sub-Histogram Equalization ��25����ȡֵ��0~1���ߵ�ֵ��ʾ���õĶԱȶȡ�
	return 1 / (1 + (log2(256) - DE_(dst) / log2(256) - DE_(src)));
}
double DE_(const Mat& src) {
	//ͬ��
	vector<Vec2d> accum(256, Vec2d(0, 0));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			accum[src.at<uchar>(i, j)][0]++;
		}//�ⲿ����������ʵ�ⲿ�ֿ���ֱ����calcHist��������������ˣ������Ϊʲô���������Ҫ�ع��ع���������д��ʱ���Ҫ������õ�ԭ��ģ���޵�
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
