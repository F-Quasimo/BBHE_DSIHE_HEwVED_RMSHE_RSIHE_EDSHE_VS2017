复原了上面描述的多种直方图均衡算法（图像增强算法）用于和我的算法做对比，参考的文章在注释里面有。我自己写的还没传上来。不会用GitHub，传的乱七八糟
复原的文献如下：
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