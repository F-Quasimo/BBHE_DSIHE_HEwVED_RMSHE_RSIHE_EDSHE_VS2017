# BBHE_DSIHE_HEwVED_RMSHE_RSIHE_EDSHE_VS2017
复原了上面描述的多种直方图均衡算法（图像增强算法）用于和我的算法做对比，参考的文章在注释里面有。
复原的文献如下：
1. Contrast enhancement using brightness preserving bi-histogram equalization	3
    - void BBHE(Mat src, Mat& dst);

2. Image enhancement based on equal area dualistic sub-image histogram equalization method	  4
    - void DSIHE(Mat src, Mat& dst);

3. 2010-Modified histogram equalization for image contrast enhancement	9
    - void HEwVED(Mat src, Mat& dst, double a);
    - void HEwVED(Mat src, Mat& dst, int a);

4. Contrast enhancement using recursive mean-separate histogram equalization for scalable brightness preservation	6
    - void RMSHE(Mat src, Mat& dst, int iterator_s);

5. Recursive sub-image histogram equalization applied to gray scale images	7
    - void RSIHE(Mat src, Mat& dst, int iterator_s);

6. 2016-Contrast Enhancement using Entropy based Dynamic Sub-Histogram Equalization
    - void EDSHE(Mat src, Mat& dst);
