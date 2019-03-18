#pragma once
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include "hist.hpp"

using namespace cv;
using namespace std;

double DEN(const Mat& src, const Mat& dst);
double DE_(const Mat& src);
int calcGrayNum(const Mat& src);
