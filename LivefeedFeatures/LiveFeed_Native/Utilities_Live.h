#ifndef UTILITIES_LIVE_H
#define UTILITIES_LIVE_H

#include <opencv2/core/types.hpp>
#include <deque>
#include <iostream>

class Utilities_Live
{
	/*variables*/
public:
	static std::string path;
	static bool isMorphStable;
	static int fps;

private:
	static cv::Point previousFaceSplitPoints[8];
	static cv::Point previousROISplitPoints[7];

	/*methods*/
public:
	static cv::Rect getFaceRectangle(const int fldPoints[]);
	static void splitViews(const cv::Mat& inputImage, cv::Mat& outputImage, const int fldPoints[], const int viewType, const int* params, int* out_splitFacePoints);
	static bool rectanglePadding(cv::Rect& rect, int offset, int maxX, int maxY);
	static void averageRGB(const cv::Mat& inputImage, const cv::Mat& maskImage, int moustache, float RGB[], float lowLimitPer, float highLimitPer, std::string skinLab);
	static void saveImage(const cv::Mat& image, std::string name);
};
#endif // UTILITIES_LIVE_H
