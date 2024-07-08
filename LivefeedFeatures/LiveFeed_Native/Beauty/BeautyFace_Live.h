#ifndef BEAUTY_BEAUTY_FACE_Live_H
#define BEAUTY_BEAUTY_FACE_Live_H

#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utilities_Live.h"

namespace Beauty
{
	class BeautyFace_Live
	{
	public:
		cv::Mat negativeImage;
		cv::Mat positiveImage;
		cv::Mat skinrectImage;
		cv::Mat foundationmaskImage;
		cv::Mat featheredfoundationmaskImage;
		int featheringFilter, featheringIterations;
		cv::Mat tempImage;
		cv::Mat cvgrayImage;
		cv::Mat faceImage;
		cv::Mat shinemaskImage;
		cv::Mat newNegMaskImage;
		cv::Mat channels[3];

		void performBeautyFace(cv::Mat& inputImage, int fldPoints[], double treatedPert);
		void beautyFaceSimulation(cv::Mat& inputImage, double treatPercent);
		void releaseMemory();
	};
}
#endif // BEAUTY_BEAUTY_FACE_Live_H

