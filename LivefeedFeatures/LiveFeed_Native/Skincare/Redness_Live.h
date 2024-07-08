#ifndef SKINCARE_REDNESS_LIVE_H
#define SKINCARE_REDNESS_LIVE_H

#include <opencv2/imgproc.hpp>
#include <iostream>
#include "Utilities_Live.h"

namespace Skincare
{
	class Redness_Live
	{
		/*variables*/
	public:
		int rednessvalues[2];
	private:
		cv::Mat foundationmaskImage;
		cv::Mat vertImage;
		cv::Mat negativeImage;
		cv::Mat featheredfoundationmaskImage;
		cv::Mat vertmaskImage;

		/*methods*/
	public:
		void performRedness(cv::Mat& inputImage, int fldPoints[], bool Islocal, double offset);
		void releaseMemory();
	private:
		void rednessLocalUntreated(cv::Mat& inputImage, const cv::Mat& negativeImage, const cv::Mat& foundationmaskImage, const cv::Mat& featheredfoundationmaskImage, double untreatedoffset);
		void rednessFaceUntreated(cv::Mat& inputImage, const cv::Mat& foundationmaskImage, const cv::Mat& featheredfoundationmaskImage, double untreatedoffset);
	};
}
#endif // SKINCARE_REDNESS_LIVE_H
