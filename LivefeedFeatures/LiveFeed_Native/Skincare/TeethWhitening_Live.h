#ifndef SKINCARE_TEETH_WHITENING_LIVE_H
#define SKINCARE_TEETH_WHITENING_LIVE_H

#include <opencv2/imgproc.hpp>
#include "Utilities_Live.h"

namespace Skincare
{
	class TeethWhitening_Live
	{
		/*variables*/
	public:

		std::string path;
		int measurements[3];
	private:
		cv::Mat mouthMaskCustom;
		cv::Mat mouthMaskFLD;

		/*methods*/
	public:
		void doTeethWhitening(cv::Mat& inputImage, int fldPoints[], bool isWithoutROI, double effectValue);
		void releaseMemory();
	private:
		double getThreshVal_Otsu_8u_WithMask(const cv::Mat& _src);
		inline void findingOffsets(cv::Scalar averageRGB, double offsets[], double effectValue);
		cv::Rect mouthMasksCreation(const int fldPoints[], cv::Mat& mouthMaskCustomed, cv::Mat& mouthMaskFLD);
	};
}
#endif // SKINCARE_TEETH_WHITENING_LIVE_H
