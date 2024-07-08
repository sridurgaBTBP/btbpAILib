#ifndef SKINCARE_HAIR_DETECTION_LIVE_H
#define SKINCARE_HAIR_DETECTION_LIVE_H

#include <opencv2/imgproc.hpp>
#include "Utilities_Live.h"
#include <deque>

namespace Skincare
{
	class HairDetection_Live
	{
		/*variables*/
	public:
		std::string path;
		std::deque<std::vector<int>> landmarkSamplesForMeasurements;
		std::deque<std::vector<int>> measurementSamples;
		int measurements[4];
	private:
		int filterSize;
		float dataArrayH[7][7], dataArrayV[7][7], dataArrayD1[7][7], dataArrayD2[7][7];
		cv::Mat grayImage;

		/*methods*/
	public:
		HairDetection_Live();
		void detectHair(cv::Mat& inputImage, const int fldPoints[]);
		void releaseMemory();
	};
}
#endif // SKINCARE_HAIR_DETECTION_LIVE_H
