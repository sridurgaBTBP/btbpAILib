#ifndef SKINCARE_HAIR_FEATURES_LIVE_H
#define SKINCARE_HAIR_FEATURES_LIVE_H

#include <opencv2/core/mat.hpp>

namespace Skincare
{
	class HairFeatures
	{
		/*methods*/
	public:
		void kkmeans_ex();
		void kkmeans(cv::Mat& inputImage);
		void hairColor(cv::Mat& inputImage, cv::Mat& maskImage224, int foreheadPos, int color[], float coverage);
	};
}
#endif // SKINCARE_HAIR_FEATURES_LIVE_H
