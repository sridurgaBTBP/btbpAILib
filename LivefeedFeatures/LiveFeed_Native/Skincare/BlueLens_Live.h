#ifndef SKINCARE_BLUELENS_LIVE_H
#define SKINCARE_BLUELENS_LIVE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace Skincare
{
	//Algorithm designed for 480*640 so can only work for a little variation in resolution.
	class BlueLens_Live
	{
		/*methods*/
	public:
		void performBlueLens(cv::Mat& inputImage, const int fldPoints[], bool isBlurBG, bool isBlueImage, int channel, int colorPercent, double clipLimit);

	private:
		cv::Rect getMask(cv::Mat& maskImage, const int fldPoints[], bool isBlurBG);
		void background(cv::Mat& inputImage, cv::Mat& blurImage, cv::Rect rect);
	};
}
#endif // SKINCARE_BLUELENS_LIVE_H
