#ifndef MORPHING_AREA_MORPHS_LIVE_H
#define MORPHING_AREA_MORPHS_LIVE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "ImageTransforms_Live.h"

namespace Morphing
{
	class AreaMorphs_Live : public ImageTransforms_Live
	{
		/*variables*/
	private:
		static int maxLipTop, maxLipBottom;
		static int maxEyeRight, maxEyeLeft;
		static int maxNose;
		static int maxJawRight, maxJawLeft;
		static int maxEyebrowRight, maxEyebrowLeft;
		static int maxLipCornerRight, maxLipCornerLeft;

		/*methods*/
	public:
		void lipMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
		void eyeMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
		void noseMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
		void jawlineMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
		void eyebrowMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
		void lipCornerMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles);
	};
}
#endif // MORPHING_AREA_MORPHS_LIVE_H