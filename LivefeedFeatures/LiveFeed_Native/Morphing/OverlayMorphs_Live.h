#ifndef  MORPHING_OVERLAY_MORPHS_LIVE_H
#define  MORPHING_OVERLAY_MORPHS_LIVE_H

#include <deque>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "ImageTransforms_Live.h"

namespace Morphing
{
	class OverlayMorphs_Live : public ImageTransforms_Live
	{
		/*variables*/
	private:
		static const int fldPointsLength = 75;
		//eyebrowMorph
		static int eyebrowTemplatesX[5];
		static int eyebrowTemplatesY[6][5];
		static cv::Point eyebrowOffsetsRight[5], eyebrowOffsetsLeft[5];
		static std::deque<std::vector<cv::Point>>  eyebrowOffsetSamplesRight, eyebrowOffsetSamplesLeft;
		static int eyebrowCurrentShapeIndex;
		//objectsOverlapping
		cv::Point2f objectsFldPoints[fldPointsLength];
		std::vector<ObjectInfo> objectsInfo;

	public:
		static bool isStillImage;

		/*methods*/
	public:
		bool eyebrowMorph(cv::Mat& inputImage, const int fldPoints[], int shapeIndex, float morphPercent, bool drawTriangles);
		void loadObjectsInfo(const std::vector<std::vector<cv::Point2f>>& contours, const float fldPoints[]);
		std::vector<ObjectInfo> objectsOverlapping(const float fldPoints[], float thresholdFactor);

		//For desktop R&D
		cv::Mat eyelashAlignment(const cv::Mat& baselineImage, const int baselineFldPoints[], const cv::Mat& currentImage, const int currentFldPoints[], bool drawTriangles);
	};
}
#endif // MORPHING_OVERLAY_MORPHS_LIVE_H
