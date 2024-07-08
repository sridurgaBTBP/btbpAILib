#ifndef MORPHING_IMAGE_TRANSFORMS_LIVE_H
#define MORPHING_IMAGE_TRANSFORMS_LIVE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace Morphing
{
	struct ObjectInfo
	{
		std::vector<cv::Point2f> polygon;
		cv::Point2f centroid;
		bool isApplied;
	};

	//#define TESTING
	class ImageTransforms_Live
	{
#ifdef  TESTING
		/*variables*/
	private:
		cv::Mat helperImage;
#endif //TESTING

		/*methods*/
	protected:
		void affineTransformation(cv::Point* pointsSrc, cv::Point* pointsDst, int* triSets, int index, const cv::Mat& cropImage, cv::Mat& maskImage, cv::Mat& tempImage, cv::Mat& morphedImage, bool drawTriangles);
		std::vector<ObjectInfo> affineTransformation(const cv::Point2f* pointsSrc, const cv::Point2f* pointsDst, const int* triSets, int triSetsLength, std::vector<ObjectInfo> polygonsInfo);
		bool checkBoundsForAffine(cv::Rect& rect, int cols, int rows);
		bool checkIQCForAffine(const int fldPoints[], float thresholds[]);
		bool checkIQCForAffine(const float fldPoints[], float thresholds[]);
	};
}
#endif // MORPHING_IMAGE_TRANSFORMS_LIVE_H
