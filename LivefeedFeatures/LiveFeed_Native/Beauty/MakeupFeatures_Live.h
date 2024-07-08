#ifndef BEAUTY_MAKEUP_FEATURES_LIVE_H
#define BEAUTY_MAKEUP_FEATURES_LIVE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace Beauty
{
	//Algorithms designed for 480*640 so can only work for a little variation in resolution.
	//Only lipstick and complexion are made to work with high resolution providing that fld points are sent on 640*480 or closer resolution.
	class MakeupFeatures_Live
	{
		/*variables*/
	private:
		static int foundationShadesLength;
		static int* foundationShades;
		static bool brightness_isOverSaturated;
		static float brightness_maxValue, brightness_overSaturatedValue, brightness_overSaturationStep;


		/*methods*/
	public:
		static void setFoundationShades(const int colors[], int length);
		int foundation(cv::Mat& inputImage, const int fldPoints[], int color[], float coverage, bool isMatchFoundation);
		void complexion(cv::Mat& inputImage, double lowResFactor, const int lowResFldPoints[], bool isTexture, float textureIntensity, bool isEvenness, float evennessIntensity);
		void wrinkles/*complexionZoneLevel*/(cv::Mat& inputImage, const int fldPoints[], bool zones[], float intensity[]);
		void brightness(cv::Mat& inputImage, const int fldPoints[], float intensity);
		void lipstick(cv::Mat& inputImage, double lowResFactor, const int lowResFldPoints[], int color[], float coverage, float intensity, float glossValue, float featheringFactor);
		void lipHealth(cv::Mat& inputImage, const int fldPoints[], float intensity);//Brand Evangelist 

	private:
		cv::Rect getFoundationMask(cv::Mat& maskImage, const int fldPoints[]);
		cv::Rect getComplexionMask(cv::Mat& maskImage, const int fldPoints[]);
		void getComplexionMaskZoneLevel(cv::Mat& maskImage, const int fldPoints[], bool zones[], cv::Rect rects[]);
		cv::Rect getLipMask(cv::Mat& maskImage, cv::Mat& maskImage2, const int fldPoints[]);

		//Features asked by clients(Mirror web)
	public:
		void foundation(cv::Mat& inputImage, const int fldPoints[], int color[], float coverage, float nonMatchFoundationintensity, float maxIntensityLimit, bool applyMaxLimits);//Rodan_Fileds
		std::vector<float> skintone_RGBLAB(cv::Mat& inputImage, const int fldPoints[]);//Mobigesture
	};
}
#endif // BEAUTY_MAKEUP_FEATURES_LIVE_H

