#ifndef BEAUTY_FOUNDATION_SIMULATION_LIVE_H
#define BEAUTY_FOUNDATION_SIMULATION_LIVE_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace Beauty
{	
	class FoundationSimulation_Live
	{
	public:
		FoundationSimulation_Live();
		~FoundationSimulation_Live();
		void setFoundationShades(const int colors[], int length);
		int foundation(cv::Mat& inputImage, const int fldPoints[], int shadeIndex, float coverage, bool isMatchFoundation, std::string skinLab);
		std::vector<float> skintone_RGBLAB(const cv::Mat& inputImage, const int fldPoints[], std::string skinLab);
		bool colorMatchingProcess(const cv::Mat& inputImage, const int fldPoints[], int matchedShadesIndex[], std::string skinLab);

	private:
		int foundationShadesLength;
		std::vector<int> foundationColorShades;
		const int numberOfShadesToPresent = 3; // shades recommended to user

		cv::Rect getFoundationMask(cv::Mat& maskImage, const int fldPoints[]);
		void RGB2LAB(float* RGBs, float* Labs);
		double foundationColorMatchWithSkinColors(const cv::Mat& inputImage, std::vector<double> LuminanceColors, const int fldPoints[], std::string skinLab);
		void foundationLABValues(std::vector<double> FRed, std::vector<double> FGreen, std::vector<double> FBlue, std::vector<double>& FLum, std::vector<double>& Fa, std::vector<double>& Fb);

	};
}

#endif
