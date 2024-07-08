#include"FoundationSimulation_Live.h"
#include "Utilities_Live.h"
#include <opencv2/opencv.hpp>
#include<fstream>


void Beauty::FoundationSimulation_Live::setFoundationShades(const int shades[], int length)
{
	foundationColorShades.clear();
	foundationShadesLength = length;
	for (int i = 0; i < foundationShadesLength; i++)
	{
		foundationColorShades.push_back(shades[i]);
	}
}

cv::Rect Beauty::FoundationSimulation_Live::getFoundationMask(cv::Mat& maskImage, const int fldPoints[])
{
	std::vector<std::vector<cv::Point>> contours(1);
	//fill face mask
	contours[0].reserve(24);
	for (int i = 0; i <= 16; i++)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	for (int i = 74; i >= 68; i--)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	cv::fillPoly(maskImage, contours, cv::Scalar(255));

	std::vector<std::vector<cv::Point>> excludedContours(5);
	//remove lip ROI
	excludedContours[0].reserve(12);
	for (int i = 48; i <= 59; i++)
	{
		excludedContours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}

	//remove left eye ROI
	excludedContours[1].reserve(10);
	for (int i = 17; i <= 21; i++)
	{
		excludedContours[1].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	int eyePoint = (int)((fldPoints[37 * 2 + 1] + fldPoints[38 * 2 + 1]) * 0.5);
	for (int i = 21; i >= 17; i--)
	{
		excludedContours[1].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] + (int)(0.33 * (eyePoint - fldPoints[i * 2 + 1]))));
	}
	excludedContours[2].reserve(6);
	for (int i = 36; i <= 41; i++)
	{
		excludedContours[2].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}

	//remove right eye ROI
	excludedContours[3].reserve(10);
	for (int i = 22; i <= 26; i++)
	{
		excludedContours[3].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	eyePoint = (int)((fldPoints[43 * 2 + 1] + fldPoints[44 * 2 + 1]) * 0.5);
	for (int i = 26; i >= 22; i--)
	{
		excludedContours[3].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] + (int)(0.33 * (eyePoint - fldPoints[i * 2 + 1]))));
	}
	excludedContours[4].reserve(6);
	for (int i = 42; i <= 47; i++)
	{
		excludedContours[4].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	cv::fillPoly(maskImage, excludedContours, cv::Scalar(0));
	return cv::boundingRect(contours[0]);
}

int Beauty::FoundationSimulation_Live::foundation(cv::Mat& inputImage, const int fldPoints[], int shadeIndex, float coverage, bool isMatchFoundation, std::string skinLab)
{
    int color[]={foundationColorShades[shadeIndex*3],foundationColorShades[shadeIndex*3+1],foundationColorShades[shadeIndex*3+2]};
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFoundationMask(maskImage, fldPoints);
	int kernalFace = (int)(rect.width * 0.04 * coverage);
	int kernalEyebrow = (int)(1.8*kernalFace);
	int kernalForehead = (int)(2.5*kernalFace);
	if (kernalFace % 2 == 0)
		kernalFace++;

	if (kernalFace < 3)
		kernalFace = 3;
	if (kernalEyebrow % 2 == 0)
		kernalEyebrow++;
	if (kernalEyebrow < 3)
		kernalEyebrow = 3;
	if (kernalForehead % 2 == 0)
		kernalForehead++;
	if (kernalForehead < 3)
		kernalForehead = 3;
	if (!Utilities_Live::rectanglePadding(rect, kernalForehead, inputImage.cols, inputImage.rows))
		return -1;

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage = maskImage(rect);
	int moustache = fldPoints[33 * 2 + 1] - rect.y;
	int index = 0;
	float ratio0, ratio1, ratio2;
	if (isMatchFoundation)
	{
		float RGB[3] = {};
		Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, skinLab);
		double mindiff = std::abs(RGB[0] - foundationColorShades[0]) + std::abs(RGB[1] - foundationColorShades[1]) + std::abs(RGB[2] - foundationColorShades[2]);
		int n = foundationShadesLength / 3;
		for (int i = 1; i < n; i++)
		{
			double diff = std::abs(RGB[0] - foundationColorShades[i * 3 + 0]) + std::abs(RGB[1] - foundationColorShades[i * 3 + 1]) + std::abs(RGB[2] - foundationColorShades[i * 3 + 2]);
			if (diff < mindiff)
			{
				mindiff = diff;
				index = i;
			}
		}
		ratio0 = 1;
		ratio1 = (float)foundationColorShades[index * 3 + 1] / foundationColorShades[index * 3 + 0];
		ratio2 = (float)foundationColorShades[index * 3 + 2] / foundationColorShades[index * 3 + 0];
	}
	else
	{
		ratio0 = 1;
		ratio1 = (float)color[1] / color[0];
		ratio2 = (float)color[2] / color[0];
		index = -1;
	}
	int foreheadLoc = (int)((fldPoints[68 * 2 + 1] + fldPoints[74 * 2 + 1])* 0.5) - rect.y;
	int eyebrowLoc = (int)((fldPoints[17 * 2 + 1] + fldPoints[26 * 2 + 1]) * 0.5) - rect.y;
	int startY = eyebrowLoc; int endY = maskROIImage.rows - 1;
	cv::Rect featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return -1;
	cv::Mat tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
	startY = foreheadLoc; endY = eyebrowLoc + kernalEyebrow;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return -1;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalEyebrow, kernalEyebrow));
	startY = 0; endY = foreheadLoc + kernalForehead;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return -1;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalForehead, kernalForehead));
	Utilities_Live::saveImage(maskROIImage, "mask.png");

	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;

	float R, sk;
	float resValue = (1.f / 255)*coverage;
	int length = inputROIImage.cols * 3;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * resValue;
			if (sk > 0)
			{
				R = inputPtr[j];
				inputPtr[j] = (uchar)(R  * (1 - sk) + R * ratio0  * sk);
				inputPtr[j + 1] = (uchar)(inputPtr[j + 1] * (1 - sk) + R * ratio1  * sk);
				inputPtr[j + 2] = (uchar)(inputPtr[j + 2] * (1 - sk) + R * ratio2 * sk);
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
	}
	return index;
}

std::vector<float> Beauty::FoundationSimulation_Live::skintone_RGBLAB(const cv::Mat& inputImage, const int fldPoints[], std::string skinLab)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFoundationMask(maskImage, fldPoints);
	Utilities_Live::saveImage(maskImage,"maskRGBLAB.png");
	cv::Mat inputROIImage = inputImage(rect).clone();
	cv::Mat maskROIImage = maskImage(rect).clone();
	int moustache = fldPoints[33 * 2 + 1] - rect.y;
	float RGB[3] = {}, LAB[3] = {};
	Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, skinLab);
	std::cout << "RAvg:: " << RGB[0] << "GAvg:: " << RGB[1] << "BAvg:: " << RGB[2] << std::endl;
	RGB2LAB(RGB, LAB);
	std::vector<float> RGBLAB(6);
	for (int i = 0; i < 3; i++)
	{
		RGBLAB[i] = RGB[i];
		RGBLAB[i + 3] = LAB[i];
	}

	std::ofstream myfile;
	myfile.open(skinLab + ".csv");
	myfile << "R" << "," << "G" << "," << "B" << "," << "L" << "," << "a" << "," << "b" << std::endl;
	std::ostringstream sstream;
	sstream << RGBLAB[0] << "," << RGBLAB[1] << "," << RGBLAB[2] << "," << RGBLAB[3] << "," << RGBLAB[4] << "," << RGBLAB[5];
	std::string varAsString = sstream.str();
	myfile << varAsString << std::endl;
	myfile.close();

	return RGBLAB;
}

bool Beauty::FoundationSimulation_Live::colorMatchingProcess(const cv::Mat& inputImage, const int fldPoints[], int matchedShadesIndex[], std::string skinLab)
{
	try
	{
        const int numberOfFoundationColours = foundationColorShades.size() / 3;
		//region Needed variables
		std::vector<double> FR;
		std::vector<double> FG;
		std::vector<double> FB;
		std::vector<double> FoundationLuminanceColors;
		std::vector<double> FoundationaColors;
		std::vector<double> FoundationbColors;
		std::vector<double> ColorCode;
		std::vector<int> Index;

		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			ColorCode.push_back(i);
		}
		for (int i = 0; i < numberOfFoundationColours * 3; i = i + 3)
		{
			FR.push_back(foundationColorShades[i]);
			FG.push_back(foundationColorShades[i + 1]);
			FB.push_back(foundationColorShades[i + 2]);
		}
		//endregion	

		//region Foundation Color Match
		foundationLABValues(FR, FG, FB, FoundationLuminanceColors, FoundationaColors, FoundationbColors);
		double FoundationColorMatchedLuminance = foundationColorMatchWithSkinColors(inputImage, FoundationLuminanceColors, fldPoints, skinLab);
		//endregion

		double FoundationColorMatchedA = 0, FoundationColorMatchedB = 0;
		double FoundationColorMatchedR = 0, FoundationColorMatchedG = 0, FoundationColorMatchedBlue = 0;

		//region Getting 3 foundation shades for the loaded image
		//region RGBs finding using founded luminance
		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			if (FoundationColorMatchedLuminance == FoundationLuminanceColors.at(i))
			{
				FoundationColorMatchedA = FoundationaColors.at(i);
				FoundationColorMatchedB = FoundationbColors.at(i);
				FoundationColorMatchedR = FR.at(i);
				FoundationColorMatchedG = FG.at(i);
				FoundationColorMatchedBlue = FB.at(i);
				break;
			}
		}
		//endregion

		//region sorting luminance and expanding the remining arrays based on the sorting			
		std::vector<double> TempoararyArray;

		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			TempoararyArray.push_back(FoundationLuminanceColors.at(i));
		}

		std::sort(TempoararyArray.begin(), TempoararyArray.end());

		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			for (int k = 0; k < numberOfFoundationColours; k++)
			{
				if (TempoararyArray.at(i) == FoundationLuminanceColors.at(k))
				{
					Index.push_back(k);
					break;
				}
			}
		}

		std::vector<double> SortedR;
		std::vector<double> SortedG;
		std::vector<double> SortedB;
		std::vector<double> SortedColorCode;

		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			SortedR.push_back(FR.at(Index.at(i)));//based on the luminance sorting
			SortedG.push_back(FG.at(Index.at(i)));
			SortedB.push_back(FB.at(Index.at(i)));
			SortedColorCode.push_back(ColorCode.at(Index.at(i)));
		}
		//endregion
		
		//region getting near RGBs
		for (int i = 0; i < numberOfFoundationColours; i++)
		{
			if (SortedR.at(i) == FoundationColorMatchedR && SortedG.at(i) == FoundationColorMatchedG && SortedB.at(i) == FoundationColorMatchedBlue)
			{
				/* storing the matched shade in index 1 and (lighter1/Darker1 shade in index 0, lighter2/Darker2 shade in index 2 when matching shade is at extreme,
				    else storing the matched shade as Darker,Ideal and Brighter in 0, 1, 2 index respectively*/
				if (i == 0)
				{
					matchedShadesIndex[0] = SortedColorCode.at(i + 1);
					matchedShadesIndex[1] = SortedColorCode.at(i);
					matchedShadesIndex[2] = SortedColorCode.at(i + 2);
				}
				else if (i == (numberOfFoundationColours - 1))
				{
					matchedShadesIndex[0] = SortedColorCode.at(i - 2);
					matchedShadesIndex[1] = SortedColorCode.at(i);
					matchedShadesIndex[2] = SortedColorCode.at(i - 1);
				}
				else
				{	
					matchedShadesIndex[0] = SortedColorCode.at(i - 1);
					matchedShadesIndex[1] = SortedColorCode.at(i);
					matchedShadesIndex[2] = SortedColorCode.at(i + 1);
				}
				break;
			}
		}
		return true;
	}
	catch (cv::Exception ex)
	{
		return false;
	}
	return true;
}

void Beauty::FoundationSimulation_Live::foundationLABValues(std::vector<double> FRed, std::vector<double> FGreen, std::vector<double> FBlue, std::vector<double>& FLum, std::vector<double>& Fa, std::vector<double>& Fb)
{
	try
	{
		int numberofcolors = FRed.size();
		float RGBs[3] = { 0, 0, 0 };
		float LABs[3] = { 0, 0, 0 };
		for (int i = 0; i < numberofcolors; i++)
		{
			RGBs[2] = FBlue.at(i);     // In live it is in RGB format
			RGBs[1] = FGreen.at(i);
			RGBs[0] = FRed.at(i);

			LABs[0] = 0;
			LABs[1] = 0;
			LABs[2] = 0;

			RGB2LAB(RGBs, LABs);

			FLum.push_back(LABs[0]);
			Fa.push_back(LABs[1]);
			Fb.push_back(LABs[2]);
		}
	}
	catch (cv::Exception ex)
	{
		return;
	}
}

void  Beauty::FoundationSimulation_Live::RGB2LAB(float* RGBs, float* Labs)
{
	try
	{
		//region RGb2LAB regular method
		float eps = 216 / (float)24389;
		float k = 24389 / (float)27;
		//XYZ conversion illuminant values       
		float D65Xr = 0.95047;   // reference white D65
		float D65Yr = 1.0f;        // reference white D65
		float D65Zr = 1.08883f;   // reference white D65 
		float r = 0, g = 0, b = 0;
		float xr = 0, yr = 0, zr = 0, fx = 0, fy = 0, fz = 0;

		// RGB to XYZ*********************************************************
		//change scale from 0-255 to 0-1
		r = RGBs[0] / 255; //R 0..1
		g = RGBs[1] / 255; //G 0..1
		b = RGBs[2] / 255; //B 0..1

		//Gamma correction for RGB to XYZ (assumes a gamma of 2.2)
		//12.92, 1.055 and 0.055 are simplifications of equations that produce the true values
		//we need to evaluate them to see how well they serve

		if (r <= 0.04045)
			r = r / 12.92f;
		else
			r = (float)std::pow((r + 0.055) / 1.055, 2.4);

		if (g <= 0.04045)
			g = g / 12.92f;
		else
			g = (float)std::pow((g + 0.055) / 1.055, 2.4);

		if (b <= 0.04045)
			b = b / 12.92f;
		else
			b = (float)std::pow((b + 0.055) / 1.055, 2.4);

		//assuming sRGB as our canon cameras capture in this standard

		xr = (0.412453f * r + 0.357580f * g + 0.180423f * b) / D65Xr;
		yr = (0.212671f * r + 0.715160f * g + 0.072169f * b) / D65Yr;
		zr = (0.019334f * r + 0.119193f * g + 0.950227f * b) / D65Zr;


		// XYZ to Lab*************************************************************************
		//CIE LAB values as this is what our clients use
		//Gamma correction for XYZ to LAB
		if (xr > eps)
			fx = (float)std::pow(xr, 0.3333);
		else
			fx = (float)(((k * xr) + 16) / 116);

		if (yr > eps)
			fy = (float)std::pow(yr, 0.3333);
		else
			fy = (float)(((k * yr) + 16) / 116);

		if (zr > eps)
			fz = (float)std::pow(zr, 0.3333);
		else
			fz = (float)(((k * zr) + 16) / 116);

		Labs[0] = (116 * fy) - 16;
		Labs[1] = 500 * (fx - fy);
		Labs[2] = 200 * (fy - fz);
		//endregion
	}
	catch (cv::Exception ex)
	{
		return;
	}
}

double Beauty::FoundationSimulation_Live::foundationColorMatchWithSkinColors(const cv::Mat& inputImage, std::vector<double> LuminanceColors, const int fldPoints[], std::string skinLab)
{
	try
	{
		std::vector<float> RGBLAB = skintone_RGBLAB(inputImage, fldPoints, skinLab);
		int index = 0;
		double mindiff = std::abs(RGBLAB[3] - LuminanceColors[0]);    // since index of L is 3
		int n = foundationShadesLength / 3;
		for (int i = 1; i < n; i++)
		{
			double diff = std::abs(RGBLAB[3] - LuminanceColors[i]);
			if (diff < mindiff)
			{
				mindiff = diff;
				index = i;
			}
		}
		return LuminanceColors.at(index);
	}
	catch (cv::Exception ex)
	{
		return 0;
	}
}

Beauty::FoundationSimulation_Live::FoundationSimulation_Live()
{

}


Beauty::FoundationSimulation_Live::~FoundationSimulation_Live()
{
}
