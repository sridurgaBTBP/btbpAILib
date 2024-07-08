#include "MakeupFeatures_Live.h"
#include "Utilities_Live.h"
#include <opencv2/imgproc.hpp>

int Beauty::MakeupFeatures_Live::foundationShadesLength = 0;
int* Beauty::MakeupFeatures_Live::foundationShades;
bool Beauty::MakeupFeatures_Live::brightness_isOverSaturated = true;
float Beauty::MakeupFeatures_Live::brightness_maxValue = 10;
float Beauty::MakeupFeatures_Live::brightness_overSaturatedValue = 10;
float Beauty::MakeupFeatures_Live::brightness_overSaturationStep = 0.2f;

void Beauty::MakeupFeatures_Live::setFoundationShades(const int shades[], int length)
{
	foundationShadesLength = length;
	foundationShades = new int[foundationShadesLength];
	for (int i = 0; i < foundationShadesLength; i++)
	{
		foundationShades[i] = shades[i];
	}
}

cv::Rect Beauty::MakeupFeatures_Live::getFoundationMask(cv::Mat& maskImage, const int fldPoints[])
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

int Beauty::MakeupFeatures_Live::foundation(cv::Mat& inputImage, const int fldPoints[], int color[], float coverage, bool isMatchFoundation)
{
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
		Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, "skinLab");
		double mindiff = std::abs(RGB[0] - foundationShades[0]) + std::abs(RGB[1] - foundationShades[1]) + std::abs(RGB[2] - foundationShades[2]);
		int n = foundationShadesLength / 3;
		for (int i = 1; i < n; i++)
		{
			double diff = std::abs(RGB[0] - foundationShades[i * 3 + 0]) + std::abs(RGB[1] - foundationShades[i * 3 + 1]) + std::abs(RGB[2] - foundationShades[i * 3 + 2]);
			if (diff < mindiff)
			{
				mindiff = diff;
				index = i;
			}
		}
		ratio0 = 1;
		ratio1 = (float)foundationShades[index * 3 + 1] / foundationShades[index * 3 + 0];
		ratio2 = (float)foundationShades[index * 3 + 2] / foundationShades[index * 3 + 0];
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

void Beauty::MakeupFeatures_Live::foundation(cv::Mat& inputImage, const int fldPoints[], int color[], float coverage, float nonMatchFoundationintensity, float maxIntensityLimit, bool applyMaxLimits)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFoundationMask(maskImage, fldPoints);
	int kernalFace = (int)(rect.width * 0.04);
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
		return;

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage = maskImage(rect);
	int moustache = fldPoints[33 * 2 + 1] - rect.y;
	float	ratio0 = 1;
	float	ratio1 = (float)color[1] / color[0];
	float	ratio2 = (float)color[2] / color[0];
	float RGB[3] = {};
	Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, "skinLab");
	float intensityFactor = ((color[0] + color[1] + color[2]) / (RGB[0] + RGB[1] + RGB[2])) - 1;
	//std::cout << "ShadeToSkin ratio: " << intensityFactor + 1;
	intensityFactor *= nonMatchFoundationintensity;
	//std::cout << ", applied ratio: " << intensityFactor + 1;
	//std::cout << ", limits range: " << 1.f / maxIntensityLimit <<" - "<< maxIntensityLimit;
	if (applyMaxLimits)
	{
		float highLimit = maxIntensityLimit - 1;
		float lowLimit = (1.f / maxIntensityLimit) - 1;
		if (intensityFactor < lowLimit)
			intensityFactor = lowLimit;
		else if (intensityFactor > highLimit)
			intensityFactor = highLimit;
	}
	//std::cout << ", applied ratio after limits: " << intensityFactor + 1 << std::endl;
	int foreheadLoc = (int)((fldPoints[68 * 2 + 1] + fldPoints[74 * 2 + 1])* 0.5) - rect.y;
	int eyebrowLoc = (int)((fldPoints[17 * 2 + 1] + fldPoints[26 * 2 + 1]) * 0.5) - rect.y;
	int startY = eyebrowLoc; int endY = maskROIImage.rows - 1;
	cv::Rect featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	cv::Mat tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
	startY = foreheadLoc; endY = eyebrowLoc + kernalEyebrow;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalEyebrow, kernalEyebrow));
	startY = 0; endY = foreheadLoc + kernalForehead;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalForehead, kernalForehead));
	Utilities_Live::saveImage(maskROIImage, "mask.png");

	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;

	float R, G, B, sk;
	float resValue = (1.f / 255)*coverage;
	int length = inputROIImage.cols * 3;
	float I = 1 + intensityFactor;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * resValue;
			if (sk > 0)
			{
				B = inputPtr[j + 2];
				G = inputPtr[j + 1];
				R = inputPtr[j];
				B = B * (1 - sk) + R * ratio2*I * sk;
				G = G * (1 - sk) + R * ratio1*I  * sk;
				R = R  * (1 - sk) + R * ratio0*I  * sk;
				if (R > 255)
					R = 255;
				if (G > 255)
					G = 255;
				if (B > 255)
					B = 255;
				inputPtr[j] = (uchar)R;
				inputPtr[j + 1] = (uchar)G;
				inputPtr[j + 2] = (uchar)B;
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
	}
}

std::vector<float> Beauty::MakeupFeatures_Live::skintone_RGBLAB(cv::Mat& inputImage, const int fldPoints[])
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFoundationMask(maskImage, fldPoints);
	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage = maskImage(rect);
	int moustache = fldPoints[33 * 2 + 1] - rect.y;
	float RGB[3] = {};
	Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, "skinLab");
	//RGB2LAB
	//change scale from 0-255 to 0-1
	double r = RGB[0] / 255.0f; //R 0..1
	double g = RGB[1] / 255.0f; //G 0..1
	double b = RGB[2] / 255.0f; //B 0..1
	 //Gamma correction for RGB to XYZ (assumes a gamma of 2.2)
	 //12.92, 1.055 and 0.055 are simplifications of equations that produce the true values
	//we need to evaluate them to see how well they serve
	if (r <= 0.04045)
		r = r / 12.92f;
	else
		r = pow((r + 0.055) / 1.055, 2.4);

	if (g <= 0.04045)
		g = g / 12.92f;
	else
		g = pow((g + 0.055) / 1.055, 2.4);

	if (b <= 0.04045)
		b = b / 12.92f;
	else
		b = pow((b + 0.055) / 1.055, 2.4);

	//assuming sRGB as our canon cameras capture in this standard
	double xr = (0.412453f * r + 0.357580f * g + 0.180423f * b) / 0.95047f;
	double yr = (0.212671f * r + 0.715160f * g + 0.072169f * b) / 1.0f;
	double zr = (0.019334f * r + 0.119193f * g + 0.950227f * b) / 1.08883f;

	// XYZ to Lab*************************************************************************
	//CIE LAB values as this is what our clients use
	//Gamma correction for XYZ to LAB
	double eps = 216.00f / 24389.00f;
	double 	k = 24389.00f / 27.00f;
	double fx, fy, fz;
	if (xr > eps)
		fx = pow(xr, 0.3333);
	else
		fx = (((k * xr) + 16) / 116);

	if (yr > eps)
		fy = pow(yr, 0.3333);
	else
		fy = (((k * yr) + 16) / 116);

	if (zr > eps)
		fz = pow(zr, 0.3333);
	else
		fz = (((k * zr) + 16) / 116);
	float LAB[3];
	LAB[0] = (float)((116 * fy) - 16);
	LAB[1] = (float)(500 * (fx - fy));
	LAB[2] = (float)(200 * (fy - fz));
	std::vector<float> RGBLAB(6);
	for (int i = 0; i < 3; i++)
	{
		RGBLAB[i] = RGB[i];
		RGBLAB[i + 3] = LAB[i];
	}
	return RGBLAB;
}

cv::Rect Beauty::MakeupFeatures_Live::getComplexionMask(cv::Mat& maskImage, const int fldPoints[])
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

	std::vector<std::vector<cv::Point>> excludedContours(3);
	//remove lip ROI
	excludedContours[0].reserve(12);
	int lipOffset1 = (int)((fldPoints[33 * 2 + 1] - fldPoints[30 * 2 + 1]) * 0.8);
	for (int i = 31; i <= 35; i++)
	{
		excludedContours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] - lipOffset1));
	}
	int lipOffset2 = (int)((fldPoints[12 * 2] - fldPoints[54 * 2]) * 0.3);
	excludedContours[0].push_back(cv::Point(fldPoints[54 * 2] + lipOffset2, fldPoints[54 * 2 + 1]));
	lipOffset2 = (int)((fldPoints[8 * 2 + 1] - fldPoints[57 * 2 + 1]) * 0.3);
	for (int i = 55; i <= 59; i++)
	{
		excludedContours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] + lipOffset2));
	}
	lipOffset2 = (int)((fldPoints[48 * 2] - fldPoints[4 * 2]) * 0.3);
	excludedContours[0].push_back(cv::Point(fldPoints[48 * 2] - lipOffset2, fldPoints[48 * 2 + 1]));

	//remove left eye ROI
	excludedContours[1].reserve(9);
	int eyebrowOffset = (int)((fldPoints[19 * 2 + 1] - fldPoints[68 * 2 + 1]) * 0.3);
	for (int i = 17; i <= 21; i++)
	{
		excludedContours[1].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] - eyebrowOffset));
	}
	excludedContours[1].push_back(cv::Point(fldPoints[39 * 2], fldPoints[39 * 2 + 1]));
	int eyeOffset = (int)((fldPoints[30 * 2 + 1] - fldPoints[40 * 2 + 1]) * 0.4);
	for (int i = 40; i <= 41; i++)
	{
		excludedContours[1].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] + eyeOffset));
	}
	excludedContours[1].push_back(cv::Point(fldPoints[0 * 2], fldPoints[0 * 2 + 1]));

	//remove right eye ROI
	excludedContours[2].reserve(9);
	eyebrowOffset = (int)((fldPoints[24 * 2 + 1] - fldPoints[74 * 2 + 1]) * 0.3);
	for (int i = 22; i <= 26; i++)
	{
		excludedContours[2].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] - eyebrowOffset));
	}
	excludedContours[2].push_back(cv::Point(fldPoints[16 * 2], fldPoints[16 * 2 + 1]));
	eyeOffset = (int)((fldPoints[30 * 2 + 1] - fldPoints[46 * 2 + 1]) * 0.4);
	for (int i = 46; i <= 47; i++)
	{
		excludedContours[2].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1] + eyeOffset));
	}
	excludedContours[2].push_back(cv::Point(fldPoints[42 * 2], fldPoints[42 * 2 + 1]));
	cv::fillPoly(maskImage, excludedContours, cv::Scalar(0));
	return cv::boundingRect(contours[0]);
}

void Beauty::MakeupFeatures_Live::complexion(cv::Mat& inputImage, double lowResFactor, const int lowResFldPoints[], bool isTexture, float textureIntensity, bool isEvenness, float evennessIntensity)
{
	int lowResRows = (int)(inputImage.rows / lowResFactor), lowResCols = (int)(inputImage.cols / lowResFactor);
	cv::Mat maskImage = cv::Mat::zeros(lowResRows, lowResCols, CV_8UC1);
	cv::Rect lowResRect = getComplexionMask(maskImage, lowResFldPoints);
	int	kernalFace = (int)(lowResRect.width*0.04* std::max(textureIntensity, evennessIntensity));
	if (kernalFace % 2 == 0)
		kernalFace++;
	if (kernalFace < 3)
		kernalFace = 3;
	//fixed size filter is applied as the users wanted to see complexion at same intensity on small faces like bigger faces(which will not be same when filter is calculated from faces size).
	int kernalComplexion = 5;
	if (!Utilities_Live::rectanglePadding(lowResRect, std::max(kernalFace, kernalComplexion), lowResCols, lowResRows))
		return;
	cv::Rect rect((int)(lowResRect.x*lowResFactor), (int)(lowResRect.y*lowResFactor), (int)(lowResRect.width*lowResFactor), (int)(lowResRect.height*lowResFactor));
	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage;
	cv::blur(maskImage(lowResRect), maskROIImage, cv::Size(kernalFace, kernalFace));
	cv::resize(maskROIImage, maskROIImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	cv::Mat complexionROIImage;
	cv::resize(inputROIImage, complexionROIImage, cv::Size(lowResRect.width, lowResRect.height));
	cv::blur(complexionROIImage, complexionROIImage, cv::Size(kernalComplexion, kernalComplexion));
	cv::resize(complexionROIImage, complexionROIImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(complexionROIImage, "complexion.png");

	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	const uchar*  complexionPtr = complexionROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;
	int complexionStep = (int)complexionROIImage.step;

	float R, G, B, SR, SG, SB, sk, sk1;
	float resValueT = (1.f / 255)*textureIntensity;
	float resValueE = (1.f / 255)*evennessIntensity;
	int length = inputROIImage.cols * 3;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			R = inputPtr[j];
			G = inputPtr[j + 1];
			B = inputPtr[j + 2];
			sk1 = maskPtr[++k];
			if (isTexture)
			{
				sk = sk1* resValueT;
				if (sk > 0)
				{
					SR = complexionPtr[j];
					SG = complexionPtr[j + 1];
					SB = complexionPtr[j + 2];
					if (R - SR > 0)
						inputPtr[j] = (uchar)(R * (1 - sk) + SR * sk);
					if (G - SG > 0)
						inputPtr[j + 1] = (uchar)(G  * (1 - sk) + SG * sk);
					if (B - SB > 0)
						inputPtr[j + 2] = (uchar)(B  * (1 - sk) + SB * sk);
				}
			}
			if (isEvenness)
			{
				sk = sk1 * resValueE;
				if (sk > 0)
				{
					SR = complexionPtr[j];
					SG = complexionPtr[j + 1];
					SB = complexionPtr[j + 2];
					if (SR - R > 0)
						inputPtr[j] = (uchar)(R * (1 - sk) + SR * sk);
					if (SG - G > 0)
						inputPtr[j + 1] = (uchar)(G  * (1 - sk) + SG * sk);
					if (SB - B > 0)
						inputPtr[j + 2] = (uchar)(B  * (1 - sk) + SB * sk);
				}
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		complexionPtr += complexionStep;
	}
}

void Beauty::MakeupFeatures_Live::getComplexionMaskZoneLevel(cv::Mat& maskImage, const int fldPoints[], bool zones[], cv::Rect rects[])
{
	std::vector<std::vector<cv::Point>> regions(3);
	//forehead.
	if (zones[0])
	{
		regions[0].resize(13);
		int xThresh_glabella = (int)((fldPoints[22 * 2] - fldPoints[21 * 2])*0.25);
		regions[0][0].x = fldPoints[68 * 2]; regions[0][0].y = fldPoints[68 * 2 + 1];
		regions[0][1].x = fldPoints[69 * 2]; regions[0][1].y = fldPoints[69 * 2 + 1];
		regions[0][2].x = fldPoints[70 * 2]; regions[0][2].y = fldPoints[70 * 2 + 1];
		regions[0][3].x = fldPoints[71 * 2]; regions[0][3].y = fldPoints[71 * 2 + 1];
		regions[0][4].x = fldPoints[72 * 2]; regions[0][4].y = fldPoints[72 * 2 + 1];
		regions[0][5].x = fldPoints[73 * 2]; regions[0][5].y = fldPoints[73 * 2 + 1];
		regions[0][6].x = fldPoints[74 * 2]; regions[0][6].y = fldPoints[74 * 2 + 1];
		regions[0][7].x = fldPoints[24 * 2]; regions[0][7].y = fldPoints[24 * 2 + 1] - (int)((fldPoints[24 * 2 + 1] - fldPoints[74 * 2 + 1])*0.2);
		regions[0][8].x = fldPoints[23 * 2]; regions[0][8].y = fldPoints[23 * 2 + 1] - (int)((fldPoints[24 * 2 + 1] - fldPoints[74 * 2 + 1])*0.2);
		regions[0][11].x = fldPoints[20 * 2]; regions[0][11].y = fldPoints[20 * 2 + 1] - (int)((fldPoints[19 * 2 + 1] - fldPoints[68 * 2 + 1])*0.2);
		regions[0][12].x = fldPoints[19 * 2]; regions[0][12].y = fldPoints[19 * 2 + 1] - (int)((fldPoints[19 * 2 + 1] - fldPoints[68 * 2 + 1])*0.2);
		double foreheadSlope = (double)(regions[0][8].y - regions[0][11].y) / (regions[0][8].x - regions[0][11].x);//no need to check denominator for zero,its always give a positive value
		int centerX = (regions[0][8].x + regions[0][11].x) / 2;
		regions[0][9].x = centerX + xThresh_glabella;
		regions[0][9].y = regions[0][11].y + (int)(foreheadSlope*(regions[0][9].x - regions[0][11].x));
		regions[0][10].x = centerX - xThresh_glabella;
		regions[0][10].y = regions[0][11].y + (int)(foreheadSlope*(regions[0][10].x - regions[0][11].x));
		cv::drawContours(maskImage, regions, 0, cv::Scalar(255), cv::FILLED);
		rects[0] = cv::boundingRect(regions[0]);
	}
	//undereye
	if (zones[1])
	{
		int rightSideDistance = 0, leftSideDistance = 0;
		rightSideDistance += fldPoints[30 * 2] - fldPoints[2 * 2];
		rightSideDistance += fldPoints[33 * 2] - fldPoints[3 * 2];
		leftSideDistance += fldPoints[14 * 2] - fldPoints[30 * 2];
		leftSideDistance += fldPoints[13 * 2] - fldPoints[33 * 2];
		int faceWidth = rightSideDistance + leftSideDistance,
			differenceInSides = std::abs(rightSideDistance - leftSideDistance);
		int poseType = -1;
		if (differenceInSides < 0.4*faceWidth)
			poseType = 0;
		else if (leftSideDistance > rightSideDistance)
			poseType = 1;
		else
			poseType = 2;
		int YThresh_underEye = (int)((fldPoints[8 * 2 + 1] - fldPoints[27 * 2 + 1])*0.15);
		//right
		regions[2].resize(5);
		cv::Rect rect0;
		if (poseType == 2)
		{
			int XThresh_crowsfeet = fldPoints[17 * 2] - fldPoints[0 * 2];
			regions[2][0].x = fldPoints[17 * 2] - (int)(0.1*XThresh_crowsfeet); regions[2][0].y = fldPoints[17 * 2 + 1];
			regions[2][1].x = fldPoints[0 * 2] + (int)(0.5*XThresh_crowsfeet); regions[2][1].y = fldPoints[17 * 2 + 1];
			regions[2][2].x = fldPoints[0 * 2] + (int)(0.4*XThresh_crowsfeet); regions[2][2].y = fldPoints[36 * 2 + 1];
			regions[2][4].x = fldPoints[36 * 2] - (int)(0.1*XThresh_crowsfeet); regions[2][4].y = fldPoints[36 * 2 + 1];
			regions[2][3].x = (regions[2][2].x + regions[2][4].x) / 2; regions[2][3].y = fldPoints[1 * 2 + 1];
			cv::drawContours(maskImage, regions, 2, cv::Scalar(255), cv::FILLED);
			rect0 = cv::boundingRect(regions[2]);
		}
		regions[1].resize(8);
		int XThresh_underEye = (int)((fldPoints[39 * 2] - fldPoints[36 * 2])*0.1);
		regions[1][0].x = fldPoints[39 * 2]; regions[1][0].y = fldPoints[39 * 2 + 1] + (int)(0.15*YThresh_underEye);
		regions[1][1].x = fldPoints[40 * 2]; regions[1][1].y = fldPoints[40 * 2 + 1] + (int)(0.25 * YThresh_underEye);
		regions[1][2].x = fldPoints[41 * 2]; regions[1][2].y = fldPoints[41 * 2 + 1] + (int)(0.25 * YThresh_underEye);
		if (poseType == 2)
		{
			regions[1][3].x = regions[2][4].x; regions[1][3].y = regions[2][4].y;
		}
		else
		{
			regions[1][3].x = (fldPoints[36 * 2] + fldPoints[0 * 2]) / 2; regions[1][3].y = fldPoints[36 * 2 + 1];
		}
		regions[1][5].x = regions[1][2].x; regions[1][5].y = regions[1][2].y + (int)(0.8*YThresh_underEye);
		regions[1][4].x = regions[1][3].x - (2 * XThresh_underEye); regions[1][4].y = regions[1][3].y + (int)(0.6*(regions[1][5].y - regions[1][3].y));
		regions[1][6].x = regions[1][1].x; regions[1][6].y = regions[1][1].y + (int)(0.8*YThresh_underEye);
		regions[1][7].x = regions[1][0].x + XThresh_underEye; regions[1][7].y = regions[1][0].y + (int)(0.6*(regions[1][6].y - regions[1][0].y));
		cv::drawContours(maskImage, regions, 1, cv::Scalar(255), cv::FILLED);
		cv::Rect rect1 = cv::boundingRect(regions[1]);
		if (poseType == 2)
		{
			rect1.width = (rect1.x + rect1.width) - rect0.x;
			rect1.height = std::max(rect0.y + rect0.height, rect1.y + rect1.height) - rect0.y;
			rect1.x = rect0.x;
			rect1.y = rect0.y;
		}
		//left
		if (poseType == 1)
		{
			int XThresh_crowsfeet = fldPoints[16 * 2] - fldPoints[26 * 2];
			regions[2][0].x = fldPoints[26 * 2] + (int)(0.1*XThresh_crowsfeet); regions[2][0].y = fldPoints[26 * 2 + 1];
			regions[2][1].x = fldPoints[16 * 2] - (int)(0.5*XThresh_crowsfeet); regions[2][1].y = fldPoints[26 * 2 + 1];
			regions[2][2].x = fldPoints[16 * 2] - (int)(0.4*XThresh_crowsfeet); regions[2][2].y = fldPoints[45 * 2 + 1];
			regions[2][4].x = fldPoints[45 * 2] + (int)(0.1*XThresh_crowsfeet); regions[2][4].y = fldPoints[45 * 2 + 1];
			regions[2][3].x = (regions[2][2].x + regions[2][4].x) / 2; regions[2][3].y = fldPoints[15 * 2 + 1];
			cv::drawContours(maskImage, regions, 2, cv::Scalar(255), cv::FILLED);
			rect0 = cv::boundingRect(regions[2]);
		}
		XThresh_underEye = (int)((fldPoints[45 * 2] - fldPoints[42 * 2])*0.1);
		regions[1][0].x = fldPoints[42 * 2]; regions[1][0].y = fldPoints[42 * 2 + 1] + (int)(0.15*YThresh_underEye);
		regions[1][1].x = fldPoints[47 * 2]; regions[1][1].y = fldPoints[47 * 2 + 1] + (int)(0.25 * YThresh_underEye);
		regions[1][2].x = fldPoints[46 * 2]; regions[1][2].y = fldPoints[46 * 2 + 1] + (int)(0.25 * YThresh_underEye);
		if (poseType == 1)
		{
			regions[1][3].x = regions[2][4].x; regions[1][3].y = regions[2][4].y;
		}
		else
		{
			regions[1][3].x = (fldPoints[45 * 2] + fldPoints[16 * 2]) / 2; regions[1][3].y = fldPoints[45 * 2 + 1];
		}
		regions[1][5].x = regions[1][2].x; regions[1][5].y = regions[1][2].y + (int)(0.8*YThresh_underEye);
		regions[1][4].x = regions[1][3].x + (2 * XThresh_underEye); regions[1][4].y = regions[1][3].y + (int)(0.6*(regions[1][5].y - regions[1][3].y));
		regions[1][6].x = regions[1][1].x; regions[1][6].y = regions[1][1].y + (int)(0.8*YThresh_underEye);
		regions[1][7].x = regions[1][0].x - XThresh_underEye; regions[1][7].y = regions[1][0].y + (int)(0.6*(regions[1][6].y - regions[1][0].y));
		cv::drawContours(maskImage, regions, 1, cv::Scalar(255), cv::FILLED);
		cv::Rect rect2 = cv::boundingRect(regions[1]);
		if (poseType == 1)
		{
			rect2.width = (rect0.x + rect0.width) - rect2.x;
			rect2.height = std::max(rect0.y + rect0.height, rect2.y + rect2.height) - rect0.y;
			rect2.x = rect2.x;
			rect2.y = rect0.y;
		}
		rects[1].x = rect1.x;
		rects[1].y = std::min(rect1.y, rect2.y);
		rects[1].width = (rect2.x + rect2.width) - rect1.x;
		rects[1].height = std::max(rect1.y + rect1.height, rect2.y + rect2.height) - rects[1].y;
	}
	//nasolabial
	if (zones[2])
	{
		//right
		if (!zones[1])
			regions[1].resize(8);
		regions[1][0].x = fldPoints[31 * 2] - (int)(0.5*(fldPoints[33 * 2] - fldPoints[31 * 2])); regions[1][0].y = fldPoints[31 * 2 + 1];
		regions[1][1].x = fldPoints[48 * 2] - (int)(0.3*(fldPoints[48 * 2] - fldPoints[31 * 2])); regions[1][1].y = (fldPoints[48 * 2 + 1] + fldPoints[31 * 2 + 1]) / 2;
		regions[1][2].x = fldPoints[48 * 2] - (int)(0.1*(fldPoints[48 * 2] - fldPoints[4 * 2])); regions[1][2].y = fldPoints[48 * 2 + 1];
		regions[1][3].x = regions[1][2].x; regions[1][3].y = fldPoints[59 * 2 + 1];
		regions[1][4].x = fldPoints[5 * 2] + (int)(0.3*(fldPoints[48 * 2] - fldPoints[5 * 2])); regions[1][4].y = regions[1][3].y;
		regions[1][5].x = fldPoints[4 * 2] + (int)(0.4*(fldPoints[48 * 2] - fldPoints[4 * 2])); regions[1][5].y = regions[1][2].y;
		regions[1][6].x = (fldPoints[4 * 2] + fldPoints[3 * 2]) / 2 + (int)(0.45*(fldPoints[48 * 2] - (fldPoints[4 * 2] + fldPoints[3 * 2]) / 2)); regions[1][6].y = regions[1][1].y;
		regions[1][7].x = fldPoints[3 * 2] + (int)(0.5*(fldPoints[48 * 2] - fldPoints[3 * 2])); regions[1][7].y = regions[1][0].y;
		cv::drawContours(maskImage, regions, 1, cv::Scalar(255), cv::FILLED);
		cv::Rect rect1 = cv::boundingRect(regions[1]);
		//left
		regions[1][0].x = fldPoints[35 * 2] + (int)(0.5*(fldPoints[35 * 2] - fldPoints[33 * 2])); regions[1][0].y = fldPoints[35 * 2 + 1];
		regions[1][1].x = fldPoints[54 * 2] - (int)(0.3*(fldPoints[54 * 2] - fldPoints[35 * 2])); regions[1][1].y = (fldPoints[54 * 2 + 1] + fldPoints[35 * 2 + 1]) / 2;
		regions[1][2].x = fldPoints[54 * 2] + (int)(0.1*(fldPoints[12 * 2] - fldPoints[54 * 2])); regions[1][2].y = fldPoints[54 * 2 + 1];
		regions[1][3].x = regions[1][2].x; regions[1][3].y = fldPoints[55 * 2 + 1];
		regions[1][4].x = fldPoints[11 * 2] - (int)(0.3*(fldPoints[11 * 2] - fldPoints[54 * 2])); regions[1][4].y = regions[1][3].y;
		regions[1][5].x = fldPoints[12 * 2] - (int)(0.4*(fldPoints[12 * 2] - fldPoints[54 * 2])); regions[1][5].y = regions[1][2].y;
		regions[1][6].x = (fldPoints[12 * 2] + fldPoints[13 * 2]) / 2 - (int)(0.45*((fldPoints[12 * 2] + fldPoints[13 * 2]) / 2 - fldPoints[54 * 2])); regions[1][6].y = regions[1][1].y;
		regions[1][7].x = fldPoints[13 * 2] - (int)(0.5*(fldPoints[13 * 2] - fldPoints[54 * 2])); regions[1][7].y = regions[1][0].y;
		cv::drawContours(maskImage, regions, 1, cv::Scalar(255), cv::FILLED);
		cv::Rect rect2 = cv::boundingRect(regions[1]);
		rects[2].x = rect1.x;
		rects[2].y = std::min(rect1.y, rect2.y);
		rects[2].width = (rect2.x + rect2.width) - rect1.x;
		rects[2].height = std::max(rect1.y + rect1.height, rect2.y + rect2.height) - rects[2].y;
	}
}

void Beauty::MakeupFeatures_Live::wrinkles/*complexionZoneLevel*/(cv::Mat& inputImage, const int fldPoints[], bool zones[], float intensity[])
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rects[3];
	getComplexionMaskZoneLevel(maskImage, fldPoints, zones, rects);
	cv::Mat complexionImage = inputImage.clone();
	//fixed size filter is applied as the users wanted to see complexion at same intensity on small faces like bigger faces(which will not be same when filter is calculated from faces size).
	cv::Rect faceRect = Utilities_Live::getFaceRectangle(fldPoints);
	if (zones[0])//forehead
	{
		int kernalFace = (int)(faceRect.width*0.03*intensity[0]);
		if (kernalFace % 2 == 0)
			kernalFace++;
		if (kernalFace < 3)
			kernalFace = 3;
		int kernalComplexion = 11;//to smooth wrinkles
		if (!Utilities_Live::rectanglePadding(rects[0], std::max(kernalFace, kernalComplexion), inputImage.cols, inputImage.rows))
			return;
		cv::Mat tempImage = maskImage(rects[0]);
		cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
		cv::blur(inputImage(rects[0]), complexionImage(rects[0]), cv::Size(kernalComplexion, kernalComplexion));
	}
	if (zones[1])//undereye
	{
		int kernalFace = (int)(faceRect.width*0.02*intensity[1]);
		if (kernalFace % 2 == 0)
			kernalFace++;
		if (kernalFace < 3)
			kernalFace = 3;
		int kernalComplexion = 9;//to smooth wrinkles
		if (!Utilities_Live::rectanglePadding(rects[1], std::max(kernalFace, kernalComplexion), inputImage.cols, inputImage.rows))
			return;
		cv::Mat tempImage = maskImage(rects[1]);
		cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
		cv::blur(inputImage(rects[1]), complexionImage(rects[1]), cv::Size(kernalComplexion, kernalComplexion));
	}
	if (zones[2])//nasolabial
	{
		int kernalFace = (int)(faceRect.width*0.03*intensity[2]);
		if (kernalFace % 2 == 0)
			kernalFace++;
		if (kernalFace < 3)
			kernalFace = 3;
		int kernalComplexion = 15;//to smooth folds
		if (!Utilities_Live::rectanglePadding(rects[2], std::max(kernalFace, kernalComplexion), inputImage.cols, inputImage.rows))
			return;
		cv::Mat tempImage = maskImage(rects[2]);
		cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
		cv::blur(inputImage(rects[2]), complexionImage(rects[2]), cv::Size(kernalComplexion, kernalComplexion));
	}
	int minX = inputImage.cols, maxX = 0, minY = inputImage.rows, maxY = 0;
	int endY[3] = {};
	for (int i = 0; i < 3; i++)
	{
		if (rects[i].width > 0)
		{
			if (rects[i].x < minX)
				minX = rects[i].x;
			if (rects[i].y < minY)
				minY = rects[i].y;
			if (rects[i].x + rects[i].width - 1 > maxX)
				maxX = rects[i].x + rects[i].width - 1;
			if (rects[i].y + rects[i].height - 1 > maxY)
				maxY = rects[i].y + rects[i].height - 1;
			endY[i] = maxY;
		}
	}
	faceRect = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);

	cv::Mat inputROIImage = inputImage(faceRect);
	cv::Mat maskROIImage = maskImage(faceRect);
	cv::Mat complexionROIImage = complexionImage(faceRect);
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	Utilities_Live::saveImage(complexionROIImage, "complexion.png");
	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	const uchar*  complexionPtr = complexionROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;
	int complexionStep = (int)complexionROIImage.step;

	float R, G, B, NR, NG, NB, sk;
	float temp = 1.f / 255;
	int length = inputROIImage.cols * 3;
	int index = 0, temp2 = (endY[index] - minY + 1);
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		if (temp2 < 0 || i == temp2)
		{
			index++;
			temp2 = (endY[index] - minY + 1);
		}
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * temp * intensity[index];
			if (sk > 0)
			{
				R = inputPtr[j];
				G = inputPtr[j + 1];
				B = inputPtr[j + 2];
				NR = complexionPtr[j];
				NG = complexionPtr[j + 1];
				NB = complexionPtr[j + 2];
				if (NR - R > 0)
					inputPtr[j] = (uchar)(R * (1 - sk) + NR * sk);
				if (NG - G > 0)
					inputPtr[j + 1] = (uchar)(G  * (1 - sk) + NG * sk);
				if (NB - B > 0)
					inputPtr[j + 2] = (uchar)(B  * (1 - sk) + NB * sk);
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		complexionPtr += complexionStep;
	}
}

void Beauty::MakeupFeatures_Live::brightness(cv::Mat& inputImage, const int fldPoints[], float intensity)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFoundationMask(maskImage, fldPoints);
	int kernalFace = (int)(rect.width * 0.04 * intensity);
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
		return;
	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage = maskImage(rect);
	int moustache = fldPoints[33 * 2 + 1] - rect.y;
	float RGB[3] = {};
	Utilities_Live::averageRGB(inputROIImage, maskROIImage, moustache, RGB, 0.20f, 0.15f, "skinLab");
	float ratio1 = RGB[0] / RGB[1];
	float ratio2 = RGB[2] / RGB[1];
	int foreheadLoc = (int)((fldPoints[68 * 2 + 1] + fldPoints[74 * 2 + 1])* 0.5) - rect.y;
	int eyebrowLoc = (int)((fldPoints[17 * 2 + 1] + fldPoints[26 * 2 + 1]) * 0.5) - rect.y;
	int startY = eyebrowLoc; int endY = maskROIImage.rows - 1;
	cv::Rect featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	cv::Mat tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
	startY = foreheadLoc; endY = eyebrowLoc + kernalEyebrow;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalEyebrow, kernalEyebrow));
	startY = 0; endY = foreheadLoc + kernalForehead;
	featureRect = cv::Rect(0, startY, maskROIImage.cols, (endY - startY + 1));
	if (!Utilities_Live::rectanglePadding(featureRect, 0, inputImage.cols, inputImage.rows))
		return;
	tempImage = maskROIImage(featureRect);
	cv::blur(tempImage, tempImage, cv::Size(kernalForehead, kernalForehead));
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;

	float R, G, B, sk;
	int length = inputROIImage.cols * 3;
	int overSaturatedpixels = 0;
	float overSaturationLimit = rect.width *  rect.height * 0.1f;
	intensity *= 10;
	if (intensity >= brightness_overSaturatedValue) {
		if (brightness_isOverSaturated)
			intensity = brightness_overSaturatedValue;
		else
			intensity = brightness_overSaturatedValue - brightness_overSaturationStep;
	}
	//std::cout << brightness_overSaturatedValue << "," << intensity << std::endl;
	float brightnessValue = intensity * 2.55f;
	float resValue = (1.f / 255)*brightnessValue;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * resValue;
			if (sk > 0)
			{
				R = inputPtr[j];
				G = inputPtr[j + 1];
				B = inputPtr[j + 2];
				R = R + (ratio1 * sk);
				G = G + sk;
				B = B + (ratio2 * sk);
				if (R > 255)
					R = 255;
				if (G > 255)
					G = 255;
				if (B > 255)
					B = 255;
				if (R == 255 || G == 255 || B == 255)
					overSaturatedpixels++;
				inputPtr[j] = (uchar)R;
				inputPtr[j + 1] = (uchar)G;
				inputPtr[j + 2] = (uchar)B;
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
	}
	if (overSaturatedpixels > overSaturationLimit) {
		brightness_overSaturatedValue -= brightness_overSaturationStep;
		brightness_isOverSaturated = true;
	}
	else {
		if (brightness_overSaturatedValue > brightness_maxValue) {
			brightness_overSaturatedValue = brightness_maxValue;
		}
		brightness_overSaturatedValue += brightness_overSaturationStep;
		brightness_isOverSaturated = false;
	}
}

cv::Rect Beauty::MakeupFeatures_Live::getLipMask(cv::Mat& maskImage, cv::Mat& maskImage2, const int fldPoints[])
{
	std::vector<std::vector<cv::Point>> contours(2);
	//fill lip mask
	contours[0].reserve(12);
	for (int i = 48; i <= 59; i++)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	cv::drawContours(maskImage, contours, 0, cv::Scalar(255), cv::FILLED);
	cv::Rect rect = cv::boundingRect(contours[0]);
	//remove innerlip
	contours[1].reserve(8);
	for (int i = 60; i <= 67; i++)
	{
		contours[1].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	cv::drawContours(maskImage, contours, 1, cv::Scalar(0), cv::FILLED);

	//bottom lip mask
	if (!maskImage2.empty())
	{
		int j = 0;
		contours[0][j].x = fldPoints[48 * 2]; contours[0][j++].y = fldPoints[48 * 2 + 1];
		contours[0][j].x = fldPoints[60 * 2]; contours[0][j++].y = fldPoints[60 * 2 + 1];
		for (int i = 67; i >= 64; i--)
		{
			contours[0][j].x = fldPoints[i * 2]; contours[0][j++].y = fldPoints[i * 2 + 1];
		}
		contours[0][j].x = fldPoints[54 * 2]; contours[0][j++].y = fldPoints[54 * 2 + 1];
		for (int i = 55; i <= 59; i++)
		{
			contours[0][j].x = fldPoints[i * 2]; contours[0][j++].y = (int)((fldPoints[i * 2 + 1] + fldPoints[66 * 2 + 1]) * 0.5);
		}
		cv::drawContours(maskImage2, contours, 0, cv::Scalar(255), cv::FILLED);
	}
	return rect;
}

void Beauty::MakeupFeatures_Live::lipstick(cv::Mat& inputImage, double lowResFactor, const int lowResFldPoints[], int color[], float coverage, float intensity, float glossValue, float featheringFactor)
{
	int lowResRows = (int)(inputImage.rows / lowResFactor), lowResCols = (int)(inputImage.cols / lowResFactor);
	cv::Mat maskImage = cv::Mat::zeros(lowResRows, lowResCols, CV_8UC1);
	cv::Mat bottomLipMaskImage = cv::Mat::zeros(lowResRows, lowResCols, CV_8UC1);
	cv::Rect lowResRect = getLipMask(maskImage, bottomLipMaskImage, lowResFldPoints);
	int kernalLip = (int)(lowResRect.width * 0.04 * 3 * coverage*featheringFactor);
	if (kernalLip % 2 == 0)
		kernalLip++;
	if (kernalLip < 3)
		kernalLip = 3;
	int kernalBottomLip = (int)(lowResRect.width * 0.02 * 3 * glossValue);
	if (kernalBottomLip % 2 == 0)
		kernalBottomLip++;
	if (kernalBottomLip < 3)
		kernalBottomLip = 3;
	int kernalGloss = (int)(lowResRect.width * 0.1);
	if (kernalGloss % 2 == 0)
		kernalGloss++;
	if (kernalGloss < 3)
		kernalGloss = 3;
	if (!Utilities_Live::rectanglePadding(lowResRect, std::max(kernalLip, std::max(kernalBottomLip, kernalGloss)), lowResCols, lowResRows))
		return;

	float ratio0, ratio1, ratio2;
	int maxChannel;
	if (color[0] == 0 && color[1] == 0 && color[2] == 0)//black color
	{
		ratio0 = ratio1 = ratio2 = 0;
		maxChannel = 0;//can be anything
	}
	else
	{
		if (color[0] >= color[1] && color[0] >= color[2])
		{
			ratio0 = 1;
			ratio1 = (float)color[1] / color[0];
			ratio2 = (float)color[2] / color[0];
			maxChannel = 0;
		}
		else
		{
			if (color[1] >= color[2])
			{
				ratio0 = (float)color[0] / color[1];
				ratio1 = 1;
				ratio2 = (float)color[2] / color[1];
				maxChannel = 1;
			}
			else
			{
				ratio0 = (float)color[0] / color[2];
				ratio1 = (float)color[1] / color[2];
				ratio2 = 1;
				maxChannel = 2;
			}
		}
	}
	cv::Rect rect((int)(lowResRect.x*lowResFactor), (int)(lowResRect.y*lowResFactor), (int)(lowResRect.width*lowResFactor), (int)(lowResRect.height*lowResFactor));

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage = maskImage(lowResRect);
	cv::Mat shineImage;
	cv::resize(inputROIImage, shineImage, cv::Size(lowResRect.width, lowResRect.height));
	int moustache = shineImage.rows;
	float RGB[3] = {};
	Utilities_Live::averageRGB(shineImage, maskROIImage, moustache, RGB, 0.8f, 0.1f, "skinLab");
	float intensityFactor = (color[maxChannel] / RGB[maxChannel]) - 1;
	intensityFactor *= intensity;
	cv::Mat channels[3];
	cv::split(shineImage, channels);
	cv::blur(channels[1], channels[1], cv::Size(kernalGloss, kernalGloss));
	cv::resize(channels[1], shineImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(shineImage, "positive.png");

	cv::blur(maskROIImage, maskROIImage, cv::Size(kernalLip, kernalLip));
	cv::resize(maskROIImage, maskROIImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	cv::Mat bottomLipMaskROIImage = bottomLipMaskImage(lowResRect);
	cv::blur(bottomLipMaskROIImage, bottomLipMaskROIImage, cv::Size(kernalBottomLip, kernalBottomLip));
	cv::resize(bottomLipMaskROIImage, bottomLipMaskROIImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(bottomLipMaskROIImage, "bottomlipmask.png");

	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	const uchar*  bottomLipMaskPtr = bottomLipMaskROIImage.data;
	const uchar* shinePtr = shineImage.data;
	int inputStep = (int)inputROIImage.step;
	int  maskStep = (int)maskROIImage.step;
	int  bottomLipmaskStep = (int)bottomLipMaskROIImage.step;
	int shineStep = (int)shineImage.step;

	//Simulate lipstick 
	glossValue *= 4;
	float  R, G, B, P, sk;
	int shine;
	float resValue = (1.f / 255)*coverage;
	float resValue1 = (1.f / 255)* glossValue;//to reduce computation inside loop
	int length = inputROIImage.cols * 3;
	float I = 1 + intensityFactor;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * resValue;
			if (sk > 0)
			{
				R = inputPtr[j];
				G = inputPtr[j + 1];
				B = inputPtr[j + 2];
				P = inputPtr[j + maxChannel];
				R = R * (1 - sk) + P * ratio0*I  * sk;
				G = G * (1 - sk) + P * ratio1*I  * sk;
				B = B * (1 - sk) + P * ratio2*I  * sk;
				sk = bottomLipMaskPtr[k] * resValue1;
				if (sk > 0)
				{
					shine = inputPtr[j + 1] - shinePtr[k];
					if (shine < 0)
						shine = 0;
					P = sk * shine;
					R += P;
					G += P;
					B += P;
				}
				if (R > 255)
					R = 255;
				if (G > 255)
					G = 255;
				if (B > 255)
					B = 255;
				inputPtr[j] = (uchar)R;
				inputPtr[j + 1] = (uchar)G;
				inputPtr[j + 2] = (uchar)B;
			}
		}
		maskPtr += maskStep;
		bottomLipMaskPtr += bottomLipmaskStep;
		inputPtr += inputStep;
		shinePtr += shineStep;
	}
}

void Beauty::MakeupFeatures_Live::lipHealth(cv::Mat& inputImage, const int fldPoints[], float intensity)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Mat bottomLipMaskImage;
	cv::Rect rect = getLipMask(maskImage, bottomLipMaskImage, fldPoints);
	int kernalLip = (int)(rect.width*0.04 * 2 * intensity);
	if (kernalLip % 2 == 0)
		kernalLip++;
	if (kernalLip < 3)
		kernalLip = 3;
	int kernalComplexion = 5;
	if (!Utilities_Live::rectanglePadding(rect, std::max(kernalLip, kernalComplexion), inputImage.cols, inputImage.rows))
		return;

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat maskROIImage;
	cv::blur(maskImage(rect), maskROIImage, cv::Size(kernalLip, kernalLip));
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	cv::Mat complexionROIImage;
	cv::blur(inputROIImage, complexionROIImage, cv::Size(kernalComplexion, kernalComplexion));
	uchar* inputPtr = inputROIImage.data;
	const uchar* maskPtr = maskROIImage.data;
	const uchar* complexionPtr = complexionROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;
	int complexionStep = (int)complexionROIImage.step;

	//Simulate lip health
	float R, G, B, NR, NG, NB, sk;
	float resValue = (1.f / 255)*intensity;
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
				G = inputPtr[j + 1];
				B = inputPtr[j + 2];
				NR = complexionPtr[j];
				NG = complexionPtr[j + 1];
				NB = complexionPtr[j + 2];
				if (NR - R < 0)
					NR = R;
				if (NG - G < 0)
					NG = G;
				if (NB - B < 0)
					NB = B;
				NR *= 1.1f;
				NG *= 1.04f;
				NB *= 1.08f;
				if (NR > 255)
					NR = 255;
				if (NG > 255)
					NG = 255;
				if (NB > 255)
					NB = 255;
				inputPtr[j] = (uchar)(R * (1 - sk) + NR * sk);
				inputPtr[j + 1] = (uchar)(G  * (1 - sk) + NG * sk);
				inputPtr[j + 2] = (uchar)(B  * (1 - sk) + NB * sk);
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		complexionPtr += complexionStep;
	}
}
