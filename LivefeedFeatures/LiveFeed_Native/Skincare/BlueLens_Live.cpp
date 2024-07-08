#include "BlueLens_Live.h"
#include "Utilities_Live.h"
#include <opencv2/imgproc.hpp>

cv::Rect Skincare::BlueLens_Live::getMask(cv::Mat& maskImage, const int fldPoints[], bool isBlurBG)
{
	cv::Rect rect = Utilities_Live::getFaceRectangle(fldPoints);
	int erosionOffset;
	if (isBlurBG)
		erosionOffset = (int)(rect.width*0.035);
	else
		erosionOffset = 0;
	std::vector<std::vector<cv::Point>> contours(1);
	//fill face mask
	contours[0].reserve(24);
	for (int i = 0; i <= 7; i++)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2] + erosionOffset, fldPoints[i * 2 + 1]));
	}
	contours[0].push_back(cv::Point(fldPoints[8 * 2], fldPoints[8 * 2 + 1]));
	for (int i = 9; i <= 16; i++)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2] - erosionOffset, fldPoints[i * 2 + 1]));
	}
	for (int i = 74; i >= 68; i--)
	{
		contours[0].push_back(cv::Point(fldPoints[i * 2], fldPoints[i * 2 + 1]));
	}
	cv::fillPoly(maskImage, contours, cv::Scalar(255));
	return cv::boundingRect(contours[0]);
}

void Skincare::BlueLens_Live::performBlueLens(cv::Mat& inputImage, const int fldPoints[], bool isBlurBG, bool isBlueImage, int channel, int colorPercent, double clipLimit)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getMask(maskImage, fldPoints, isBlurBG);
	int kernalFace = (int)(rect.width * 0.05);
	if (kernalFace % 2 == 0)
		kernalFace++;
	if (kernalFace < 3)
		kernalFace = 3;
	const int resizeFactor = 4;
	if (!Utilities_Live::rectanglePadding(rect, (kernalFace*resizeFactor), inputImage.cols, inputImage.rows))
		return;
	cv::Mat maskROIImage = maskImage(rect);
	cv::Mat tempImage;
	cv::resize(maskROIImage, tempImage, cv::Size(maskROIImage.cols / resizeFactor, maskROIImage.rows / resizeFactor));
	cv::blur(tempImage, tempImage, cv::Size(kernalFace, kernalFace));
	cv::resize(tempImage, maskROIImage, cv::Size(maskROIImage.cols, maskROIImage.rows));
	Utilities_Live::saveImage(maskROIImage, "mask.png");

	cv::Mat channels[3], clacheImage;
	cv::split(inputImage, channels);
	if (channel == -1)
	{
		clacheImage = channels[1] / 2 + channels[2] / 2;
	}
	else
	{
		clacheImage = channels[channel];
	}
	cv::Mat blurImage;
	if (isBlurBG)
	{
		if (channel != 1)
		{
			blurImage = channels[1];
		}
		else
		{
			blurImage = clacheImage.clone();//make a copy as clache will change this G channel image.
		}
	}
	if (clipLimit > 0)
	{
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(32, 32));
		clahe->apply(clacheImage, clacheImage);
	}

	cv::Mat blurROIImage;
	if (isBlurBG)
	{
		cv::resize(blurImage, tempImage, cv::Size(inputImage.cols / 4, inputImage.rows / 4));
		//filter size for back ground blurring should be set based on image resolution but not based on face size
		cv::blur(tempImage, tempImage, cv::Size(3, 3));
		cv::resize(tempImage, blurImage, cv::Size(inputImage.cols, inputImage.rows));
		blurROIImage = blurImage(rect);
	}

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat clacheROIImage = clacheImage(rect);
	uchar*  inputPtr = inputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	const uchar* clachePtr = clacheROIImage.data;
	const uchar* blurPtr = blurROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;
	int clacheStep = (int)clacheROIImage.step;
	int blurStep = (int)blurROIImage.step;

	float B, sk, I;
	float alpha = colorPercent / 100.f;
	float temp = 1.f / 255;
	if (!isBlurBG&&clipLimit == 0)
		temp *= alpha;
	int length = inputROIImage.cols * 3;
	float factorG, factorR;
	if (isBlueImage)
	{
		factorG = 0.95f;
		factorR = 0.2f;
	}
	else
	{
		factorG = 1;
		factorR = 1;
	}
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * temp;
			B = clachePtr[k];
			if (isBlurBG)
			{
				I = blurPtr[k] * 0.25f;
				if (sk > 0)
				{
					if (clipLimit > 0)
					{
						inputPtr[j] = (uchar)(I * (1 - sk) + B*factorR * sk);
						inputPtr[j + 1] = (uchar)(I * (1 - sk) + B *factorG* sk);
						inputPtr[j + 2] = (uchar)(I * (1 - sk) + B * sk);
					}
					else
					{
						inputPtr[j] = (uchar)(I * (1 - sk) + (inputPtr[j] * (1 - alpha) + B*alpha *factorR)* sk);
						inputPtr[j + 1] = (uchar)(I * (1 - sk) + (inputPtr[j + 1] * (1 - alpha) + B*alpha *factorG)* sk);
						inputPtr[j + 2] = (uchar)(I * (1 - sk) + (inputPtr[j + 2] * (1 - alpha) + B*alpha) * sk);
					}
				}
				else
				{
					inputPtr[j] = (uchar)I;
					inputPtr[j + 1] = (uchar)I;
					inputPtr[j + 2] = (uchar)I;
				}
			}
			else
			{
				if (sk > 0)
				{
					inputPtr[j] = (uchar)(inputPtr[j] * (1 - sk) + B*factorR * sk);
					inputPtr[j + 1] = (uchar)(inputPtr[j + 1] * (1 - sk) + B *factorG * sk);
					inputPtr[j + 2] = (uchar)(inputPtr[j + 2] * (1 - sk) + B * sk);
				}
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		clachePtr += clacheStep;
		blurPtr += blurStep;
	}
	if (isBlurBG)
	{
		background(inputImage, blurImage, cv::Rect(0, 0, rect.x, inputImage.rows));
		background(inputImage, blurImage, cv::Rect(rect.x + rect.width - 1, 0, inputImage.cols - (rect.x + rect.width), inputImage.rows));
		background(inputImage, blurImage, cv::Rect(rect.x, 0, rect.width, rect.y));
		background(inputImage, blurImage, cv::Rect(rect.x, rect.y + rect.height - 1, rect.width, inputImage.rows - (rect.y + rect.height)));
	}
}

void Skincare::BlueLens_Live::background(cv::Mat& inputImage, cv::Mat& blurImage, cv::Rect rect)
{

	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat blurROIImage = blurImage(rect);

	uchar* inputPtr = inputROIImage.data;
	const uchar* blurPtr = blurROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int blurStep = (int)blurROIImage.step;
	int length = inputROIImage.cols * 3;

	uchar I;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			I = (uchar)(blurPtr[++k] * 0.25f);
			inputPtr[j] = I;
			inputPtr[j + 1] = I;
			inputPtr[j + 2] = I;
		}
		inputPtr += inputStep;
		blurPtr += blurStep;
	}
}
