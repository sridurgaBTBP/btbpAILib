#include "Utilities_Live.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

std::string  Utilities_Live::path;
bool Utilities_Live::isMorphStable = false;
int Utilities_Live::fps = 1;
cv::Point Utilities_Live::previousFaceSplitPoints[];
cv::Point Utilities_Live::previousROISplitPoints[];

cv::Rect Utilities_Live::getFaceRectangle(const int fldPoints[])
{
	int minX = fldPoints[0], maxX = fldPoints[0];//0*2
	int minY = fldPoints[1], maxY = fldPoints[1];//0*2+1
	for (int i = 1; i <= 16; i++)
	{
		if (fldPoints[i * 2] < minX)
			minX = fldPoints[i * 2];
		else if (fldPoints[i * 2] > maxX)
			maxX = fldPoints[i * 2];

		if (fldPoints[i * 2 + 1] < minY)
			minY = fldPoints[i * 2 + 1];
		else if (fldPoints[i * 2 + 1] > maxY)
			maxY = fldPoints[i * 2 + 1];
	}
	for (int i = 68; i <= 74; i++)
	{
		if (fldPoints[i * 2] < minX)
			minX = fldPoints[i * 2];
		else if (fldPoints[i * 2] > maxX)
			maxX = fldPoints[i * 2];

		if (fldPoints[i * 2 + 1] < minY)
			minY = fldPoints[i * 2 + 1];
		else if (fldPoints[i * 2 + 1] > maxY)
			maxY = fldPoints[i * 2 + 1];
	}
	return cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

/**
 splitViews 

 @param inputImage original frame
 @param outputImage applied effect frame
 @param fldPoints points
 @param viewType split face or screen <- 0 split face> : <- !=0 split screen>
 @param params <param[0] - 0 - right side effect on> <param[1] - window width> <param[2] -view width
 @param out_splitFacePoints - to draw the points line on the canvas (web and android benifits)
 */
void Utilities_Live::splitViews(const cv::Mat& inputImage, cv::Mat& outputImage, const int fldPoints[], const int viewType, const int* params, int* out_splitFacePoints)
{
	cv::Mat maskImage = cv::Mat::zeros(inputImage.rows, inputImage.cols, CV_8UC1);
	cv::Rect rect = getFaceRectangle(fldPoints);
	int kernalMax = (int)(2.5 * rect.width * 0.04);
	if (kernalMax % 2 == 0)
		kernalMax++;
	if (kernalMax < 3)
		kernalMax = 3;
	//This is the maximum featuring kernal(or maximum featuring effect) that is applied in entire features.Need to update this if more larger kernal is used.At present it is used in foundation and brightness as kernalForehead.
	if (!rectanglePadding(rect, kernalMax, inputImage.cols, inputImage.rows))
		return;

	int endX = rect.x + rect.width, endY = rect.y + rect.height;
	std::vector<std::vector<cv::Point>> contours(1);
	if (viewType == 0)
	{// split face
		contours[0].push_back(cv::Point(fldPoints[71 * 2], rect.y));
		contours[0].push_back(cv::Point(fldPoints[27 * 2], fldPoints[27 * 2 + 1]));
		contours[0].push_back(cv::Point(fldPoints[30 * 2], fldPoints[30 * 2 + 1]));
		contours[0].push_back(cv::Point(fldPoints[33 * 2], fldPoints[33 * 2 + 1]));
		contours[0].push_back(cv::Point(fldPoints[51 * 2], fldPoints[51 * 2 + 1]));
		contours[0].push_back(cv::Point(fldPoints[57 * 2], fldPoints[57 * 2 + 1]));
		contours[0].push_back(cv::Point(fldPoints[8 * 2], endY));
		int splitFaceSide = 0;
		if (params != nullptr)
		{
			splitFaceSide = params[0];
		}
		if (splitFaceSide == 0)
		{
			contours[0].push_back(cv::Point(endX, endY));
			contours[0].push_back(cv::Point(endX, rect.y));
		}
		else
		{
			contours[0].push_back(cv::Point(rect.x, endY));
			contours[0].push_back(cv::Point(rect.x, rect.y));
		}
		cv::drawContours(maskImage, contours, 0, cv::Scalar(255), cv::FILLED, 8);
	}
	else
	{//fixed points - split screen

		if (params == nullptr)
			return;
		float splitScreenAnchor = params[0] / 100.f;
		int windowWidth = params[1];
		int viewWidth = params[2];
		int orgOffest = (viewWidth - windowWidth) / 2;
		int splitPointX = (int)(((splitScreenAnchor * windowWidth) + orgOffest) * ((float)inputImage.cols / viewWidth));
		contours[0].push_back(cv::Point(splitPointX, rect.y));
		contours[0].push_back(cv::Point(splitPointX, endY));
		contours[0].push_back(cv::Point(endX, endY));
		contours[0].push_back(cv::Point(endX, rect.y));
		cv::drawContours(maskImage, contours, 0, cv::Scalar(255), cv::FILLED, 8);
	}
	cv::Mat inputROIImage = inputImage(rect);
	cv::Mat outputROIImage = outputImage(rect);
	cv::Mat maskROIImage = maskImage(rect);
	Utilities_Live::saveImage(maskROIImage, "splitmask.png");
	const uchar*  inputPtr = inputROIImage.data;
	uchar* outputPtr = outputROIImage.data;
	const uchar*  maskPtr = maskROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int ouputStep = (int)outputROIImage.step;
	int maskStep = (int)maskROIImage.step;

	int length = inputROIImage.cols * 3;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			if (maskPtr[++k] == 0)
			{
				outputPtr[j] = inputPtr[j];
				outputPtr[j + 1] = inputPtr[j + 1];
				outputPtr[j + 2] = inputPtr[j + 2];
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		outputPtr += ouputStep;
	}

	if (viewType == 0)
	{
		int n = (int)contours[0].size() - 2;//last two points are not needed.
        
		for (int i = 0; i < n; i++)
		{
			out_splitFacePoints[i * 2] = contours[0][i].x;
			out_splitFacePoints[i * 2 + 1] = contours[0][i].y;
		}
	}
}

bool Utilities_Live::rectanglePadding(cv::Rect& rect, int offset, int maxX, int maxY)
{
	rect.x -= offset;
	rect.y -= offset;
	rect.width += (2 * offset);
	rect.height += (2 * offset);
	if (rect.x < 0)
	{
		rect.width = rect.width + rect.x;
		rect.x = 0;
	}
	if (rect.y < 0)
	{
		rect.height = rect.height + rect.y;
		rect.y = 0;
	}
	if (rect.width + rect.x > maxX)
	{
		rect.width = maxX - rect.x;
	}
	if (rect.height + rect.y > maxY)
	{
		rect.height = maxY - rect.y;
	}
	if (rect.width <= 0 || rect.height <= 0)
		return false;//padding fails;
	else
		return true;
}

void Utilities_Live::averageRGB(const cv::Mat& inputImage, const cv::Mat& maskImage, int moustache, float RGB[], float lowLimitPer, float highLimitPer, std::string skinLab)
{
	double cnt = 0;
	int pixels[256] = {};
	int length = inputImage.cols * 3;
	if (moustache > inputImage.rows)
		moustache = inputImage.rows;
	const uchar* inputPtr = inputImage.data;
	const uchar* maskPtr = maskImage.data;
	int inputStep = (int)inputImage.step;
	int maskStep = (int)maskImage.step;
	cv::Mat colorGUIImage = cv::Mat::zeros(inputImage.size(), inputImage.type());   // created a GUI image to observe oversaturated(shine), understaturated area  
	uchar* UIptr = colorGUIImage.data;
	int UIStep = (int)colorGUIImage.step;
	for (int i = 0; i < moustache; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			if (maskPtr[++k] > 0)
			{
				pixels[inputPtr[j]]++;
				cnt++;
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
}
	int lowLimit = (int)(lowLimitPer*cnt);
	int highLimit = (int)(highLimitPer*cnt);
	cnt = 0;
	int lowIntensity = 0, highIntensity = 255;
	for (int i = 0; i < 256; i++)
	{
		cnt += pixels[i];
		pixels[i] = 0;
		if (cnt > lowLimit)
		{
			lowIntensity = i;
			break;
		}
	}
	cnt = 0;
	for (int i = 255; i >= 0; i--)
	{
		cnt += pixels[i];
		pixels[i] = 0;
		if (cnt > highLimit)
		{
			highIntensity = i;
			break;
		}
	}
	cnt = 0;
	double R = 0, G = 0, B = 0, P = 0;
	inputPtr = inputImage.data;
	maskPtr = maskImage.data;
	for (int i = 0; i < moustache; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			P = inputPtr[j];
			if (maskPtr[++k] > 0 && P >= lowIntensity && P <= highIntensity)
			{
				R += P;
				G += inputPtr[j + 1];
				B += inputPtr[j + 2];
				cnt++;
				UIptr[j + 1] = 255;
			}
			else if(maskPtr[k] > 0 && P < lowIntensity)
			{
				UIptr[j + 2] = 255;       // under saturated area
			}
			else if (maskPtr[k] > 0 && P > highIntensity)
			{
				UIptr[j] = 255;   // oversaturated area
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
		UIptr += UIStep;
	}
	if (cnt > 0)
	{
		RGB[0] = (float)(R / cnt);
		RGB[1] = (float)(G / cnt);
		RGB[2] = (float)(B / cnt);
	}
	cv::imwrite(skinLab + ".jpg", colorGUIImage);
}



//#define TESTING
void Utilities_Live::saveImage(const cv::Mat& image, std::string name)
{
#ifdef TESTING
	if (image.channels() > 1)
	{
		cv::Mat outImage;
		cv::cvtColor(image, outImage, cv::COLOR_RGB2BGR);
		cv::imwrite(path + name, outImage);
	}
	else
	{
		cv::imwrite(path + name, image);
	}
#endif //TESTING
}
