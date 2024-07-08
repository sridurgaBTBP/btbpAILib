#include "Redness_Live.h"
//#include <android/log.h>

//#define LOG_TAG "Redness"
//#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

void Skincare::Redness_Live::performRedness(cv::Mat& inputImage, int fldPoints[], bool IsLocal, double offset)
{
	const int rows = inputImage.rows, cols = inputImage.cols;
	int i;
	//LOGD("Redness: Foundation mask creation started");
#pragma region creating foundation mask
	std::vector<std::vector<cv::Point>> faceContour(1);
	cv::Point P1;

	for (i = 0; i <= 16; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		faceContour[0].push_back(P1);
	}
	for (i = 74; i >= 68; i--)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		faceContour[0].push_back(P1);
	}
	foundationmaskImage = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::drawContours(foundationmaskImage, faceContour, 0, cv::Scalar(255), cv::FILLED, 8);

	std::vector<std::vector<cv::Point>> removedcontours(4), removedcontours1(3);

	for (i = 48; i <= 59; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[0].push_back(P1);
		removedcontours1[0].push_back(P1);
	}
	for (i = 17; i <= 21; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[1].push_back(P1);
		removedcontours1[1].push_back(P1);
	}
	for (i = 39; i <= 41; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[1].push_back(P1);
		removedcontours1[1].push_back(P1);
	}
	P1.x = fldPoints[36 * 2];
	P1.y = fldPoints[36 * 2 + 1];
	removedcontours[1].push_back(P1);
	removedcontours1[1].push_back(P1);
	for (i = 22; i <= 26; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[2].push_back(P1);
		removedcontours1[2].push_back(P1);
	}
	for (i = 45; i <= 47; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[2].push_back(P1);
		removedcontours1[2].push_back(P1);
	}
	P1.x = fldPoints[42 * 2];
	P1.y = fldPoints[42 * 2 + 1];
	removedcontours[2].push_back(P1);
	removedcontours1[2].push_back(P1);

	for (i = 30; i <= 35; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[3].push_back(P1);
	}
#pragma endregion
	//LOGD("Redness: Foundation mask creation completed");

	int SkinAreaHeight = fldPoints[8 * 2 + 1] - fldPoints[71 * 2 + 1];

	//LOGD("Redness: Negative image creation started");
#pragma region Redness negative image creation
	int filterSize = (int)(SkinAreaHeight * 0.0023);
	if (filterSize <= 7)
	{
		filterSize = 7;
	}
	if (filterSize % 2 == 0)
	{
		filterSize++;
	}
	int resizefactor = (int)(SkinAreaHeight * 0.0019);
	if (resizefactor < 2)
	{
		resizefactor = 2;
	}
	int Smoothiterations = (int)(SkinAreaHeight * 0.0056);
	if (Smoothiterations < 5)
	{
		Smoothiterations = 5;
	}
	cv::resize(inputImage, vertImage, cv::Size(cols / resizefactor, rows / resizefactor), 0, 0, 1);
	for (i = 0; i < Smoothiterations; i++)
	{
		cv::blur(vertImage, vertImage, cv::Size(filterSize, filterSize), cv::Point(-1, -1), 1);
	}
	cv::resize(vertImage, negativeImage, cv::Size(cols, rows), 0, 0, 1);
	cv::subtract(negativeImage, inputImage, negativeImage);

#pragma endregion

	//LOGD("Redness: Negative image creation completed");

	//LOGD("Redness: Foundation mask creation started");
#pragma region Heavy + light feathering

	int featheringFilter = 0, featheringIterations = 0;
	if (SkinAreaHeight > 1500)
	{
		featheringFilter = (int)(SkinAreaHeight * 0.012);
		featheringIterations = (int)(SkinAreaHeight * 0.02);
	}
	else
	{
		featheringFilter = (int)(SkinAreaHeight * 0.05);
		featheringIterations = (int)(SkinAreaHeight * 0.02);
	}
	if (featheringFilter < 3)
	{
		featheringFilter = 3;
	}
	if (featheringFilter % 2 == 0)
	{
		featheringFilter++;
	}
	if (featheringIterations == 0)
	{
		featheringIterations = 2;
	}
	cv::resize(foundationmaskImage, vertmaskImage, cv::Size(cols / 2, rows / 2), 0, 0, 1);

	for (i = 0; i < featheringIterations; i++)
	{
		cv::blur(vertmaskImage, vertmaskImage, cv::Size(featheringFilter, featheringFilter), cv::Point(-1, -1), 1);
	}
	cv::resize(vertmaskImage, featheredfoundationmaskImage, cv::Size(cols, rows), 0, 0, 1);

	cv::fillPoly(featheredfoundationmaskImage, removedcontours1, cv::Scalar(0), 8, 0, cv::Point(0, 0));

	if (SkinAreaHeight > 1500)
	{
		featheringFilter = (int)(SkinAreaHeight * 0.012);
		featheringIterations = (int)(SkinAreaHeight * 0.02);
	}
	else
	{
		featheringFilter = (int)(SkinAreaHeight * 0.01);
		featheringIterations = (int)(SkinAreaHeight * 0.02);
	}

	if (featheringFilter < 3)
	{
		featheringFilter = 3;
	}
	if (featheringFilter % 2 == 0)
	{
		featheringFilter++;
	}
	if (featheringIterations == 0)
	{
		featheringIterations = 2;
	}
	cv::resize(featheredfoundationmaskImage, vertmaskImage, cv::Size(cols / 2, rows / 2), 0, 0, 1);

	for (i = 0; i < featheringIterations; i++)
	{
		cv::blur(vertmaskImage, vertmaskImage, cv::Size(featheringFilter, featheringFilter), cv::Point(-1, -1), 1);
	}
	cv::resize(vertmaskImage, featheredfoundationmaskImage, cv::Size(cols, rows), 0, 0, 1);

	//vertmaskImage.release();

	cv::fillPoly(foundationmaskImage, removedcontours, cv::Scalar(0), 8, 0, cv::Point(0, 0));

	int erosionFilter = 11;

	if (SkinAreaHeight < 700)
	{
		erosionFilter = 3;
	}
	int Anchor = erosionFilter / 2;

	cv::Mat element = cv::getStructuringElement(0, cv::Size(erosionFilter, erosionFilter), cv::Point(Anchor, Anchor));

	cv::erode(foundationmaskImage, foundationmaskImage, element, cv::Point(Anchor, Anchor), 3, 1, cv::Scalar(0));

	//element.release();

#pragma endregion
	//LOGD("Redness: Foundation mask creation completed");

	if (IsLocal)
	{
		//LOGD("Redness: local redness started");
		rednessLocalUntreated(inputImage, negativeImage, foundationmaskImage, featheredfoundationmaskImage, offset);
		//LOGD("Redness:  local redness completed");
	}
	else
	{
		//LOGD("Redness: overall redness started");
		rednessFaceUntreated(inputImage, foundationmaskImage, featheredfoundationmaskImage, offset);
		//LOGD("Redness: overall redness completed");
	}
}

void Skincare::Redness_Live::rednessLocalUntreated(cv::Mat& inputImage, const cv::Mat& negativeImage, const cv::Mat& foundationmaskImage, const cv::Mat& featheredfoundationmaskImage, double untreatedoffset)
{
	int rows = inputImage.rows;
	int cols = inputImage.cols;
	const int cn = inputImage.channels();
	int i, j;
	uchar* inputPtr = inputImage.data;
	uchar* anticipatedPtr = negativeImage.data;
	uchar* maskPtr = foundationmaskImage.data;
	uchar* featheredfoundationmaskPtr = featheredfoundationmaskImage.data;
	int step = cols * cn;
	double RR = 0, RG = 0, RB = 0;
	double NR = 0, NG = 0, NB = 0;
	double LUR = 0, LUG = 0, LUB = 0;

	double sk = 0;
	double sk2 = 0;

	double RminG = 0;
	double decVal = 0;
	double NGRratio = 0;

	double sum = 0, count = 0;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			sk2 = featheredfoundationmaskPtr[i*cols + j];
			if (sk2 > 0)
			{
				RB = inputPtr[i*step + j*cn + 2];
				RG = inputPtr[i*step + j*cn + 1];
				RR = inputPtr[i*step + j*cn + 0];
				NB = anticipatedPtr[i*step + j*cn + 2];
				NG = anticipatedPtr[i*step + j*cn + 1];
				NR = anticipatedPtr[i*step + j*cn + 0];

				RminG = RR - (0.8*RG);
				decVal = untreatedoffset*RminG;
				if ((NR + NB) > 0)
				{
					NGRratio = (NG / (NR + NB));
				}
				else {
					NGRratio = 1.1;
				}
				sk = maskPtr[i*cols + j];
				if (NG > NB && NG > NR && (NGRratio > 1.0) && sk > 0)
				{
					sum += RminG;
					count++;

					LUB = RB - NB - decVal;
					LUG = RG - NG - decVal;
					LUR = RR - NR;
					if (LUB < 0)
					{
						LUB = 0;
					}
					if (LUB > 255)
					{
						LUB = 255;
					}
					if (LUG < 0)
					{
						LUG = 0;
					}
					if (LUG > 255)
					{
						LUG = 255;
					}
					if (LUR < 0)
					{
						LUR = 0;
					}
					if (LUR > 255)
					{
						LUR = 255;
					}
					inputPtr[i*step + j*cn + 2] = (uchar)(((RB / 255) * (255 - sk2)) + ((LUB / 255) * sk2));
					inputPtr[i*step + j*cn + 1] = (uchar)(((RG / 255) * (255 - sk2)) + ((LUG / 255) * sk2));
					inputPtr[i*step + j*cn + 0] = (uchar)(((RR / 255) * (255 - sk2)) + ((LUR / 255) * sk2));
				}
			}
		}
	}

	if (count > 0)
	{
		rednessvalues[0] = (int)(sum / count);
		rednessvalues[1] = (int)(rednessvalues[0] + (untreatedoffset)*rednessvalues[0]);
	}

}

void Skincare::Redness_Live::rednessFaceUntreated(cv::Mat& inputImage, const cv::Mat& foundationmaskImage, const cv::Mat& featheredfoundationmaskImage, double untreatedoffset)
{
	int rows = inputImage.rows;
	int cols = inputImage.cols;
	int cn = inputImage.channels();
	int i, j;

	uchar* inputPtr = inputImage.data;
	uchar* maskPtr = foundationmaskImage.data;
	uchar* featheredfoundationmaskPtr = featheredfoundationmaskImage.data;
	int step = cols * cn;
	double RR = 0, RG = 0, RB = 0;
	double UR = 0, UG = 0, UB = 0;
	double sk = 0;
	double sk2 = 0;

	double RminG = 0;
	double decVal = 0;

	double sum = 0, count = 0;

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			RB = inputPtr[i*step + j*cn + 2];
			RG = inputPtr[i*step + j*cn + 1];
			RR = inputPtr[i*step + j*cn + 0];

			sk = maskPtr[i*cols + j];
			sk2 = featheredfoundationmaskPtr[i*cols + j];

			RminG = RR - (0.8*RG);
			decVal = untreatedoffset*RminG;

			if (sk > 0)
			{
				sum += RminG;
				count++;
			}

			if (sk2 > 0)
			{
				UB = RB - decVal;
				UG = RG - decVal;
				UR = RR;

				if (UB < 0)
				{
					UB = 0;
				}
				if (UB > 255)
				{
					UB = 255;
				}
				if (UG < 0)
				{
					UG = 0;
				}
				if (UG > 255)
				{
					UG = 255;
				}

				inputPtr[i*step + j*cn + 2] = (uchar)(((RB / 255) * (255 - sk2)) + ((UB / 255) * sk2));
				inputPtr[i*step + j*cn + 1] = (uchar)(((RG / 255) * (255 - sk2)) + ((UG / 255) * sk2));
				inputPtr[i*step + j*cn + 0] = (uchar)UR;
			}
		}
	}
	if (count > 0)
	{
		rednessvalues[0] = (int)(sum / count);
		rednessvalues[1] = (int)(rednessvalues[0] + (untreatedoffset)*rednessvalues[0]);
	}
}

void Skincare::Redness_Live::releaseMemory()
{
	foundationmaskImage.release();
	vertImage.release();
	negativeImage.release();
	featheredfoundationmaskImage.release();
	vertmaskImage.release();
}
