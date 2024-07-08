#include "HairDetection_Live.h"
//#include <android/log.h>

//#define LOG_TAG "HairDetection"
//#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
//#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

Skincare::HairDetection_Live::HairDetection_Live() :filterSize(7)
{
	int anchor = filterSize / 2;
	int i = 0, j = 0;
	for (i = 0; i < filterSize; i++)
	{
		for (j = 0; j < filterSize; j++)
		{
			dataArrayH[i][j] = 0;
			dataArrayV[i][j] = 0;
			dataArrayD1[i][j] = 0;
			dataArrayD2[i][j] = 0;
		}
	}
	//horizontal and vertical kernals
	for (i = 0; i < anchor; i++)
	{
		dataArrayH[i][anchor] = (float)(i + 1);
		dataArrayH[filterSize - 1 - i][anchor] = (float)(i + 1);
		dataArrayV[anchor][i] = (float)(i + 1);
		dataArrayV[anchor][filterSize - 1 - i] = (float)(i + 1);
	}
	dataArrayH[anchor][anchor] = (float)(-(anchor*(anchor + 1)));
	dataArrayV[anchor][anchor] = (float)(-(anchor*(anchor + 1)));
	//diagonal kernals
	for (i = 0; i < anchor; i++)
	{
		dataArrayD1[i][i] = (float)(i + 1);
		dataArrayD1[filterSize - 1 - i][filterSize - 1 - i] = (float)(i + 1);
		dataArrayD2[i][filterSize - 1 - i] = (float)(i + 1);
		dataArrayD2[filterSize - 1 - i][i] = (float)(i + 1);
	}
	dataArrayD1[anchor][anchor] = (float)(-(anchor*(anchor + 1)));
	dataArrayD2[anchor][anchor] = (float)(-(anchor*(anchor + 1)));
}

void Skincare::HairDetection_Live::detectHair(cv::Mat& inputImage, const int fldPoints[])
{
	int i, j, rows = inputImage.rows, cols = inputImage.cols, cn = inputImage.channels(), step = cols*cn;
	grayImage.create(rows, cols, CV_8UC1);
	uchar* inputPtr = inputImage.data;
	uchar* grayPtr = grayImage.data;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			grayPtr[i*cols + j] = (uchar)((inputPtr[i*step + j*cn + 0] + inputPtr[i*step + j*cn + 1] + inputPtr[i*step + j*cn + 2]) / 3);
		}
	}

	//defining hair Frizz ROI
	int faceWidth = fldPoints[16 * 2] - fldPoints[0 * 2] + 1;
	int faceHeight = fldPoints[8 * 2 + 1] - fldPoints[71 * 2 + 1] + 1;
	cv::Rect2i hairRect;
	hairRect.x = fldPoints[0 * 2] - (int)(faceWidth * 0.2);
	hairRect.y = 0;//fldPoints[71 * 2 + 1] - (int)(faceHeight*0.25);
	hairRect.width = (fldPoints[16 * 2] + (int)(faceWidth * 0.2)) - hairRect.x + 1;
	hairRect.height = fldPoints[33 * 2 + 1] - hairRect.y + 1;
	if (hairRect.x < 0)
	{
		hairRect.width = hairRect.width + hairRect.x;
		hairRect.x = 0;
	}
	if (hairRect.y < 0)
	{
		hairRect.height = hairRect.height + hairRect.y;
		hairRect.y = 0;
	}

	if (hairRect.width + hairRect.x > inputImage.cols)
	{
		hairRect.width = inputImage.cols - hairRect.x;
	}

	if (hairRect.height + hairRect.y > inputImage.rows)
	{
		hairRect.height = inputImage.rows - hairRect.y;
	}
	cv::Mat cropImage = grayImage(hairRect);
	std::vector<std::vector<cv::Point>> contours(1);
	cv::Point P1;
	P1.x = fldPoints[68 * 2] - hairRect.x;
	P1.y = hairRect.height;
	contours[0].push_back(P1);
	for (i = 68; i <= 74; i++)
	{
		P1.x = fldPoints[i * 2] - hairRect.x;
		P1.y = fldPoints[i * 2 + 1] - hairRect.y;
		contours[0].push_back(P1);
	}
	P1.x = fldPoints[74 * 2] - hairRect.x;
	P1.y = hairRect.height;
	contours[0].push_back(P1);
	rows = cropImage.rows, cols = cropImage.cols;
	cv::Mat maskImage = cv::Mat::ones(rows, cols, CV_8UC1);
	drawContours(maskImage, contours, 0, cv::Scalar(0), cv::FILLED, 8);

	//extracting hair edges
	cv::Mat kernal = cv::Mat(filterSize, filterSize, CV_32F, dataArrayH);
	cv::Mat edgeMapImage1, edgeMapImage2;
	filter2D(cropImage, edgeMapImage1, -1, kernal);

	kernal.data = (uchar*)dataArrayV;
	filter2D(cropImage, edgeMapImage2, -1, kernal);
	uchar* edgeMap1Ptr = edgeMapImage1.data;
	uchar* edgeMap2Ptr = edgeMapImage2.data;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			edgeMap1Ptr[i*cols + j] = std::max(edgeMap1Ptr[i*cols + j], edgeMap2Ptr[i*cols + j]);
		}
	}

	kernal.data = (uchar*)dataArrayD1;
	filter2D(cropImage, edgeMapImage2, -1, kernal);
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			edgeMap1Ptr[i*cols + j] = std::max(edgeMap1Ptr[i*cols + j], edgeMap2Ptr[i*cols + j]);
		}
	}

	kernal.data = (uchar*)dataArrayD2;
	filter2D(cropImage, edgeMapImage2, -1, kernal);
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			edgeMap1Ptr[i*cols + j] = std::max(edgeMap1Ptr[i*cols + j], edgeMap2Ptr[i*cols + j]);
		}
	}

	/*measuring frizz*/
	//stabilizing the fld points for measuring frizz
	const unsigned int pointsLength = 75, pointsLongLength = 2 * pointsLength;
	int pointsRequired = 11;
	std::vector<int> points(pointsLongLength);
	for (i = 0; i < pointsLongLength; i++)
	{
		points[i] = fldPoints[i];
	}
	landmarkSamplesForMeasurements.push_back(points);
	double meanError = 0, maxMeanError = 0;
	double limit = (double)std::sqrt(std::pow(faceWidth, 2) + std::pow(faceHeight, 2));
	const int noofMeasurementSamples = 3 * Utilities_Live::fps;
	while (landmarkSamplesForMeasurements.size() > noofMeasurementSamples / 2)
	{
		landmarkSamplesForMeasurements.pop_front();
	}
	int sampleCount = (int)landmarkSamplesForMeasurements.size();
	measurements[3] = 1;
	if (sampleCount > noofMeasurementSamples / 3)//one second here and remaining two seconds at gui.
	{
		faceWidth = points[16 * 2] - points[0 * 2] + 1;
		faceHeight = points[8 * 2 + 1] - points[71 * 2 + 1] + 1;
		maxMeanError = 0;
		limit = (double)std::sqrt(std::pow(faceWidth, 2) + std::pow(faceHeight, 2));
		int current = sampleCount - 1;
		for (i = 0; i < current; i++)
		{
			meanError = 0;
			for (j = 0; j < pointsLength; j++)
			{
				if (j == 0 || j == 16 || j == 8 || j == 33 || (j >= 68 && j <= 74))//required points only
				{
					meanError += (double)std::sqrt(std::pow((landmarkSamplesForMeasurements[current][j * 2] - landmarkSamplesForMeasurements[i][j * 2]), 2) + std::pow((landmarkSamplesForMeasurements[current][j * 2 + 1] - landmarkSamplesForMeasurements[i][j * 2 + 1]), 2));
				}
			}
			if (meanError > maxMeanError)
			{
				maxMeanError = meanError;
			}
		}
		maxMeanError /= pointsRequired;
		//LOGD("meanError and limit: %d,%d",(int)maxMeanError,(int)(limit*0.025));
		if (maxMeanError < limit*0.025)
		{
			measurements[3] = 0;
		}
	}
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	cv::Mat histMat;
	cv::calcHist(&edgeMapImage1, 1, 0, maskImage, histMat, 1, &histSize, &histRange);
	int histLimit = (int)(rows*cols*0.02);
	float* ps = (float*)histMat.data;
	int max = 0;
	int Value = 0;
	for (i = 255; i >= 0; i--)
	{
		Value += (int)ps[i];

		if (Value > histLimit)
		{
			max = i;
			break;
		}
	}
	uchar thersh = (uchar)(max / 3);
	uchar* maskPtr = maskImage.data;
	uchar intensity;
	int frizzPixels = 0, hairMaskPixels = 0;
	double averageSeverity = 0, areaEffected = 0;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			if (maskPtr[i*cols + j] > 0)
			{
				intensity = edgeMap1Ptr[i*cols + j];
				if (intensity > thersh)
				{
					averageSeverity += intensity;
					frizzPixels++;
				}
				hairMaskPixels++;
			}
		}
	}
	frizzPixels *= 2;//To get some scalable measurements
	if (frizzPixels > 0)
	{
		averageSeverity /= frizzPixels;
		areaEffected = (double)frizzPixels / hairMaskPixels;
	}
	std::vector<int> sample(2);
	sample[0] = (int)averageSeverity;//BG will effect these measurements
	sample[1] = (int)(areaEffected * 100);
	measurementSamples.push_back(sample);
	while (measurementSamples.size() > noofMeasurementSamples)
	{
		measurementSamples.pop_front();
	}
	sampleCount = (int)measurementSamples.size();
	for (j = 0; j < 2; j++)//assigning 1st sample to reset.
	{
		measurements[j] = measurementSamples[0][j];
	}
	for (i = sampleCount - 1; i > 0; i--)
	{
		for (j = 0; j < 2; j++)
		{
			measurements[j] += measurementSamples[i][j];
		}
	}
	for (j = 0; j < 2; j++)
	{
		measurements[j] /= sampleCount;
	}
	measurements[2] = (int)(measurements[0] * (measurements[1] / 100.0) * 4);//(int)(averageSeverity * areaEffected * 4);
	grayImage *= 0.5;
	edgeMapImage1.copyTo(cropImage, maskImage);
	rows = inputImage.rows; cols = inputImage.cols;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			intensity = grayPtr[i*cols + j];
			inputPtr[i*step + j*cn + 0] = intensity;
			inputPtr[i*step + j*cn + 1] = intensity;
			inputPtr[i*step + j*cn + 2] = intensity;
		}
	}
}

void Skincare::HairDetection_Live::releaseMemory()
{
	grayImage.release();
}
