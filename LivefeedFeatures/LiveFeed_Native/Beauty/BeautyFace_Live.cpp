#include "BeautyFace_Live.h"

void Beauty::BeautyFace_Live::performBeautyFace(cv::Mat& inputImage, int fldPoints[], double treatPert)
{
	const int rows = inputImage.rows, cols = inputImage.cols;
	int i;

	//LOGD("BeautyFace::Foundation mask creation started");	
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
	P1.x = fldPoints[0];
	P1.y = fldPoints[1];
	faceContour[0].push_back(P1);
	foundationmaskImage = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::drawContours(foundationmaskImage, faceContour, 0, cv::Scalar(255), cv::FILLED, 8);
	//LOGD("BeautyFace::Foundation mask creation ended");	

	//LOGD("BeautyFace::Remove contours defining started");	
	std::vector<std::vector<cv::Point>> removedcontours;
	removedcontours.resize(4);
	//Eye-Left
	for (i = 17; i <= 21; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[0].push_back(P1);
	}
	for (i = 39; i <= 41; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[0].push_back(P1);
	}
	P1.x = fldPoints[36 * 2];
	P1.y = fldPoints[36 * 2 + 1];
	removedcontours[0].push_back(P1);
	P1.x = fldPoints[17 * 2];
	P1.y = fldPoints[17 * 2 + 1];
	removedcontours[0].push_back(P1);
	//Eye-Right
	for (i = 22; i <= 26; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[1].push_back(P1);
	}
	for (i = 45; i <= 47; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[1].push_back(P1);
	}
	P1.x = fldPoints[42 * 2];
	P1.y = fldPoints[42 * 2 + 1];
	removedcontours[1].push_back(P1);

	P1.x = fldPoints[22 * 2];
	P1.y = fldPoints[22 * 2 + 1];
	removedcontours[1].push_back(P1);
	//Lip
	for (i = 48; i <= 59; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[2].push_back(P1);
	}

	for (i = 30; i <= 35; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		removedcontours[3].push_back(P1);
	}
	P1.x = fldPoints[30 * 2];
	P1.y = fldPoints[30 * 2 + 1];
	removedcontours[3].push_back(P1);
	//LOGD("BeautyFace::Remove contours defining ended");	

	//LOGD("BeautyFace::Skin rectangle for neg image thresholding started");		
	cv::Point rectPoints[4];
	double horizslope = 0, vertslope = 0;
	cv::Point nosePoint(0, 0);
	cv::Point chinPoint(0, 0);
	cv::Point startPoint(0, 0);
	cv::Point endPoint(0, 0);

	int startX = 0, endX = cols - 1;
	int startY = 0, endY = rows - 1;

	nosePoint.x = fldPoints[27 * 2];
	nosePoint.y = fldPoints[27 * 2 + 1];

	chinPoint.x = fldPoints[8 * 2];
	chinPoint.y = fldPoints[8 * 2 + 1];

	if ((chinPoint.x - nosePoint.x) != 0)
	{
		vertslope = (chinPoint.y - nosePoint.y) / (double)(chinPoint.x - nosePoint.x);
	}

	if ((fldPoints[42 * 2] - fldPoints[39 * 2]) != 0)
	{
		horizslope = (fldPoints[42 * 2 + 1] - fldPoints[39 * 2 + 1]) / (double)(fldPoints[42 * 2] - fldPoints[39 * 2]);
	}

	if (vertslope == 0)
	{
		rectPoints[0].x = fldPoints[36 * 2];
		rectPoints[0].y = fldPoints[19 * 2 + 1];

		rectPoints[1].x = fldPoints[45 * 2];
		rectPoints[1].y = fldPoints[19 * 2 + 1];

		rectPoints[2].x = fldPoints[45 * 2];
		rectPoints[2].y = fldPoints[57 * 2 + 1];

		rectPoints[3].x = fldPoints[36 * 2];
		rectPoints[3].y = fldPoints[57 * 2 + 1];
	}
	else
	{
		cv::Point tempPoint0(0, 0);
		cv::Point tempPoint1(0, 0), tempPoint2(0, 0);
		cv::Point tempPoint3(0, 0), tempPoint4(0, 0);

		tempPoint1.x = fldPoints[36 * 2];
		tempPoint1.y = fldPoints[36 * 2 + 1];

		rectPoints[0].y = fldPoints[19 * 2 + 1];
		rectPoints[0].x = (int)((rectPoints[0].y - tempPoint1.y) / vertslope) + tempPoint1.x;//Point 1		

		double c1 = 0, c2 = 0;
		tempPoint0.x = fldPoints[45 * 2];
		tempPoint0.y = fldPoints[45 * 2 + 1];

		tempPoint2.y = fldPoints[19 * 2 + 1];
		tempPoint2.x = (int)((tempPoint2.y - tempPoint0.y) / vertslope) + tempPoint0.x;//second vertical line

		c1 = tempPoint2.y - (vertslope*tempPoint2.x);

		tempPoint3.x = fldPoints[45 * 2];
		tempPoint3.y = rectPoints[0].y + (int)(horizslope*(tempPoint3.x - rectPoints[0].x));	//first horizontal line

		c2 = tempPoint3.y - (horizslope*tempPoint3.x);

		rectPoints[1].x = (int)((c2 - c1) / (vertslope - horizslope));
		rectPoints[1].y = (int)(horizslope * rectPoints[1].x + c2);

		tempPoint4.x = fldPoints[57 * 2];
		tempPoint4.y = fldPoints[57 * 2 + 1];//second horizontal line
		c2 = tempPoint4.y - (horizslope*tempPoint4.x);

		rectPoints[2].x = (int)((c2 - c1) / (vertslope - horizslope));
		rectPoints[2].y = (int)(horizslope * rectPoints[2].x + c2);

		c1 = rectPoints[0].y - (vertslope*rectPoints[0].x);//first vertical line

		rectPoints[3].x = (int)((c2 - c1) / (vertslope - horizslope));
		rectPoints[3].y = (int)(horizslope * rectPoints[3].x + c2);
	}
	for (i = 0; i < 4; i++)
	{
		if (rectPoints[i].x < 0)
		{
			rectPoints[i].x = 0;
		}
		if (rectPoints[i].x > endX)
		{
			rectPoints[i].x = endX;
		}
		if (rectPoints[i].y < 0)
		{
			rectPoints[i].y = 0;
		}
		if (rectPoints[i].y > endY)
		{
			rectPoints[i].y = endY;
		}
	}

	std::vector<std::vector<cv::Point>> rectContour;
	rectContour.resize(1);
	skinrectImage = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (i = 0; i < 4; i++)
	{
		rectContour[0].push_back(rectPoints[i]);
	}
	drawContours(skinrectImage, rectContour, 0, cv::Scalar(255), cv::FILLED, 8);
	//LOGD("BeautyFace::Skin rectangle for neg image thresholding ended");	

	//LOGD("BeautyFace::Smoothing for Negimage started");	
	int SkinAreaHeight = fldPoints[8 * 2 + 1] - fldPoints[71 * 2 + 1];

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
	if (resizefactor < 4)
	{
		resizefactor = 4;
	}
	int Smoothiterations = (int)(SkinAreaHeight * 0.0028);
	if (Smoothiterations < 2)
	{
		Smoothiterations = 2;
	}
	cv::resize(inputImage, tempImage, cv::Size(cols / resizefactor, rows / resizefactor), 0, 0, 1);
	for (i = 0; i < Smoothiterations; i++)
	{
		cv::blur(tempImage, tempImage, cv::Size(filterSize, filterSize), cv::Point(-1, -1), 1);
	}
	cv::resize(tempImage, negativeImage, cv::Size(cols, rows), 0, 0, 1);
	//LOGD("BeautyFace::Smoothing for Negimage ended");	

	positiveImage = negativeImage.clone();

	//LOGD("BeautyFace::Neg Image subtraction started");	
	cv::subtract(negativeImage, inputImage, negativeImage);
	//LOGD("BeautyFace::Neg Image subtraction ended");	


	//LOGD("BeautyFace::Pos Image subtraction started");	
	cv::subtract(inputImage, positiveImage, positiveImage);
	//LOGD("BeautyFace::Pos Image subtraction ended");	

	//LOGD("BeautyFace::1st level feathering started");	
	featheringFilter = 0; featheringIterations = 0;
	featheringFilter = (int)(SkinAreaHeight * 0.012);
	featheringIterations = (int)(SkinAreaHeight * 0.011);
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
	int erosionFilter1 = (int)(SkinAreaHeight * 0.01111);
	if (erosionFilter1 < 3)
	{
		erosionFilter1 = 3;
	}
	if (erosionFilter1 % 2 == 0)
	{
		erosionFilter1++;
	}
	if (erosionFilter1 > 11)
	{
		erosionFilter1 = 11;
	}
	int erosionIterations = 2;
	cv::Mat element2 = cv::getStructuringElement(0, cv::Size(erosionFilter1, erosionFilter1), cv::Point(-1, -1));
	cv::resize(foundationmaskImage, tempImage, cv::Size(cols / 4, rows / 4), 0, 0, 1);
	cv::erode(tempImage, tempImage, element2, cv::Point(-1, -1), 1, erosionIterations, cv::Scalar(0));
	for (i = 0; i < featheringIterations; i++)
	{
		cv::blur(tempImage, tempImage, cv::Size(featheringFilter, featheringFilter), cv::Point(-1, -1), 1);
	}
	cv::resize(tempImage, featheredfoundationmaskImage, cv::Size(cols, rows), 0, 0, 1);
	//LOGD("BeautyFace::1st level feathering ended");		

	cv::fillPoly(featheredfoundationmaskImage, removedcontours, cv::Scalar(0), 8, 0, cv::Point(0, 0));
	cv::fillPoly(foundationmaskImage, removedcontours, cv::Scalar(0), 8, 0, cv::Point(0, 0));
	beautyFaceSimulation(inputImage, treatPert);
}


void Beauty::BeautyFace_Live::beautyFaceSimulation(cv::Mat& inputImage, double treatPercent)
{
	int i = 0, j = 0;
	int rows = inputImage.rows, cols = inputImage.cols;

	//LOGD("BeautyFace::Shine mask creation started");	
	cv::split(inputImage, channels);
	cv::addWeighted(channels[0], 0.333, channels[1], 0.333, 0, cvgrayImage);
	cv::scaleAdd(channels[2], 0.333, cvgrayImage, cvgrayImage);

	cvgrayImage.copyTo(faceImage, foundationmaskImage);
	cv::Scalar avgPixelIntensity = cv::mean(cvgrayImage, foundationmaskImage);
	double meanAvg = avgPixelIntensity[0];
	cv::threshold(faceImage, shinemaskImage, (meanAvg*0.9), 255, cv::THRESH_BINARY);

	int featheringFilter1 = 7, featheringIterations1 = 4;
	cv::resize(shinemaskImage, tempImage, cv::Size(cols / 8, rows / 8), 0, 0, 1);
	for (i = 0; i < featheringIterations1; i++)
	{
		cv::blur(tempImage, tempImage, cv::Size(featheringFilter1, featheringFilter1), cv::Point(-1, -1), 1);
	}
	cv::resize(tempImage, shinemaskImage, cv::Size(cols, rows), 0, 0, 1);
	//LOGD("BeautyFace::Shine mask creation ended");		

	//LOGD("BeautyFace::2nd level feathering started");	
	cvgrayImage = cv::Mat::zeros(rows, cols, CV_8UC3);
	cv::split(negativeImage, channels);
	cv::addWeighted(channels[0], 0.333, channels[1], 0.333, 0, cvgrayImage);
	cv::scaleAdd(channels[2], 0.333, cvgrayImage, cvgrayImage);

	faceImage = cv::Mat::zeros(rows, cols, CV_8UC1);
	cvgrayImage.copyTo(faceImage, skinrectImage);
	cv::threshold(faceImage, newNegMaskImage, 30, 255, cv::THRESH_BINARY);
	cv::bitwise_not(newNegMaskImage, newNegMaskImage);
	cv::bitwise_and(featheredfoundationmaskImage, newNegMaskImage, featheredfoundationmaskImage);

	cv::resize(featheredfoundationmaskImage, tempImage, cv::Size(cols / 4, rows / 4), 0, 0, 1);
	for (i = 0; i < featheringIterations; i++)
	{
		cv::blur(tempImage, tempImage, cv::Size(featheringFilter, featheringFilter), cv::Point(-1, -1), 1);
	}
	cv::resize(tempImage, featheredfoundationmaskImage, cv::Size(cols, rows), 0, 0, 1);
	//LOGD("BeautyFace::2nd level feathering ended");


	uchar* inputPtr = inputImage.data;
	uchar* ffoundationmaskPtr = featheredfoundationmaskImage.data;
	uchar* negPtr = negativeImage.data;
	uchar* posPtr = positiveImage.data;
	uchar* fshinemaskPtr = shinemaskImage.data;
	int cn = inputImage.channels();
	int step = cols * cn;

	double sk = 0;
	double sk1 = 0, sk2 = 0;
	double RR = 0, RG = 0, RB = 0;
	double TR = 0, TG = 0, TB = 0;
	double NR = 0, NG = 0, NB = 0;
	double PR = 0, PG = 0, PB = 0;
	double FR = 0, FG = 0, FB = 0;

	//LOGD("BeautyFace::Simulation started");	
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			sk = ffoundationmaskPtr[i*cols + j];
			if (sk > 0)
			{
				RB = inputPtr[i*step + j*cn + 2];
				RG = inputPtr[i*step + j*cn + 1];
				RR = inputPtr[i*step + j*cn + 0];
				NB = negPtr[i*step + j*cn + 2];
				NG = negPtr[i*step + j*cn + 1];
				NR = negPtr[i*step + j*cn + 0];

				//Treating
				TR = RR + (treatPercent *NR);
				TG = RG + (treatPercent *NG);
				TB = RB + (treatPercent *NB);

				sk1 = fshinemaskPtr[i*cols + j];
				if (sk1 > 0)
				{
					//Mattifying
					PB = posPtr[i*step + j*cn + 2];
					PG = posPtr[i*step + j*cn + 1];
					PR = posPtr[i*step + j*cn + 0];

					TB = TB - (treatPercent *PB);
					TG = TG - (treatPercent *PG);
					TR = TR - (treatPercent *PR);
				}

				if (TR > 255)
				{
					TR = 255;
				}
				if (TR < 0)
				{
					TR = 0;
				}
				if (TG > 255)
				{
					TG = 255;
				}

				if (TG < 0)
				{
					TG = 0;
				}
				if (TB > 255)
				{
					TB = 255;
				}
				if (TB < 0)
				{
					TB = 0;
				}

				FR = RR - ((sk / 255)*(RR - TR));
				FG = RG - ((sk / 255)*(RG - TG));
				FB = RB - ((sk / 255)*(RB - TB));

				inputPtr[i*cols*cn + j*cn + 0] = (uchar)FR;
				inputPtr[i*cols*cn + j*cn + 1] = (uchar)FG;
				inputPtr[i*cols*cn + j*cn + 2] = (uchar)FB;
			}
		}
	}
	//LOGD("BeautyFace::Simulation ended");	
}

void Beauty::BeautyFace_Live::releaseMemory()
{
	negativeImage.release();
	positiveImage.release();
	skinrectImage.release();
	foundationmaskImage.release();
	featheredfoundationmaskImage.release();
	tempImage.release();
	cvgrayImage.release();
	faceImage.release();
	shinemaskImage.release();
	newNegMaskImage.release();

	channels[0].release();
	channels[1].release();
	channels[2].release();
}
