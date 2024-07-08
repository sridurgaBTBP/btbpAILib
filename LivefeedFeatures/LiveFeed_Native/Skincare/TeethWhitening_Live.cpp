#include "TeethWhitening_Live.h"

void Skincare::TeethWhitening_Live::doTeethWhitening(cv::Mat& inputImage, int fldPoints[], bool isWithoutROI, double effectValue)
{
	int i = 0, inputImageRows = inputImage.rows, inputImageCols = inputImage.cols;
	measurements[0] = 0;
	measurements[1] = 0;

	//check for mouth is opened or not
	int innerLayerWidth = fldPoints[66 * 2 + 1] - fldPoints[62 * 2 + 1];
	int outerLayerWidth = fldPoints[57 * 2 + 1] - fldPoints[51 * 2 + 1];
	if (innerLayerWidth < 0.2*outerLayerWidth)
	{
		measurements[2] = 1;//setting failure status
		return;
		//mouth is not opened.
	}

	mouthMaskCustom = cv::Mat::zeros(inputImageRows, inputImageCols, CV_8UC1);
	mouthMaskFLD = cv::Mat::zeros(inputImageRows, inputImageCols, CV_8UC1);
	cv::Rect mouthRect = mouthMasksCreation(fldPoints, mouthMaskCustom, mouthMaskFLD);
	if (0 <= mouthRect.x && 0 <= mouthRect.width && mouthRect.x + mouthRect.width <= inputImageCols
		&& 0 <= mouthRect.y && 0 <= mouthRect.height && mouthRect.y + mouthRect.height <= inputImageRows)
	{
		//rectangle within the bounds.
	}
	else
	{
		measurements[2] = 1;
		return;
	}
	cv::Mat mouthCropMaskCustomed = mouthMaskCustom(mouthRect);
	cv::Mat mouthCropMaskFLD = mouthMaskFLD(mouthRect).clone();
	cv::Mat teethImage = inputImage(mouthRect);

	cv::Mat channelsRGB[3];
	cv::split(teethImage, channelsRGB);
	cv::Mat teethMask = cv::Mat::zeros(mouthCropMaskCustomed.rows, mouthCropMaskCustomed.cols, CV_8UC1);
	channelsRGB[1].copyTo(teethMask, mouthCropMaskCustomed);

	double teethThresh = getThreshVal_Otsu_8u_WithMask(teethMask);
	cv::threshold(teethMask, teethMask, teethThresh, 255, cv::THRESH_BINARY);
	cv::Mat teethMaskContours = teethMask.clone();
	std::vector<std::vector<cv::Point>> regions;
    cv::findContours(teethMaskContours, regions, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	int maxObjectIndex = -1;
	double regionArea, maxRegionArea = 0;
	for (i = 0; i < regions.size(); i++)
	{
		regionArea = cv::contourArea(regions[i]);
		if (regionArea > maxRegionArea)
		{
			maxRegionArea = regionArea;
			maxObjectIndex = i;
		}
	}
	cv::Rect2i maxObjectRectangle = cv::boundingRect(regions[maxObjectIndex]);
	int j;
	const int imgRows = teethMask.rows, imgCols = teethMask.cols;
	int step = (int)teethMask.step;
	uchar* maskPtr = teethMask.data;
	const uchar* mouthPtr = mouthCropMaskFLD.data;
	int midPos = std::max(fldPoints[60 * 2 + 1], fldPoints[64 * 2 + 1]) - std::min(fldPoints[50 * 2 + 1], fldPoints[52 * 2 + 1]);

	if (maxObjectRectangle.x <= 2 || (maxObjectRectangle.x + maxObjectRectangle.width) >= teethMask.cols - 2)
	{
		if (maxObjectRectangle.x <= 2)
		{
			for (i = midPos; i < imgRows; i++)
			{
				for (j = 0; j < imgCols; j++)
				{
					if (mouthPtr[i*step + j] == 0)
						maskPtr[i*step + j] = 0;
					else
						break;
				}
			}
		}
		if ((maxObjectRectangle.x + maxObjectRectangle.width) >= teethMask.cols - 2)
		{
			for (i = midPos; i < imgRows; i++)
			{
				for (j = imgCols - 1; j >= 0; j--)
				{
					if (mouthPtr[i*step + j] == 0)
						maskPtr[i*step + j] = 0;
					else
						break;
				}
			}
		}
	}

	midPos = std::min(fldPoints[61 * 2 + 1], fldPoints[63 * 2 + 1]) - mouthRect.y;
	for (i = 0; i < midPos; i++)
	{
		for (j = 0; j < imgCols; j++)
		{
			maskPtr[i*step + j] = 0;
		}
	}

	cv::Mat teethMaskCopy = teethMask.clone();
	for (i = 0; i < regions.size(); i++)//removing all the objects except max object
	{
		if (i != maxObjectIndex)
			cv::drawContours(teethMask, regions, i, cv::Scalar(0), cv::FILLED);
	}
	teethMaskCopy.copyTo(teethMask, mouthCropMaskFLD);//copying all the objects which are inside the inner lip mask

	for (i = 0; i < imgRows; i++)//removing image extreme pixels left due to drawcontours
	{
		maskPtr[i*step + 0] = 0;
		maskPtr[i*step + (imgCols - 1)] = 0;
	}
	for (i = 0; i < imgCols; i++)
	{
		maskPtr[0 + i] = 0;
		maskPtr[(imgRows - 1)*step + i] = 0;
	}

	cv::Scalar averageRGB = cv::mean(teethImage, teethMask);
	if (averageRGB[0] < 100)
	{
		//Teeth are not present in teethMask.
		measurements[2] = 1;
		return;
	}
	uchar arr[] = { (uchar)averageRGB[0], (uchar)averageRGB[1], (uchar)averageRGB[2] };
	cv::Mat rgbImg = cv::Mat(1, 1, CV_8UC3, arr);
	cv::Mat labImg;
	cv::cvtColor(rgbImg, labImg, cv::COLOR_RGB2Lab);
	cv::Scalar averageLAB = cv::mean(labImg);
	measurements[0] = (int)(averageLAB[0] * (100.0 / 255.0));

	double brightnessRatio = 1 + (effectValue / 10);
	double offsets[3];
	findingOffsets(averageRGB, offsets, effectValue);
	int kernalSize = (int)(mouthRect.width * 0.05);
	if (kernalSize % 2 == 0)
	{
		kernalSize--;
	}
	if (kernalSize < 3)
	{
		kernalSize = 3;
	}

	for (i = 0; i < 2; i++)
	{
		cv::blur(teethMask, teethMask, cv::Size(kernalSize, kernalSize));
	}

	cv::Mat teethWhiteningImage = teethImage.clone();
	uchar* teethWhiteningPtr = teethWhiteningImage.data;
	int totalPixels = imgRows*imgCols;
	int cn = teethImage.channels();
	double OR, OG, OB, WR, WG, WB, sk;

	for (i = 0; i < totalPixels; i++)
	{
		sk = maskPtr[i];
		if (sk > 0)
		{
			OB = teethWhiteningPtr[i * cn + 2];
			OG = teethWhiteningPtr[i * cn + 1];
			OR = teethWhiteningPtr[i * cn + 0];

			WB = OB + offsets[2];
			WG = OG + offsets[1];
			WR = OR + offsets[0];

			WB *= brightnessRatio;
			WG *= brightnessRatio;
			WR *= brightnessRatio;

			if (WB > 255)
				WB = 255;

			if (WG > 255)
				WG = 255;

			if (WR > 255)
				WR = 255;
			if (sk < 255)
			{
				WB = ((OB / 255) * (255 - sk)) + ((WB / 255) * sk);
				WG = ((OG / 255) * (255 - sk)) + ((WG / 255) * sk);
				WR = ((OR / 255) * (255 - sk)) + ((WR / 255) * sk);
			}

			teethWhiteningPtr[i * cn + 2] = (uchar)WB;
			teethWhiteningPtr[i * cn + 1] = (uchar)WG;
			teethWhiteningPtr[i * cn + 0] = (uchar)WR;
		}
	}

	averageRGB[2] += offsets[2];
	averageRGB[1] += offsets[1];
	averageRGB[0] += offsets[0];

	averageRGB[2] *= brightnessRatio;
	averageRGB[1] *= brightnessRatio;
	averageRGB[0] *= brightnessRatio;
	if (averageRGB[2] > 255)
		averageRGB[2] = 255;
	if (averageRGB[1] > 255)
		averageRGB[1] = 255;
	if (averageRGB[0] > 255)
		averageRGB[0] = 255;
	arr[0] = (uchar)averageRGB[0]; arr[1] = (uchar)averageRGB[1]; arr[2] = (uchar)averageRGB[2];
	rgbImg.data = arr;
	cv::cvtColor(rgbImg, labImg, cv::COLOR_RGB2Lab);
	averageLAB = cv::mean(labImg);
	measurements[1] = (int)(averageLAB[0] * (100.0 / 255.0));

	if (!isWithoutROI)
	{
		std::vector<std::vector<cv::Point>> layers;
		layers.resize(1);
		cv::Point P1;
		for (i = 60; i <= 67; i++)
		{
			P1.x = fldPoints[i * 2] - mouthRect.x;
			P1.y = fldPoints[i * 2 + 1] - mouthRect.y;
			layers[0].push_back(P1);
		}
		P1.x = fldPoints[60 * 2] - mouthRect.x;
		P1.y = fldPoints[60 * 2 + 1] - mouthRect.y;
		layers[0].push_back(P1);
		cv::drawContours(teethWhiteningImage, layers, 0, cv::Scalar(0, 255, 0), 1);
	}
	teethWhiteningImage.copyTo(teethImage);
	measurements[2] = 0;
}


double Skincare::TeethWhitening_Live::getThreshVal_Otsu_8u_WithMask(const cv::Mat& src)
{
	cv::Size size = src.size();
	int step = (int)src.step;
	if (src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
		step = size.width;
	}
	const int N = 256;
	int M = 0;
	int i, j, h[N] = { 0 };
	for (i = 0; i < size.height; i++)
	{
		const uchar* psrc = src.ptr() + step*i;
		j = 0;
		for (; j < size.width; j++)
		{
			if (psrc[j] > 0)
			{
				h[psrc[j]]++;
				++M;
			}
		}
	}

	double mu = 0, scale = 1. / M;
	for (i = 0; i < N; i++)
		mu += i*(double)h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i*p_i) / q1;
		mu2 = (mu - q1*mu1) / q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}

cv::Rect Skincare::TeethWhitening_Live::mouthMasksCreation(const int fldPoints[], cv::Mat& mouthMaskCustomed, cv::Mat& mouthMaskFLD)
{
	std::vector<std::vector<cv::Point>> layers(2);
	//Teeeth mask creation
	int i = 0;
	cv::Point P1;
	for (i = 49; i <= 53; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		layers[0].push_back(P1);
	}
	P1.x = fldPoints[64 * 2];
	P1.y = fldPoints[64 * 2 + 1];
	layers[0].push_back(P1);
	P1.x = fldPoints[64 * 2];
	P1.y = fldPoints[66 * 2 + 1];
	layers[0].push_back(P1);
	P1.x = fldPoints[66 * 2];
	P1.y = fldPoints[66 * 2 + 1];
	layers[0].push_back(P1);
	P1.x = fldPoints[60 * 2];
	P1.y = fldPoints[66 * 2 + 1];
	layers[0].push_back(P1);
	P1.x = fldPoints[60 * 2];
	P1.y = fldPoints[60 * 2 + 1];
	layers[0].push_back(P1);
	for (i = 60; i <= 67; i++)
	{
		P1.x = fldPoints[i * 2];
		P1.y = fldPoints[i * 2 + 1];
		layers[1].push_back(P1);
	}
	cv::drawContours(mouthMaskCustomed, layers, 0, cv::Scalar(255), cv::FILLED);
	cv::Rect boundingRectangle = cv::boundingRect(layers[0]);
	cv::drawContours(mouthMaskFLD, layers, 1, cv::Scalar(255), cv::FILLED);
	return boundingRectangle;
}

inline void Skincare::TeethWhitening_Live::findingOffsets(cv::Scalar averageRGB, double offsets[], double effectValue)
{
	std::string order = "";
	if (averageRGB[0] > averageRGB[1])
	{
		if (averageRGB[0] > averageRGB[2])
		{
			if (averageRGB[1] > averageRGB[2])
				order = "RGB";
			else
				order = "RBG";
		}
		else
		{
			order = "BRG";
		}
	}
	else
	{
		if (averageRGB[1] > averageRGB[2])
		{
			if (averageRGB[0] > averageRGB[2])
				order = "GRB";
			else
				order = "GBR";
		}
		else
		{
			order = "BGR";
		}
	}

	double average_RtoG = averageRGB[0] / averageRGB[1];
	double average_RtoB = averageRGB[0] / averageRGB[2];
	double average_GtoB = averageRGB[1] / averageRGB[2];
	double idealWhiteRatio1 = 1.02, idealWhiteRatio2 = 1.03;
	double ratioDeviation1 = 0, ratioDeviation2 = 0;

	if (order == "RGB")
	{
		offsets[0] = 0;
		ratioDeviation1 = average_RtoG - idealWhiteRatio1;
		if (ratioDeviation1 > 0)
			offsets[1] = averageRGB[0] / (average_RtoG - (effectValue*ratioDeviation1)) - averageRGB[1];
		else
			offsets[1] = 0;

		ratioDeviation2 = average_RtoB - idealWhiteRatio2;
		if (ratioDeviation2 > 0)
			offsets[2] = averageRGB[0] / (average_RtoB - (effectValue*ratioDeviation2)) - averageRGB[2];
		else
			offsets[2] = 0;
	}
	else if (order == "RBG")
	{
		offsets[0] = 0;
		ratioDeviation1 = average_RtoG - idealWhiteRatio2;
		if (ratioDeviation1 > 0)
			offsets[1] = averageRGB[0] / (average_RtoG - (effectValue*ratioDeviation1)) - averageRGB[1];
		else
			offsets[1] = 0;

		ratioDeviation2 = average_RtoB - idealWhiteRatio1;
		if (ratioDeviation2 > 0)
			offsets[2] = averageRGB[0] / (average_RtoB - (effectValue*ratioDeviation2)) - averageRGB[2];
		else
			offsets[2] = 0;
	}
	else if (order == "GRB")
	{
		offsets[1] = 0;
		ratioDeviation1 = (1 / average_RtoG) - idealWhiteRatio1;
		if (ratioDeviation1 > 0)
			offsets[0] = averageRGB[1] / ((1 / average_RtoG) - (effectValue*ratioDeviation1)) - averageRGB[0];
		else
			offsets[0] = 0;

		ratioDeviation2 = average_GtoB - idealWhiteRatio2;
		if (ratioDeviation2 > 0)
			offsets[2] = averageRGB[1] / (average_GtoB - (effectValue*ratioDeviation2)) - averageRGB[2];
		else
			offsets[2] = 0;
	}
	else if (order == "GBR")
	{
		offsets[1] = 0;
		ratioDeviation1 = (1 / average_RtoG) - idealWhiteRatio2;
		if (ratioDeviation1 > 0)
			offsets[0] = averageRGB[1] / ((1 / average_RtoG) - (effectValue*ratioDeviation1)) - averageRGB[0];
		else
			offsets[0] = 0;

		ratioDeviation2 = average_GtoB - idealWhiteRatio1;
		if (ratioDeviation2 > 0)
			offsets[2] = averageRGB[1] / (average_GtoB - (effectValue*ratioDeviation2)) - averageRGB[2];
		else
			offsets[2] = 0;
	}
	else if (order == "BRG")
	{
		offsets[2] = 0;
		ratioDeviation1 = (1 / average_RtoB) - idealWhiteRatio1;
		if (ratioDeviation1 > 0)
			offsets[0] = averageRGB[2] / ((1 / average_RtoB) - (effectValue*ratioDeviation1)) - averageRGB[0];
		else
			offsets[0] = 0;

		ratioDeviation2 = (1 / average_GtoB) - idealWhiteRatio2;
		if (ratioDeviation2 > 0)
			offsets[1] = averageRGB[2] / ((1 / average_GtoB) - (effectValue*ratioDeviation2)) - averageRGB[1];
		else
			offsets[1] = 0;
	}
	else if (order == "BGR")
	{
		offsets[2] = 0;
		ratioDeviation1 = (1 / average_RtoB) - idealWhiteRatio2;
		if (ratioDeviation1 > 0)
			offsets[0] = averageRGB[2] / ((1 / average_RtoB) - (effectValue*ratioDeviation1)) - averageRGB[0];
		else
			offsets[0] = 0;

		ratioDeviation2 = (1 / average_GtoB) - idealWhiteRatio1;
		if (ratioDeviation2 > 0)
			offsets[1] = averageRGB[2] / ((1 / average_GtoB) - (effectValue*ratioDeviation2)) - averageRGB[1];
		else
			offsets[1] = 0;
	}
}

void Skincare::TeethWhitening_Live::releaseMemory()
{
	mouthMaskCustom.release();
	mouthMaskFLD.release();
}
