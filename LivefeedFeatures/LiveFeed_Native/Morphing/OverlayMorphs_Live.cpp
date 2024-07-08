#include "OverlayMorphs_Live.h"
#include "Utilities_Live.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

int Morphing::OverlayMorphs_Live::eyebrowTemplatesX[5] = { 0, 45, 100,155,210 };//x coords are chosen by observing the x distance between fld points
int Morphing::OverlayMorphs_Live::eyebrowTemplatesY[6][5] = {
	/*straight*/{ 221, 207, 200, 203, 213 },
	/*curved*/{ 234, 209, 200, 212, 237 },
	/*softArch*/{ 226, 203, 200, 214, 230 },
	/*highArch*/{ 248, 203, 200, 216, 237 },
	/*S shaped */{ 240, 209, 200, 218, 229 },
	/*upward*/{ 214, 199, 200, 216, 231 }
};
cv::Point Morphing::OverlayMorphs_Live::eyebrowOffsetsRight[5], Morphing::OverlayMorphs_Live::eyebrowOffsetsLeft[5];
std::deque<std::vector<cv::Point>> Morphing::OverlayMorphs_Live::eyebrowOffsetSamplesRight, Morphing::OverlayMorphs_Live::eyebrowOffsetSamplesLeft;
int Morphing::OverlayMorphs_Live::eyebrowCurrentShapeIndex = -1;
bool Morphing::OverlayMorphs_Live::isStillImage = false;

bool Morphing::OverlayMorphs_Live::eyebrowMorph(cv::Mat& inputImage, const int fldPoints[], int shapeIndex, float morphPercent, bool drawTriangles)
{
	float thresholds[3] = { 10,0.07f,0.3f };
	if (!checkIQCForAffine(fldPoints, thresholds))
	{
		eyebrowOffsetSamplesRight.clear();
		eyebrowOffsetSamplesLeft.clear();
		return false;
	}

	int paddX = (int)((fldPoints[21 * 2] - fldPoints[17 * 2])*0.1);
	cv::Rect rectRight;
	rectRight.x = fldPoints[17 * 2] - paddX;
	rectRight.y = fldPoints[68 * 2 + 1] + (int)((std::min(fldPoints[18 * 2 + 1], std::min(fldPoints[19 * 2 + 1], fldPoints[20 * 2 + 1])) - fldPoints[68 * 2 + 1]) * 0.25);
	rectRight.width = fldPoints[21 * 2] + paddX - rectRight.x + 1;
	rectRight.height = (fldPoints[36 * 2 + 1] + fldPoints[37 * 2 + 1]) / 2 - rectRight.y + 1;
	if (checkBoundsForAffine(rectRight, inputImage.cols, inputImage.rows))
	{
		eyebrowOffsetSamplesRight.clear();
		eyebrowOffsetSamplesLeft.clear();
		return true;
	}
	paddX = (int)((fldPoints[26 * 2] - fldPoints[22 * 2])*0.1);
	cv::Rect rectLeft;
	rectLeft.x = fldPoints[22 * 2] - paddX;
	rectLeft.y = fldPoints[74 * 2 + 1] + (int)((std::min(fldPoints[23 * 2 + 1], std::min(fldPoints[24 * 2 + 1], fldPoints[25 * 2 + 1])) - fldPoints[74 * 2 + 1]) * 0.25);
	rectLeft.width = fldPoints[26 * 2] + paddX - rectLeft.x + 1;
	rectLeft.height = (fldPoints[44 * 2 + 1] + fldPoints[45 * 2 + 1]) / 2 - rectLeft.y + 1;
	if (checkBoundsForAffine(rectLeft, inputImage.cols, inputImage.rows))
	{
		eyebrowOffsetSamplesRight.clear();
		eyebrowOffsetSamplesLeft.clear();
		return true;
	}

	const int N = 5;
	std::vector<cv::Point> templatesRight(N);
	for (int i = 0; i < N; i++)
	{
		templatesRight[i].x = eyebrowTemplatesX[i];
		templatesRight[i].y = eyebrowTemplatesY[shapeIndex][i];
	}
	//rotation
	double angle = std::atan2(fldPoints[45 * 2 + 1] - fldPoints[36 * 2 + 1], fldPoints[45 * 2] - fldPoints[36 * 2])*(180 / CV_PI);
	cv::Point2f center((float)(fldPoints[36 * 2] + fldPoints[45 * 2]) / 2, (float)(fldPoints[36 * 2 + 1] + fldPoints[45 * 2 + 1]) / 2);
	cv::Mat m = cv::getRotationMatrix2D(center, -angle, 1);
	cv::transform(templatesRight, templatesRight, m);
	//scale (doing scaling after rotation(after straighting) but not at a time in transform because eyebrow extremes Y points are not accurate.So we can not calculate distance(x^2+y^2) correctly)
	double scale = (double)(fldPoints[21 * 2] - fldPoints[17 * 2] + 1) / (templatesRight[4].x - templatesRight[0].x + 1);
	for (int i = 0; i < N; i++)
	{
		templatesRight[i].x = (int)(templatesRight[i].x* scale);
		templatesRight[i].y = (int)(templatesRight[i].y *scale);
	}
	//shift
	int shiftX = fldPoints[21 * 2] - templatesRight[4].x;
	int shiftY;
	if (shapeIndex != 0)
		shiftY = fldPoints[21 * 2 + 1] - templatesRight[4].y;
	else
		shiftY = fldPoints[20 * 2 + 1] - templatesRight[3].y;
	for (int i = 0; i < N; i++)
	{
		templatesRight[i].x += shiftX;
		templatesRight[i].y += shiftY;
	}
	int paddY = 0;
	int minY = inputImage.rows - 1;
	int minYIndex = 0;
	for (int i = 0; i < N; i++)
	{
		int temp = std::min(fldPoints[(17 + i) * 2 + 1], templatesRight[i].y);
		if (temp < minY)
		{
			minY = temp;
			paddY = std::abs(fldPoints[(17 + i) * 2 + 1] - templatesRight[i].y);
			minYIndex = i;
		}
	}
	int startY = 0;
	if (minY - paddY < rectRight.y)
	{
		if (fldPoints[(17 + minYIndex) * 2 + 1] < templatesRight[minYIndex].y)
		{
			float maxpercent = (float)(fldPoints[(17 + minYIndex) * 2 + 1] - rectRight.y) / (templatesRight[minYIndex].y - fldPoints[(17 + minYIndex) * 2 + 1]);
			if (morphPercent > maxpercent)
				morphPercent = maxpercent;
		}
		else
		{
			float maxpercent = (float)(rectRight.y - fldPoints[(17 + minYIndex) * 2 + 1]) / (2 * (templatesRight[minYIndex].y - fldPoints[(17 + minYIndex) * 2 + 1]));
			if (morphPercent > maxpercent)
				morphPercent = maxpercent;
		}
	}
	else
	{
		startY = ((minY - paddY) - rectRight.y) / 2;
	}
	int eyePointY = (std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (int)((std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (fldPoints[18 * 2 + 1] + fldPoints[19 * 2 + 1] + fldPoints[20 * 2 + 1]) / 3)*0.15)) - rectRight.y;
	const int length = 24;
	cv::Point pointsSrcRight[length], pointsDstRight[length];
	pointsSrcRight[0].x = fldPoints[17 * 2] - rectRight.x; pointsSrcRight[0].y = fldPoints[17 * 2 + 1] - (int)((fldPoints[17 * 2 + 1] - fldPoints[18 * 2 + 1]) *0.25) - rectRight.y;
	pointsSrcRight[1].x = fldPoints[18 * 2] - rectRight.x; pointsSrcRight[1].y = fldPoints[18 * 2 + 1] - rectRight.y;
	pointsSrcRight[2].x = fldPoints[19 * 2] - rectRight.x; pointsSrcRight[2].y = fldPoints[19 * 2 + 1] - rectRight.y;
	pointsSrcRight[3].x = fldPoints[20 * 2] - rectRight.x; pointsSrcRight[3].y = fldPoints[20 * 2 + 1] - rectRight.y;
	pointsSrcRight[4].x = fldPoints[21 * 2] - rectRight.x; pointsSrcRight[4].y = fldPoints[21 * 2 + 1] - rectRight.y;
	pointsSrcRight[5].x = pointsSrcRight[4].x; pointsSrcRight[5].y = (eyePointY + pointsSrcRight[4].y) / 2;
	pointsSrcRight[6].x = pointsSrcRight[3].x; pointsSrcRight[6].y = (eyePointY + pointsSrcRight[3].y) / 2;
	pointsSrcRight[7].x = pointsSrcRight[2].x; pointsSrcRight[7].y = (eyePointY + pointsSrcRight[2].y) / 2;
	pointsSrcRight[8].x = pointsSrcRight[1].x; pointsSrcRight[8].y = (eyePointY + pointsSrcRight[1].y) / 2;
	pointsSrcRight[9].x = pointsSrcRight[0].x; pointsSrcRight[9].y = ((rectRight.height - 1) + pointsSrcRight[0].y) / 2;
	//support coords
	pointsSrcRight[10].x = 0;                   pointsSrcRight[10].y = pointsSrcRight[0].y;
	pointsSrcRight[11].x = pointsSrcRight[0].x; pointsSrcRight[11].y = startY;
	pointsSrcRight[12].x = pointsSrcRight[1].x; pointsSrcRight[12].y = startY;
	pointsSrcRight[13].x = pointsSrcRight[2].x; pointsSrcRight[13].y = startY;
	pointsSrcRight[14].x = pointsSrcRight[3].x; pointsSrcRight[14].y = startY;
	pointsSrcRight[15].x = pointsSrcRight[4].x; pointsSrcRight[15].y = startY;
	pointsSrcRight[16].x = rectRight.width - 1; pointsSrcRight[16].y = pointsSrcRight[4].y;
	pointsSrcRight[17].x = rectRight.width - 1; pointsSrcRight[17].y = pointsSrcRight[5].y;
	pointsSrcRight[18].x = pointsSrcRight[4].x; pointsSrcRight[18].y = eyePointY;
	pointsSrcRight[19].x = pointsSrcRight[3].x; pointsSrcRight[19].y = eyePointY;
	pointsSrcRight[20].x = pointsSrcRight[2].x; pointsSrcRight[20].y = eyePointY;
	pointsSrcRight[21].x = pointsSrcRight[1].x; pointsSrcRight[21].y = eyePointY;
	pointsSrcRight[22].x = pointsSrcRight[0].x; pointsSrcRight[22].y = rectRight.height - 1;
	pointsSrcRight[23].x = 0;                   pointsSrcRight[23].y = pointsSrcRight[9].y;


	std::vector<cv::Point> templatesLeft(N);
	for (int i = 0; i < N; i++)
	{
		//mirror template
		templatesLeft[N - 1 - i].x = eyebrowTemplatesX[4] - eyebrowTemplatesX[i];
		templatesLeft[N - 1 - i].y = eyebrowTemplatesY[shapeIndex][i];
	}
	//rotation 
	cv::transform(templatesLeft, templatesLeft, m);
	//scale (doing scaling after rotation(after straighting) but not at a time in transform because eyebrow extremes Y points are not accurate.So we can not calculate distance(x^2+y^2) correctly)
	scale = (double)(fldPoints[26 * 2] - fldPoints[22 * 2] + 1) / (templatesLeft[4].x - templatesLeft[0].x + 1);
	for (int i = 0; i < N; i++)
	{
		templatesLeft[i].x = (int)(templatesLeft[i].x* scale);
		templatesLeft[i].y = (int)(templatesLeft[i].y *scale);
	}
	//shift
	shiftX = fldPoints[22 * 2] - templatesLeft[0].x;
	if (shapeIndex != 0)
		shiftY = fldPoints[22 * 2 + 1] - templatesLeft[0].y;
	else
		shiftY = fldPoints[23 * 2 + 1] - templatesLeft[1].y;
	for (int i = 0; i < N; i++)
	{
		templatesLeft[i].x += shiftX;
		templatesLeft[i].y += shiftY;
	}
	paddY = 0;
	minY = inputImage.rows - 1;
	minYIndex = 0;
	for (int i = 0; i < 5; i++)
	{
		int temp = std::min(fldPoints[(22 + i) * 2 + 1], templatesLeft[i].y);
		if (temp < minY)
		{
			minY = temp;
			paddY = std::abs(fldPoints[(22 + i) * 2 + 1] - templatesLeft[i].y);
			minYIndex = i;
		}
	}
	startY = 0;
	if (minY - paddY < rectLeft.y)
	{
		if (fldPoints[(22 + minYIndex) * 2 + 1] < templatesLeft[minYIndex].y)
		{
			float maxpercent = (float)(fldPoints[(22 + minYIndex) * 2 + 1] - rectLeft.y) / (templatesLeft[minYIndex].y - fldPoints[(22 + minYIndex) * 2 + 1]);
			if (morphPercent > maxpercent)
				morphPercent = maxpercent;
		}
		else
		{
			float maxpercent = (float)(rectLeft.y - fldPoints[(22 + minYIndex) * 2 + 1]) / (2 * (templatesLeft[minYIndex].y - fldPoints[(22 + minYIndex) * 2 + 1]));
			if (morphPercent > maxpercent)
				morphPercent = maxpercent;
		}
	}
	else
	{
		startY = ((minY - paddY) - rectLeft.y) / 2;
	}
	eyePointY = (std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (int)((std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (fldPoints[23 * 2 + 1] + fldPoints[24 * 2 + 1] + fldPoints[25 * 2 + 1]) / 3)*0.15)) - rectLeft.y;
	cv::Point pointsSrcLeft[length], pointsDstLeft[length];
	pointsSrcLeft[0].x = fldPoints[26 * 2] - rectLeft.x; pointsSrcLeft[0].y = fldPoints[26 * 2 + 1] - (int)((fldPoints[26 * 2 + 1] - fldPoints[25 * 2 + 1]) *0.25) - rectLeft.y;
	pointsSrcLeft[1].x = fldPoints[25 * 2] - rectLeft.x; pointsSrcLeft[1].y = fldPoints[25 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[2].x = fldPoints[24 * 2] - rectLeft.x; pointsSrcLeft[2].y = fldPoints[24 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[3].x = fldPoints[23 * 2] - rectLeft.x; pointsSrcLeft[3].y = fldPoints[23 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[4].x = fldPoints[22 * 2] - rectLeft.x; pointsSrcLeft[4].y = fldPoints[22 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[5].x = pointsSrcLeft[4].x; pointsSrcLeft[5].y = (eyePointY + pointsSrcLeft[4].y) / 2;
	pointsSrcLeft[6].x = pointsSrcLeft[3].x; pointsSrcLeft[6].y = (eyePointY + pointsSrcLeft[3].y) / 2;
	pointsSrcLeft[7].x = pointsSrcLeft[2].x; pointsSrcLeft[7].y = (eyePointY + pointsSrcLeft[2].y) / 2;
	pointsSrcLeft[8].x = pointsSrcLeft[1].x; pointsSrcLeft[8].y = (eyePointY + pointsSrcLeft[1].y) / 2;
	pointsSrcLeft[9].x = pointsSrcLeft[0].x; pointsSrcLeft[9].y = ((rectLeft.height - 1) + pointsSrcLeft[0].y) / 2;
	//support coords
	pointsSrcLeft[10].x = rectLeft.width - 1; pointsSrcLeft[10].y = pointsSrcLeft[0].y;
	pointsSrcLeft[11].x = pointsSrcLeft[0].x; pointsSrcLeft[11].y = startY;
	pointsSrcLeft[12].x = pointsSrcLeft[1].x; pointsSrcLeft[12].y = startY;
	pointsSrcLeft[13].x = pointsSrcLeft[2].x; pointsSrcLeft[13].y = startY;
	pointsSrcLeft[14].x = pointsSrcLeft[3].x; pointsSrcLeft[14].y = startY;
	pointsSrcLeft[15].x = pointsSrcLeft[4].x; pointsSrcLeft[15].y = startY;
	pointsSrcLeft[16].x = 0;                  pointsSrcLeft[16].y = pointsSrcLeft[4].y;
	pointsSrcLeft[17].x = 0;                  pointsSrcLeft[17].y = pointsSrcLeft[5].y;
	pointsSrcLeft[18].x = pointsSrcLeft[4].x; pointsSrcLeft[18].y = eyePointY;
	pointsSrcLeft[19].x = pointsSrcLeft[3].x; pointsSrcLeft[19].y = eyePointY;
	pointsSrcLeft[20].x = pointsSrcLeft[2].x; pointsSrcLeft[20].y = eyePointY;
	pointsSrcLeft[21].x = pointsSrcLeft[1].x; pointsSrcLeft[21].y = eyePointY;
	pointsSrcLeft[22].x = pointsSrcLeft[0].x; pointsSrcLeft[22].y = rectLeft.height - 1;
	pointsSrcLeft[23].x = rectLeft.width - 1; pointsSrcLeft[23].y = pointsSrcLeft[9].y;

	//destination
	std::vector<cv::Point> offsetsRight(N), offsetsLeft(N);
	for (int i = 0; i < 5; i++)
	{
		offsetsRight[i].x = (int)(((templatesRight[i].x - rectRight.x) - pointsSrcRight[i].x)*morphPercent);
		offsetsRight[i].y = (int)(((templatesRight[i].y - rectRight.y) - pointsSrcRight[i].y)*morphPercent);

		offsetsLeft[i].x = (int)(((templatesLeft[4 - i].x - rectLeft.x) - pointsSrcLeft[i].x)*morphPercent);
		offsetsLeft[i].y = (int)(((templatesLeft[4 - i].y - rectLeft.y) - pointsSrcLeft[i].y)*morphPercent);
	}
	if (eyebrowCurrentShapeIndex != shapeIndex || isStillImage)
	{
		//clearing previous data and assigning new.
		eyebrowOffsetSamplesRight.clear();
		eyebrowOffsetSamplesLeft.clear();
		for (int i = 0; i < 5; i++)
		{
			eyebrowOffsetsRight[i].x = offsetsRight[i].x;
			eyebrowOffsetsRight[i].y = offsetsRight[i].y;
			eyebrowOffsetsLeft[i].x = offsetsLeft[i].x;
			eyebrowOffsetsLeft[i].y = offsetsLeft[i].y;
		}
		eyebrowCurrentShapeIndex = shapeIndex;
	}
	if (!Utilities_Live::isMorphStable)
	{
		//taking average of offsets in this case of unstability(person moving)
		eyebrowOffsetSamplesRight.push_back(offsetsRight);
		eyebrowOffsetSamplesLeft.push_back(offsetsLeft);
		while (eyebrowOffsetSamplesRight.size() > Utilities_Live::fps)
		{
			eyebrowOffsetSamplesRight.pop_front();
			eyebrowOffsetSamplesLeft.pop_front();
		}
		int sampleCount = (int)eyebrowOffsetSamplesRight.size();
		cv::Point avgRight[N], avgLeft[N];
		for (int i = 0; i < sampleCount; i++)
		{
			for (int j = 0; j < N; j++)
			{
				avgRight[j] += eyebrowOffsetSamplesRight[i][j];
				avgLeft[j] += eyebrowOffsetSamplesLeft[i][j];
			}
		}
		for (int j = 0; j < N; j++)
		{
			avgRight[j] /= sampleCount;
			avgLeft[j] /= sampleCount;
		}
		//std::cout << sampleCount << std::endl;
		for (int i = 0; i < 5; i++)
		{
			pointsDstRight[i].x = pointsSrcRight[i].x + avgRight[i].x;  pointsDstRight[i].y = pointsSrcRight[i].y + avgRight[i].y;
			pointsDstLeft[i].x = pointsSrcLeft[i].x + avgLeft[i].x;  pointsDstLeft[i].y = pointsSrcLeft[i].y + avgLeft[i].y;
			//assigning new data to have it for else(stable) condition.
			eyebrowOffsetsRight[i].x = offsetsRight[i].x;
			eyebrowOffsetsRight[i].y = offsetsRight[i].y;
			eyebrowOffsetsLeft[i].x = offsetsLeft[i].x;
			eyebrowOffsetsLeft[i].y = offsetsLeft[i].y;
		}
	}
	else
	{
		for (int i = 0; i < 5; i++)
		{
			pointsDstRight[i].x = pointsSrcRight[i].x + eyebrowOffsetsRight[i].x;  pointsDstRight[i].y = pointsSrcRight[i].y + eyebrowOffsetsRight[i].y;
			pointsDstLeft[i].x = pointsSrcLeft[i].x + eyebrowOffsetsLeft[i].x;  pointsDstLeft[i].y = pointsSrcLeft[i].y + eyebrowOffsetsLeft[i].y;
		}
	}
	pointsDstRight[5].x = pointsDstRight[4].x; pointsDstRight[5].y = pointsDstRight[4].y + (pointsSrcRight[5].y - pointsSrcRight[4].y);
	pointsDstRight[6].x = pointsDstRight[3].x; pointsDstRight[6].y = pointsDstRight[3].y + (pointsSrcRight[6].y - pointsSrcRight[3].y);
	pointsDstRight[7].x = pointsDstRight[2].x; pointsDstRight[7].y = pointsDstRight[2].y + (pointsSrcRight[7].y - pointsSrcRight[2].y);
	pointsDstRight[8].x = pointsDstRight[1].x; pointsDstRight[8].y = pointsDstRight[1].y + (pointsSrcRight[8].y - pointsSrcRight[1].y);
	pointsDstRight[9].x = pointsDstRight[0].x; pointsDstRight[9].y = pointsDstRight[0].y + (pointsSrcRight[9].y - pointsSrcRight[0].y);

	pointsDstLeft[5].x = pointsDstLeft[4].x; pointsDstLeft[5].y = pointsDstLeft[4].y + (pointsSrcLeft[5].y - pointsSrcLeft[4].y);
	pointsDstLeft[6].x = pointsDstLeft[3].x; pointsDstLeft[6].y = pointsDstLeft[3].y + (pointsSrcLeft[6].y - pointsSrcLeft[3].y);
	pointsDstLeft[7].x = pointsDstLeft[2].x; pointsDstLeft[7].y = pointsDstLeft[2].y + (pointsSrcLeft[7].y - pointsSrcLeft[2].y);
	pointsDstLeft[8].x = pointsDstLeft[1].x; pointsDstLeft[8].y = pointsDstLeft[1].y + (pointsSrcLeft[8].y - pointsSrcLeft[1].y);
	pointsDstLeft[9].x = pointsDstLeft[0].x; pointsDstLeft[9].y = pointsDstLeft[0].y + (pointsSrcLeft[9].y - pointsSrcLeft[0].y);

	for (int i = 10; i < length; i++)
	{
		pointsDstRight[i].x = pointsSrcRight[i].x; pointsDstRight[i].y = pointsSrcRight[i].y;
		pointsDstLeft[i].x = pointsSrcLeft[i].x; pointsDstLeft[i].y = pointsSrcLeft[i].y;
	}
	int triSets[] = { 0,10,11, 0,1,11, 1,11,12, 1,2,12, 2,12,13, 2,3,13, 3,13,14, 3,4,14, 4,14,15, 4,16,15, 4,16,17,
		4,5,17, 5,4,3, 5,6,3, 6,3,2, 6,7,2, 7,2,1, 7,8,1, 8,1,0, 8,9,0, 9,0,10, 9,23,10,
		9,23,22, 22,21,9, 21,9,8, 21,20,8, 20,8,7, 20,19,7, 19,7,6, 19,18,6, 18,6,5, 18,17,5 };

	cv::Rect* rects[] = { &rectRight,&rectLeft };
	cv::Point* pointsSrc[] = { pointsSrcRight,pointsSrcLeft };
	cv::Point* pointsDst[] = { pointsDstRight,pointsDstLeft };
	for (int j = 0; j < 2; j++)
	{
		cv::Mat cropImage = inputImage(*rects[j]).clone();
		cv::Mat morphedImage = cropImage.clone();
		Utilities_Live::saveImage(morphedImage, "input.png");
		cv::Mat maskImage, tempImage;
		for (int i = 0; i < 96; i = i + 3)
		{
			affineTransformation(pointsSrc[j], pointsDst[j], triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
		}
		morphedImage.copyTo(inputImage(*rects[j]));
	}
	return true;
}

void Morphing::OverlayMorphs_Live::loadObjectsInfo(const std::vector<std::vector<cv::Point2f>>& contours, const float fldPoints[])
{
	objectsInfo.clear();
	for (int i = 0; i < fldPointsLength; i++)
	{
		objectsFldPoints[i].x = fldPoints[i * 2];
		objectsFldPoints[i].y = fldPoints[i * 2 + 1];
	}
	int n = (int)contours.size();
	for (int i = 0; i < n; i++)
	{
		ObjectInfo objInfo;
		objInfo.polygon = contours[i];
		int n1 = (int)objInfo.polygon.size();
		cv::Point2f point;
		for (int j = 0; j < n1; j++)
		{
			point.x += objInfo.polygon[j].x;
			point.y += objInfo.polygon[j].y;
		}
		objInfo.centroid.x = point.x / n1;
		objInfo.centroid.y = point.y / n1;
		objInfo.isApplied = false;
		objectsInfo.push_back(objInfo);
	}
}

std::vector<Morphing::ObjectInfo> Morphing::OverlayMorphs_Live::objectsOverlapping(const float fldPoints[], float thresholdFactor)
{
	float thresholds[3] = { thresholdFactor*20,thresholdFactor*0.2f, thresholdFactor*0.6f };
	if (!checkIQCForAffine(fldPoints, thresholds))
	{
		std::vector<ObjectInfo> objectsInfoDst;
		return objectsInfoDst;
	}
	cv::Point2f frameFldPoints[fldPointsLength];
	for (int i = 0; i < fldPointsLength; i++)
	{
		frameFldPoints[i].x = fldPoints[i * 2];
		frameFldPoints[i].y = fldPoints[i * 2 + 1];
	}
	int triSets[] = { 0,17,36, 0,1,36, 1,2,40, 1,36,40, 2,29,40, 29,31,2, 2,3,31, 3,31,48, 3,4,48, 16,26,45, 16,15,45, 15,14,47, 15,45,47, 14,29,47, 29,35,14, 13,14,35, 13,35,54, 12,13,54,
		4,5,48, 5,6,48, 48,58,6, 6,7,58, 7,8,58, 58,56,8, 8,9,56, 9,10,56, 56,54,10, 10,11,54, 11,12,54, 29,31,35, 31,48,51, 31,51,35, 35,54,51, 48,51,58, 54,51,56, 51,56,58, 21,22,27, 21,27,40,
		21,40,19, 19,40,36, 17,19,36, 22,27,47, 22,24,47, 24,47,45, 24,26,45, 40,29,27, 47,29,27, 0,17,68, 68,69,17, 17,19,69, 19,69,70, 19,21,70, 70,71,21, 21,22,71, 71,72,22, 22,24,72, 24,72,73,
		24,26,73, 73,74,26, 26,16,74 };
	std::vector<ObjectInfo> objectsInfoDst = affineTransformation(objectsFldPoints, frameFldPoints, triSets, 180, objectsInfo);
	return objectsInfoDst;
}

cv::Mat Morphing::OverlayMorphs_Live::eyelashAlignment(const cv::Mat& baselineImage, const int baselineFldPoints[], const cv::Mat& currentImage, const int currentFldPoints[], bool drawTriangles)
{
	const int N = 5;
	const int length = 24;
	cv::Point pointsSrc[length], pointsDst[length];
	pointsSrc[0].x = currentFldPoints[0 * 2]; pointsSrc[0].y = currentFldPoints[0 * 2 + 1];
	pointsSrc[1].x = currentFldPoints[1 * 2]; pointsSrc[1].y = currentFldPoints[1 * 2 + 1];
	pointsSrc[2].x = currentFldPoints[2 * 2]; pointsSrc[2].y = currentFldPoints[2 * 2 + 1];
	pointsSrc[3].x = currentFldPoints[3 * 2]; pointsSrc[3].y = currentFldPoints[3 * 2 + 1];
	pointsSrc[4].x = currentFldPoints[4 * 2]; pointsSrc[4].y = currentFldPoints[4 * 2 + 1];
	int lashEndY = (int)(baselineImage.rows*0.9);
	pointsSrc[5].x = pointsSrc[4].x; pointsSrc[5].y = lashEndY;
	pointsSrc[6].x = pointsSrc[3].x; pointsSrc[6].y = lashEndY;
	pointsSrc[7].x = pointsSrc[2].x; pointsSrc[7].y = lashEndY;
	pointsSrc[8].x = pointsSrc[1].x; pointsSrc[8].y = lashEndY;
	pointsSrc[9].x = pointsSrc[0].x; pointsSrc[9].y = lashEndY;
	//support coords
	pointsSrc[10].x = 0;              pointsSrc[10].y = pointsSrc[0].y;
	pointsSrc[11].x = pointsSrc[0].x; pointsSrc[11].y = 0;
	pointsSrc[12].x = pointsSrc[1].x; pointsSrc[12].y = 0;
	pointsSrc[13].x = pointsSrc[2].x; pointsSrc[13].y = 0;
	pointsSrc[14].x = pointsSrc[3].x; pointsSrc[14].y = 0;
	pointsSrc[15].x = pointsSrc[4].x; pointsSrc[15].y = 0;
	pointsSrc[16].x = baselineImage.cols - 1; pointsSrc[16].y = pointsSrc[4].y;
	pointsSrc[17].x = baselineImage.cols - 1; pointsSrc[17].y = pointsSrc[5].y;
	pointsSrc[18].x = pointsSrc[4].x; pointsSrc[18].y = baselineImage.rows - 1;
	pointsSrc[19].x = pointsSrc[3].x; pointsSrc[19].y = baselineImage.rows - 1;
	pointsSrc[20].x = pointsSrc[2].x; pointsSrc[20].y = baselineImage.rows - 1;
	pointsSrc[21].x = pointsSrc[1].x; pointsSrc[21].y = baselineImage.rows - 1;
	pointsSrc[22].x = pointsSrc[0].x; pointsSrc[22].y = baselineImage.rows - 1;
	pointsSrc[23].x = 0;              pointsSrc[23].y = pointsSrc[9].y;

	//destination
	pointsDst[0].x = baselineFldPoints[0 * 2]; pointsDst[0].y = baselineFldPoints[0 * 2 + 1];
	pointsDst[1].x = baselineFldPoints[1 * 2]; pointsDst[1].y = baselineFldPoints[1 * 2 + 1];
	pointsDst[2].x = baselineFldPoints[2 * 2]; pointsDst[2].y = baselineFldPoints[2 * 2 + 1];
	pointsDst[3].x = baselineFldPoints[3 * 2]; pointsDst[3].y = baselineFldPoints[3 * 2 + 1];
	pointsDst[4].x = baselineFldPoints[4 * 2]; pointsDst[4].y = baselineFldPoints[4 * 2 + 1];
	pointsDst[5].x = pointsDst[4].x; pointsDst[5].y = pointsDst[4].y + (pointsSrc[5].y - pointsSrc[4].y);
	pointsDst[6].x = pointsDst[3].x; pointsDst[6].y = pointsDst[3].y + (pointsSrc[6].y - pointsSrc[3].y);
	pointsDst[7].x = pointsDst[2].x; pointsDst[7].y = pointsDst[2].y + (pointsSrc[7].y - pointsSrc[2].y);
	pointsDst[8].x = pointsDst[1].x; pointsDst[8].y = pointsDst[1].y + (pointsSrc[8].y - pointsSrc[1].y);
	pointsDst[9].x = pointsDst[0].x; pointsDst[9].y = pointsDst[0].y + (pointsSrc[9].y - pointsSrc[0].y);
	for (int i = 10; i < length; i++)
	{
		pointsDst[i].x = pointsSrc[i].x; pointsDst[i].y = pointsSrc[i].y;
	}
	int triSets[] = { 0,10,11, 0,1,11, 1,11,12, 1,2,12, 2,12,13, 2,3,13, 3,13,14, 3,4,14, 4,14,15, 4,16,15, 4,16,17,
		4,5,17, 5,4,3, 5,6,3, 6,3,2, 6,7,2, 7,2,1, 7,8,1, 8,1,0, 8,9,0, 9,0,10, 9,23,10,
		9,23,22, 22,21,9, 21,9,8, 21,20,8, 20,8,7, 20,19,7, 19,7,6, 19,18,6, 18,6,5, 18,17,5 };

	cv::Mat alignedImage = currentImage.clone();
	cv::Mat morphedImage = cv::Mat::zeros(currentImage.rows, currentImage.cols, currentImage.type());
	cv::Mat maskImage, tempImage;
	for (int i = 0; i < 96; i = i + 3)
	{
		affineTransformation(pointsSrc, pointsDst, triSets, i, alignedImage, maskImage, tempImage, morphedImage, drawTriangles);
	}


	const uchar*  baselinePtr = baselineImage.data;
	const uchar*  morphedPtr = morphedImage.data;
	uchar* alignedPtr = alignedImage.data;
	int baselineStep = (int)baselineImage.step;
	int  morphedStep = (int)morphedImage.step;
	int alignedStep = (int)alignedImage.step;
	int colPixels = baselineImage.cols * 3;
	for (int i = 0; i < baselineImage.rows; i++)
	{
		for (int j = 0; j < colPixels; j += 3)
		{
			if (baselinePtr[j] > 0)
			{
				alignedPtr[j] = 0;
				alignedPtr[j + 1] = 255;
				alignedPtr[j + 2] = 255;
			}
			else if (morphedPtr[j] > 0)
			{
				alignedPtr[j] = 255;
				alignedPtr[j + 1] = 0;
				alignedPtr[j + 2] = 0;
			}
			else
			{
				alignedPtr[j] = 0;
				alignedPtr[j + 1] = 0;
				alignedPtr[j + 2] = 0;
			}
		}
		baselinePtr += baselineStep;
		morphedPtr += morphedStep;
		alignedPtr += alignedStep;
	}
	return alignedImage;
}

