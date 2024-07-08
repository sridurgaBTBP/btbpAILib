#include "AreaMorphs_Live.h"
#include "Utilities_Live.h"
#include <opencv2/imgproc.hpp>

int Morphing::AreaMorphs_Live::maxLipTop = 0, Morphing::AreaMorphs_Live::maxLipBottom = 0;
int Morphing::AreaMorphs_Live::maxEyeRight = 0, Morphing::AreaMorphs_Live::maxEyeLeft = 0;
int Morphing::AreaMorphs_Live::maxNose = 0;
int Morphing::AreaMorphs_Live::maxJawRight = 0, Morphing::AreaMorphs_Live::maxJawLeft = 0;
int Morphing::AreaMorphs_Live::maxEyebrowRight = 0, Morphing::AreaMorphs_Live::maxEyebrowLeft = 0;
int Morphing::AreaMorphs_Live::maxLipCornerRight = 0, Morphing::AreaMorphs_Live::maxLipCornerLeft = 0;

void  Morphing::AreaMorphs_Live::lipMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 3;//min to do the operation
	int morphSpaceTop = std::min(fldPoints[61 * 2 + 1] - fldPoints[50 * 2 + 1], std::min(fldPoints[62 * 2 + 1] - fldPoints[51 * 2 + 1], fldPoints[63 * 2 + 1] - fldPoints[52 * 2 + 1])) + 1;
	int morphSpaceBottom = std::min(fldPoints[56 * 2 + 1] - fldPoints[65 * 2 + 1], std::min(fldPoints[57 * 2 + 1] - fldPoints[66 * 2 + 1], fldPoints[58 * 2 + 1] - fldPoints[67 * 2 + 1])) + 1;
	if (morphSpaceTop < thresh || morphSpaceBottom < thresh)
		return;
	int max = maxLipTop;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceTop*0.3f);
		if (max == 0)
			max = 1;
		maxLipTop = max;
	}
	int min = -max;
	int offsetTop = (int)std::round(morphPercent*(max - min) + min);
	if (offsetTop == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetTop = -1;
		else
			offsetTop = 1;
	}
	cv::Rect rect;
	rect.x = fldPoints[2 * 48];
	rect.y = std::min(fldPoints[50 * 2 + 1], fldPoints[52 * 2 + 1]) - (2 * max) > fldPoints[33 * 2 + 1] ? std::min(fldPoints[50 * 2 + 1], fldPoints[52 * 2 + 1]) - (2 * max) : fldPoints[33 * 2 + 1];
	rect.width = fldPoints[2 * 54] - rect.x + 1;
	max = maxLipBottom;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceBottom*0.3f);
		if (max == 0)
			max = 1;
		maxLipBottom = max;
	}
	min = -max;
	int offsetBottom = (int)std::round(morphPercent*(max - min) + min);
	if (offsetBottom == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetBottom = -1;
		else
			offsetBottom = 1;
	}
	rect.height = std::max(std::max(fldPoints[56 * 2 + 1], fldPoints[57 * 2 + 1]), fldPoints[58 * 2 + 1]) + (2 * max) - rect.y + 1;
	if (checkBoundsForAffine(rect, inputImage.cols, inputImage.rows))
		return;
	const int length = 21;
	cv::Point pointsSrc[length];
	pointsSrc[0].x = fldPoints[48 * 2] - rect.x; pointsSrc[0].y = fldPoints[48 * 2 + 1] - rect.y;
	pointsSrc[1].x = fldPoints[50 * 2] - rect.x; pointsSrc[1].y = fldPoints[50 * 2 + 1] - rect.y;
	pointsSrc[2].x = fldPoints[52 * 2] - rect.x; pointsSrc[2].y = fldPoints[52 * 2 + 1] - rect.y;
	pointsSrc[3].x = fldPoints[54 * 2] - rect.x; pointsSrc[3].y = fldPoints[54 * 2 + 1] - rect.y;
	pointsSrc[4].x = fldPoints[55 * 2] - rect.x; pointsSrc[4].y = fldPoints[55 * 2 + 1] - rect.y;
	pointsSrc[5].x = fldPoints[56 * 2] - rect.x; pointsSrc[5].y = fldPoints[56 * 2 + 1] - rect.y;
	pointsSrc[6].x = fldPoints[57 * 2] - rect.x; pointsSrc[6].y = fldPoints[57 * 2 + 1] - rect.y;
	pointsSrc[7].x = fldPoints[58 * 2] - rect.x; pointsSrc[7].y = fldPoints[58 * 2 + 1] - rect.y;
	pointsSrc[8].x = fldPoints[59 * 2] - rect.x; pointsSrc[8].y = fldPoints[59 * 2 + 1] - rect.y;
	//support coords
	pointsSrc[9].x = fldPoints[49 * 2] - rect.x;  pointsSrc[9].y = 0;
	pointsSrc[10].x = pointsSrc[1].x;             pointsSrc[10].y = 0;
	pointsSrc[11].x = pointsSrc[2].x;             pointsSrc[11].y = 0;
	pointsSrc[12].x = fldPoints[53 * 2] - rect.x; pointsSrc[12].y = 0;
	pointsSrc[13].x = pointsSrc[4].x;             pointsSrc[13].y = rect.height - 1;
	pointsSrc[14].x = pointsSrc[5].x;             pointsSrc[14].y = rect.height - 1;
	pointsSrc[15].x = pointsSrc[7].x;             pointsSrc[15].y = rect.height - 1;
	pointsSrc[16].x = pointsSrc[8].x;             pointsSrc[16].y = rect.height - 1;
	pointsSrc[17].x = fldPoints[61 * 2] - rect.x; pointsSrc[17].y = fldPoints[61 * 2 + 1] - rect.y;
	pointsSrc[18].x = fldPoints[63 * 2] - rect.x; pointsSrc[18].y = fldPoints[63 * 2 + 1] - rect.y;
	pointsSrc[19].x = fldPoints[65 * 2] - rect.x; pointsSrc[19].y = fldPoints[65 * 2 + 1] - rect.y;
	pointsSrc[20].x = fldPoints[67 * 2] - rect.x; pointsSrc[20].y = fldPoints[67 * 2 + 1] - rect.y;

	cv::Point pointsDst[length];
	for (int i = 0; i < length; i++)
	{
		pointsDst[i].x = pointsSrc[i].x; pointsDst[i].y = pointsSrc[i].y;
	}
	int triSets[] = { 0,1,9, 1,9,10, 1,10,11, 1,2,11, 2,11,12, 2,3,12, 3,4,13, 4,5,13, 5,14,13, 5,6,14, 6,14,15, 6,7,15, 7,15,16,
		7,8,16, 0,8,16, 0,1,17, 1,2,17, 2,18,17, 2,3,18, 3,4,19, 4,5,19, 5,6,19, 19,20,6, 6,7,20, 7,8,20, 0,8,20 };
	//adding offsets to Y positions to make lip plump/slim
	pointsDst[1].y -= offsetTop;
	pointsDst[2].y -= offsetTop;
	pointsDst[4].y += offsetBottom;
	pointsDst[5].y += offsetBottom;
	pointsDst[6].y += offsetBottom;
	pointsDst[7].y += offsetBottom;
	pointsDst[8].y += offsetBottom;
	cv::Mat cropImage = inputImage(rect).clone();
	Utilities_Live::saveImage(cropImage, "input.png");
	cv::Mat morphedImage = cropImage.clone();
	cv::Mat maskImage, tempImage;
	for (int i = 0; i < 78; i = i + 3)
	{
		affineTransformation(pointsSrc, pointsDst, triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
	}
	morphedImage.copyTo(inputImage(rect));
}

void  Morphing::AreaMorphs_Live::eyeMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 4;//min to do the operation
	int eyePoint/*sync with eyebrow morph*/ = std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (int)((std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (fldPoints[18 * 2 + 1] + fldPoints[19 * 2 + 1] + fldPoints[20 * 2 + 1]) / 3)*0.15);
	cv::Rect rectRight;
	rectRight.x = std::min(fldPoints[36 * 2], fldPoints[17 * 2]);
	rectRight.y = std::min((std::min((eyePoint + fldPoints[18 * 2 + 1]) / 2, (eyePoint + fldPoints[19 * 2 + 1]) / 2)), (eyePoint + fldPoints[20 * 2 + 1]) / 2);
	rectRight.width = std::max(fldPoints[39 * 2], fldPoints[21 * 2]) - rectRight.x + 1;
	const int length = 15;
	cv::Point pointsSrcRight[length];
	pointsSrcRight[0].x = (int)(rectRight.width*0.33); pointsSrcRight[0].y = (int)(fldPoints[37 * 2 + 1] - (fldPoints[37 * 2 + 1] - rectRight.y)*0.4) - rectRight.y;
	pointsSrcRight[1].x = (int)(rectRight.width*0.66); pointsSrcRight[1].y = (int)(fldPoints[38 * 2 + 1] - (fldPoints[38 * 2 + 1] - rectRight.y)*0.4) - rectRight.y;
	pointsSrcRight[2].x = pointsSrcRight[1].x;         pointsSrcRight[2].y = (int)(fldPoints[40 * 2 + 1] + (fldPoints[38 * 2 + 1] - rectRight.y)*0.4) - rectRight.y;//adding same offset on bottom also
	pointsSrcRight[3].x = pointsSrcRight[0].x;         pointsSrcRight[3].y = (int)(fldPoints[41 * 2 + 1] + (fldPoints[37 * 2 + 1] - rectRight.y)*0.4) - rectRight.y;
	int morphSpaceRight = std::min(pointsSrcRight[2].y - pointsSrcRight[1].y, pointsSrcRight[3].y - pointsSrcRight[0].y) + 1;
	if (morphSpaceRight < thresh)
		return;
	int max = maxEyeRight;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceRight*0.1f);//upto max_eye=0.3*space, the const factor 0.5 will work.if it has to be >0.3,the factor might be >0.5
		if (max == 0)
			max = 1;
		maxEyeRight = max;
	}
	int min = -max;
	int offsetRight = (int)std::round(morphPercent*(max - min) + min);
	if (offsetRight == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetRight = -1;
		else
			offsetRight = 1;
	}
	rectRight.height = std::max(pointsSrcRight[2].y, pointsSrcRight[3].y) + (int)(2.5 * max) + 1;
	if (checkBoundsForAffine(rectRight, inputImage.cols, inputImage.rows))
		return;
	//support coords
	pointsSrcRight[4].x = fldPoints[17 * 2] - rectRight.x; pointsSrcRight[4].y = (eyePoint + fldPoints[17 * 2 + 1]) / 2 - rectRight.y;
	pointsSrcRight[5].x = fldPoints[18 * 2] - rectRight.x; pointsSrcRight[5].y = (eyePoint + fldPoints[18 * 2 + 1]) / 2 - rectRight.y;
	pointsSrcRight[6].x = fldPoints[19 * 2] - rectRight.x; pointsSrcRight[6].y = (eyePoint + fldPoints[19 * 2 + 1]) / 2 - rectRight.y;
	pointsSrcRight[7].x = fldPoints[20 * 2] - rectRight.x; pointsSrcRight[7].y = (eyePoint + fldPoints[20 * 2 + 1]) / 2 - rectRight.y;
	pointsSrcRight[8].x = fldPoints[21 * 2] - rectRight.x; pointsSrcRight[8].y = (eyePoint + fldPoints[21 * 2 + 1]) / 2 - rectRight.y;
	pointsSrcRight[9].x = rectRight.width - 1;             pointsSrcRight[9].y = pointsSrcRight[2].y;
	pointsSrcRight[10].x = rectRight.width - 1;            pointsSrcRight[10].y = rectRight.height - 1;
	pointsSrcRight[11].x = pointsSrcRight[2].x;            pointsSrcRight[11].y = rectRight.height - 1;
	pointsSrcRight[12].x = pointsSrcRight[3].x;            pointsSrcRight[12].y = rectRight.height - 1;
	pointsSrcRight[13].x = 0;                              pointsSrcRight[13].y = rectRight.height - 1;
	pointsSrcRight[14].x = 0;                              pointsSrcRight[14].y = pointsSrcRight[3].y;


	eyePoint/*sync with eyebrow morph*/ = std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (int)((std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (fldPoints[23 * 2 + 1] + fldPoints[24 * 2 + 1] + fldPoints[25 * 2 + 1]) / 3)*0.15);
	cv::Rect rectLeft;
	rectLeft.x = std::min(fldPoints[42 * 2], fldPoints[22 * 2]);
	rectLeft.y = std::min((std::min((eyePoint + fldPoints[23 * 2 + 1]) / 2, (eyePoint + fldPoints[24 * 2 + 1]) / 2)), (eyePoint + fldPoints[25 * 2 + 1]) / 2);
	rectLeft.width = std::max(fldPoints[45 * 2], fldPoints[26 * 2]) - rectLeft.x + 1;
	cv::Point pointsSrcLeft[length];
	pointsSrcLeft[0].x = (int)(rectLeft.width*0.66); pointsSrcLeft[0].y = (int)(fldPoints[44 * 2 + 1] - (fldPoints[44 * 2 + 1] - rectLeft.y)*0.4) - rectLeft.y;
	pointsSrcLeft[1].x = (int)(rectLeft.width*0.33); pointsSrcLeft[1].y = (int)(fldPoints[43 * 2 + 1] - (fldPoints[43 * 2 + 1] - rectLeft.y)*0.4) - rectLeft.y;
	pointsSrcLeft[2].x = pointsSrcLeft[1].x;         pointsSrcLeft[2].y = (int)(fldPoints[47 * 2 + 1] + (fldPoints[43 * 2 + 1] - rectLeft.y)*0.4) - rectLeft.y;//adding same offset on bottom also
	pointsSrcLeft[3].x = pointsSrcLeft[0].x;         pointsSrcLeft[3].y = (int)(fldPoints[46 * 2 + 1] + (fldPoints[44 * 2 + 1] - rectLeft.y)*0.4) - rectLeft.y;
	int morphSpaceLeft = std::min(pointsSrcLeft[2].y - pointsSrcLeft[1].y, pointsSrcLeft[3].y - pointsSrcLeft[0].y) + 1;
	if (morphSpaceLeft < thresh)
		return;
	max = maxEyeLeft;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceLeft*0.1f);
		if (max == 0)
			max = 1;
		maxEyeLeft = max;
	}
	min = -max;
	int offsetLeft = (int)std::round(morphPercent*(max - min) + min);
	if (offsetLeft == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetLeft = -1;
		else
			offsetLeft = 1;
	}
	rectLeft.height = std::max(pointsSrcLeft[2].y, pointsSrcLeft[3].y) + (int)(2.5 * max) + 1;
	if (checkBoundsForAffine(rectLeft, inputImage.cols, inputImage.rows))
		return;
	//support coords
	pointsSrcLeft[4].x = fldPoints[26 * 2] - rectLeft.x; pointsSrcLeft[4].y = (eyePoint + fldPoints[26 * 2 + 1]) / 2 - rectLeft.y;
	pointsSrcLeft[5].x = fldPoints[25 * 2] - rectLeft.x; pointsSrcLeft[5].y = (eyePoint + fldPoints[25 * 2 + 1]) / 2 - rectLeft.y;
	pointsSrcLeft[6].x = fldPoints[24 * 2] - rectLeft.x; pointsSrcLeft[6].y = (eyePoint + fldPoints[24 * 2 + 1]) / 2 - rectLeft.y;
	pointsSrcLeft[7].x = fldPoints[23 * 2] - rectLeft.x; pointsSrcLeft[7].y = (eyePoint + fldPoints[23 * 2 + 1]) / 2 - rectLeft.y;
	pointsSrcLeft[8].x = fldPoints[22 * 2] - rectLeft.x; pointsSrcLeft[8].y = (eyePoint + fldPoints[22 * 2 + 1]) / 2 - rectLeft.y;
	pointsSrcLeft[9].x = 0;                              pointsSrcLeft[9].y = pointsSrcLeft[2].y;
	pointsSrcLeft[10].x = 0;                             pointsSrcLeft[10].y = rectLeft.height - 1;
	pointsSrcLeft[11].x = pointsSrcLeft[2].x;            pointsSrcLeft[11].y = rectLeft.height - 1;
	pointsSrcLeft[12].x = pointsSrcLeft[3].x;            pointsSrcLeft[12].y = rectLeft.height - 1;
	pointsSrcLeft[13].x = rectLeft.width - 1;            pointsSrcLeft[13].y = rectLeft.height - 1;
	pointsSrcLeft[14].x = rectLeft.width - 1;            pointsSrcLeft[14].y = pointsSrcLeft[3].y;

	cv::Point pointsDstRight[length], pointsDstLeft[length];
	for (int i = 0; i < length; i++)
	{
		pointsDstRight[i].x = pointsSrcRight[i].x; pointsDstRight[i].y = pointsSrcRight[i].y;
		pointsDstLeft[i].x = pointsSrcLeft[i].x; pointsDstLeft[i].y = pointsSrcLeft[i].y;
	}
	int triSets[] = { 0,4,5, 0,5,6, 0,6,1, 1,6,7, 1,7,8, 1,8,9, 1,9,2, 9,10,11, 9,2,11,
		2,11,12, 2,3,12, 3,12,14, 12,13,14, 0,3,14, 0,14,4, 0,1,3, 1,2,3 };
	//adding offsets to both X and Y positions to make eye enlarge/reduce.
	pointsDstRight[0].x -= offsetRight; pointsDstRight[0].y -= offsetRight;
	pointsDstRight[1].x += offsetRight; pointsDstRight[1].y -= offsetRight;
	pointsDstRight[2].x += offsetRight; pointsDstRight[2].y += offsetRight;
	pointsDstRight[3].x -= offsetRight; pointsDstRight[3].y += offsetRight;

	pointsDstLeft[0].x += offsetLeft; pointsDstLeft[0].y -= offsetLeft;
	pointsDstLeft[1].x -= offsetLeft; pointsDstLeft[1].y -= offsetLeft;
	pointsDstLeft[2].x -= offsetLeft; pointsDstLeft[2].y += offsetLeft;
	pointsDstLeft[3].x += offsetLeft; pointsDstLeft[3].y += offsetLeft;

	cv::Rect* rects[] = { &rectRight,&rectLeft };
	cv::Point* pointsSrc[] = { pointsSrcRight,pointsSrcLeft };
	cv::Point* pointsDst[] = { pointsDstRight,pointsDstLeft };
	for (int j = 0; j < 2; j++)
	{
		cv::Mat cropImage = inputImage(*rects[j]).clone();
		Utilities_Live::saveImage(cropImage, "input.png");
		cv::Mat morphedImage = cropImage.clone();
		cv::Mat maskImage, tempImage;
		for (int i = 0; i < 51; i = i + 3)
		{
			affineTransformation(pointsSrc[j], pointsDst[j], triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
		}
		morphedImage.copyTo(inputImage(*rects[j]));
	}
}

void  Morphing::AreaMorphs_Live::noseMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 4;//min to do the operation
	int startX = fldPoints[31 * 2] - (int)((fldPoints[35 * 2] - fldPoints[31 * 2])*0.4);
	int endX = fldPoints[35 * 2] + (int)((fldPoints[35 * 2] - fldPoints[31 * 2])*0.4);
	int morphSpace = endX - startX + 1;
	if (morphSpace < thresh)
		return;
	int max = maxNose;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpace*0.05f);
		if (max == 0)
			max = 1;
		maxNose = max;
	}
	int min = -max;
	int offset = (int)std::round(morphPercent*(max - min) + min);
	if (offset == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offset = -1;
		else
			offset = 1;
	}
	int eyePointY = std::max(std::max(fldPoints[41 * 2 + 1], fldPoints[40 * 2 + 1]), std::max(fldPoints[47 * 2 + 1], fldPoints[46 * 2 + 1]));
	cv::Rect rect;
	rect.x = startX - (2 * max);
	rect.y = eyePointY + (int)((fldPoints[30 * 2 + 1] - eyePointY) *0.2);
	rect.width = endX + (2 * max) - rect.x + 1;
	rect.height = fldPoints[33 * 2 + 1] + (int)((fldPoints[51 * 2 + 1] - fldPoints[33 * 2 + 1])*0.3) - rect.y + 1;
	if (checkBoundsForAffine(rect, inputImage.cols, inputImage.rows))
		return;
	const int length = 13;
	cv::Point pointsSrc[length];
	pointsSrc[0].x = startX - rect.x; pointsSrc[0].y = (int)(0.1*rect.height);
	pointsSrc[1].x = endX - rect.x;   pointsSrc[1].y = pointsSrc[0].y;
	pointsSrc[2].x = pointsSrc[1].x;  pointsSrc[2].y = fldPoints[33 * 2 + 1] - rect.y;
	pointsSrc[3].x = pointsSrc[0].x;  pointsSrc[3].y = pointsSrc[2].y;
	//support coords
	pointsSrc[4].x = 0;               pointsSrc[4].y = 0;
	pointsSrc[5].x = rect.width / 2;  pointsSrc[5].y = 0;
	pointsSrc[6].x = rect.width - 1;  pointsSrc[6].y = 0;
	pointsSrc[7].x = rect.width - 1;  pointsSrc[7].y = rect.height / 2;
	pointsSrc[8].x = rect.width - 1;  pointsSrc[8].y = rect.height - 1;
	pointsSrc[9].x = rect.width / 2;  pointsSrc[9].y = rect.height - 1;
	pointsSrc[10].x = 0;              pointsSrc[10].y = rect.height - 1;
	pointsSrc[11].x = 0;              pointsSrc[11].y = rect.height / 2;
	pointsSrc[12].x = (pointsSrc[0].x + pointsSrc[2].x) / 2; pointsSrc[12].y = (pointsSrc[0].y + pointsSrc[2].y) / 2;

	cv::Point pointsDst[length];
	for (int i = 0; i < length; i++)
	{
		pointsDst[i].x = pointsSrc[i].x; pointsDst[i].y = pointsSrc[i].y;
	}
	int triSets[] = { 0,3,12, 0,1,12, 1,2,12, 2,3,12, 0,4,11, 0,4,5, 0,1,5, 1,6,5,
		1,6,7, 2,7,1, 2,7,8, 2,9,8, 2,9,3, 10,3,9, 10,3,11, 0,3,11 };
	//adding offsets to X positions to make nose broad/narrow
	pointsDst[0].x -= offset;
	pointsDst[1].x += offset;
	pointsDst[2].x += offset;
	pointsDst[3].x -= offset;

	cv::Mat cropImage = inputImage(rect).clone();
	Utilities_Live::saveImage(cropImage, "input.png");
	cv::Mat morphedImage = cropImage.clone();
	cv::Mat maskImage, tempImage;
	for (int i = 0; i < 48; i = i + 3)
	{
		affineTransformation(pointsSrc, pointsDst, triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
	}
	morphedImage.copyTo(inputImage(rect));
}

void  Morphing::AreaMorphs_Live::jawlineMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 3;//min to do the operation
	int morphSpaceRight = fldPoints[48 * 2] - fldPoints[4 * 2] + 1;
	int morphSpaceLeft = fldPoints[12 * 2] - fldPoints[54 * 2] + 1;
	if (morphSpaceRight < thresh || morphSpaceLeft < thresh)
		return;
	int max = maxJawRight;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceRight*0.1f);
		if (max == 0)
			max = 1;
		maxJawRight = max;
	}
	int min = -max;
	int offsetRight = (int)std::round(morphPercent*(max - min) + min);
	if (offsetRight == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetRight = -1;
		else
			offsetRight = 1;
	}
	int paddX = (int)(2.5 * max);
	cv::Rect rectRight;
	if (fldPoints[2 * 2] < (fldPoints[3 * 2] - paddX))
		rectRight.x = fldPoints[2 * 2];
	else
		rectRight.x = fldPoints[3 * 2] - paddX;
	rectRight.y = fldPoints[2 * 2 + 1];
	if (fldPoints[6 * 2] > (fldPoints[5 * 2] + paddX))
		rectRight.width = fldPoints[6 * 2] - rectRight.x + 1;
	else
		rectRight.width = fldPoints[5 * 2] + paddX - rectRight.x + 1;
	rectRight.height = fldPoints[6 * 2 + 1] - rectRight.y + 1;
	if (checkBoundsForAffine(rectRight, inputImage.cols, inputImage.rows))
		return;
	const int length = 11;
	cv::Point pointsSrcRight[length];
	pointsSrcRight[0].x = fldPoints[3 * 2] - rectRight.x; pointsSrcRight[0].y = fldPoints[3 * 2 + 1] - rectRight.y;
	pointsSrcRight[1].x = fldPoints[4 * 2] - rectRight.x; pointsSrcRight[1].y = fldPoints[4 * 2 + 1] - rectRight.y;
	pointsSrcRight[2].x = fldPoints[5 * 2] - rectRight.x; pointsSrcRight[2].y = fldPoints[5 * 2 + 1] - rectRight.y;
	//support coords
	pointsSrcRight[3].x = fldPoints[2 * 2] - rectRight.x; pointsSrcRight[3].y = 0;
	pointsSrcRight[4].x = rectRight.width - 1;            pointsSrcRight[4].y = 0;
	pointsSrcRight[5].x = rectRight.width - 1;            pointsSrcRight[5].y = (pointsSrcRight[0].y + pointsSrcRight[1].y) / 2;
	pointsSrcRight[6].x = rectRight.width - 1;            pointsSrcRight[6].y = (pointsSrcRight[1].y + pointsSrcRight[2].y) / 2;
	pointsSrcRight[7].x = fldPoints[6 * 2] - rectRight.x; pointsSrcRight[7].y = rectRight.height - 1;
	pointsSrcRight[8].x = pointsSrcRight[2].x - paddX;    pointsSrcRight[8].y = pointsSrcRight[2].y;
	pointsSrcRight[9].x = pointsSrcRight[1].x - paddX;    pointsSrcRight[9].y = pointsSrcRight[1].y;
	pointsSrcRight[10].x = pointsSrcRight[0].x - paddX;   pointsSrcRight[10].y = pointsSrcRight[0].y;

	max = maxJawLeft;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceLeft*0.1f);
		if (max == 0)
			max = 1;
		maxJawLeft = max;
	}
	min = -max;
	int offsetLeft = (int)std::round(morphPercent*(max - min) + min);
	if (offsetLeft == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetLeft = -1;
		else
			offsetLeft = 1;
	}
	paddX = (int)(2.5 * max);
	cv::Rect rectLeft;
	if (fldPoints[10 * 2] < (fldPoints[11 * 2] - paddX))
		rectLeft.x = fldPoints[10 * 2];
	else
		rectLeft.x = fldPoints[11 * 2] - paddX;
	rectLeft.y = fldPoints[14 * 2 + 1];
	if (fldPoints[14 * 2] > (fldPoints[13 * 2] + paddX))
		rectLeft.width = fldPoints[14 * 2] - rectLeft.x + 1;
	else
		rectLeft.width = fldPoints[13 * 2] + paddX - rectLeft.x + 1;
	rectLeft.height = fldPoints[10 * 2 + 1] - rectLeft.y + 1;
	if (checkBoundsForAffine(rectLeft, inputImage.cols, inputImage.rows))
		return;
	cv::Point pointsSrcLeft[length];
	pointsSrcLeft[0].x = fldPoints[13 * 2] - rectLeft.x; pointsSrcLeft[0].y = fldPoints[13 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[1].x = fldPoints[12 * 2] - rectLeft.x; pointsSrcLeft[1].y = fldPoints[12 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[2].x = fldPoints[11 * 2] - rectLeft.x; pointsSrcLeft[2].y = fldPoints[11 * 2 + 1] - rectLeft.y;
	//support coords
	pointsSrcLeft[3].x = fldPoints[14 * 2] - rectLeft.x; pointsSrcLeft[3].y = 0;
	pointsSrcLeft[4].x = 0;                              pointsSrcLeft[4].y = 0;
	pointsSrcLeft[5].x = 0;                              pointsSrcLeft[5].y = (pointsSrcLeft[0].y + pointsSrcLeft[1].y) / 2;
	pointsSrcLeft[6].x = 0;                              pointsSrcLeft[6].y = (pointsSrcLeft[1].y + pointsSrcLeft[2].y) / 2;
	pointsSrcLeft[7].x = fldPoints[10 * 2] - rectLeft.x; pointsSrcLeft[7].y = rectLeft.height - 1;
	pointsSrcLeft[8].x = pointsSrcLeft[2].x + paddX;     pointsSrcLeft[8].y = pointsSrcLeft[2].y;
	pointsSrcLeft[9].x = pointsSrcLeft[1].x + paddX;     pointsSrcLeft[9].y = pointsSrcLeft[1].y;
	pointsSrcLeft[10].x = pointsSrcLeft[0].x + paddX;    pointsSrcLeft[10].y = pointsSrcLeft[0].y;

	cv::Point pointsDstRight[length], pointsDstLeft[length];
	for (int i = 0; i < length; i++)
	{
		pointsDstRight[i].x = pointsSrcRight[i].x; pointsDstRight[i].y = pointsSrcRight[i].y;
		pointsDstLeft[i].x = pointsSrcLeft[i].x; pointsDstLeft[i].y = pointsSrcLeft[i].y;
	}
	int triSets[] = { 0,10,3, 0,3,4, 0,4,5, 0,1,5, 0,1,10, 1,9,10,
		1,5,6, 1,2,6, 1,2,9, 2,8,9, 2,6,7, 2,7,8 };
	//adding offsets to X positions to make jawline broad/narrow
	pointsDstRight[0].x -= offsetRight;
	pointsDstRight[1].x -= offsetRight;
	pointsDstRight[2].x -= offsetRight;

	pointsDstLeft[0].x += offsetLeft;
	pointsDstLeft[1].x += offsetLeft;
	pointsDstLeft[2].x += offsetLeft;

	cv::Rect* rects[] = { &rectRight,&rectLeft };
	cv::Point* pointsSrc[] = { pointsSrcRight,pointsSrcLeft };
	cv::Point* pointsDst[] = { pointsDstRight,pointsDstLeft };
	for (int j = 0; j < 2; j++)
	{
		cv::Mat cropImage = inputImage(*rects[j]).clone();
		Utilities_Live::saveImage(cropImage, "input.png");
		cv::Mat morphedImage = cropImage.clone();
		cv::Mat maskImage, tempImage;
		for (int i = 0; i < 36; i = i + 3)
		{
			affineTransformation(pointsSrc[j], pointsDst[j], triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
		}
		morphedImage.copyTo(inputImage(*rects[j]));
	}
}

void  Morphing::AreaMorphs_Live::eyebrowMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 3;//min to do the operation
	int paddX = (int)((fldPoints[21 * 2] - fldPoints[17 * 2])*0.1);
	double slope = (double)(fldPoints[70 * 2 + 1] - fldPoints[68 * 2 + 1]) / (fldPoints[21 * 2] - fldPoints[68 * 2]);
	cv::Rect rectRight;
	rectRight.x = fldPoints[17 * 2] - paddX;
	rectRight.y = (int)(slope*(fldPoints[21 * 2] - fldPoints[68 * 2])) + fldPoints[68 * 2 + 1];
	rectRight.width = fldPoints[21 * 2] + paddX - rectRight.x + 1;
	int eyePointY = (std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (int)((std::min(fldPoints[37 * 2 + 1], fldPoints[38 * 2 + 1]) - (fldPoints[18 * 2 + 1] + fldPoints[19 * 2 + 1] + fldPoints[20 * 2 + 1]) / 3)*0.15)) - rectRight.y;
	const int length = 24;
	cv::Point pointsSrcRight[length];
	pointsSrcRight[0].x = fldPoints[17 * 2] - rectRight.x; pointsSrcRight[0].y = fldPoints[17 * 2 + 1] - rectRight.y;
	pointsSrcRight[1].x = fldPoints[18 * 2] - rectRight.x; pointsSrcRight[1].y = fldPoints[18 * 2 + 1] - rectRight.y;
	pointsSrcRight[2].x = fldPoints[19 * 2] - rectRight.x; pointsSrcRight[2].y = fldPoints[19 * 2 + 1] - rectRight.y;
	pointsSrcRight[3].x = fldPoints[20 * 2] - rectRight.x; pointsSrcRight[3].y = fldPoints[20 * 2 + 1] - rectRight.y;
	pointsSrcRight[4].x = fldPoints[21 * 2] - rectRight.x; pointsSrcRight[4].y = fldPoints[21 * 2 + 1] - rectRight.y;
	pointsSrcRight[5].x = pointsSrcRight[4].x;             pointsSrcRight[5].y = (eyePointY + pointsSrcRight[4].y) / 2;
	pointsSrcRight[6].x = pointsSrcRight[3].x;             pointsSrcRight[6].y = (eyePointY + pointsSrcRight[3].y) / 2;
	pointsSrcRight[7].x = pointsSrcRight[2].x;             pointsSrcRight[7].y = (eyePointY + pointsSrcRight[2].y) / 2;
	pointsSrcRight[8].x = pointsSrcRight[1].x;             pointsSrcRight[8].y = (eyePointY + pointsSrcRight[1].y) / 2;
	pointsSrcRight[9].x = pointsSrcRight[0].x;             pointsSrcRight[9].y = (eyePointY + pointsSrcRight[0].y) / 2;
	//support coords
	pointsSrcRight[10].x = 0;                   pointsSrcRight[10].y = pointsSrcRight[0].y;
	pointsSrcRight[11].x = pointsSrcRight[0].x; pointsSrcRight[11].y = (int)(slope*(fldPoints[17 * 2] - fldPoints[68 * 2])) + fldPoints[68 * 2 + 1] - rectRight.y;
	pointsSrcRight[12].x = pointsSrcRight[1].x; pointsSrcRight[12].y = (int)(slope*(fldPoints[18 * 2] - fldPoints[68 * 2])) + fldPoints[68 * 2 + 1] - rectRight.y;
	pointsSrcRight[13].x = pointsSrcRight[2].x; pointsSrcRight[13].y = (int)(slope*(fldPoints[19 * 2] - fldPoints[68 * 2])) + fldPoints[68 * 2 + 1] - rectRight.y;
	pointsSrcRight[14].x = pointsSrcRight[3].x; pointsSrcRight[14].y = (int)(slope*(fldPoints[20 * 2] - fldPoints[68 * 2])) + fldPoints[68 * 2 + 1] - rectRight.y;
	pointsSrcRight[15].x = pointsSrcRight[4].x; pointsSrcRight[15].y = 0;
	pointsSrcRight[16].x = rectRight.width - 1; pointsSrcRight[16].y = pointsSrcRight[4].y;
	pointsSrcRight[17].x = rectRight.width - 1; pointsSrcRight[17].y = pointsSrcRight[5].y;
	int morphSpaceRight = std::min(std::min(pointsSrcRight[0].y - pointsSrcRight[11].y, pointsSrcRight[1].y - pointsSrcRight[12].y), pointsSrcRight[2].y - pointsSrcRight[13].y) + 1;
	if (morphSpaceRight < thresh)
		return;
	int max = maxEyebrowRight;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(0.1*morphSpaceRight);
		if (max == 0)
			max = 1;
		maxEyebrowRight = max;
	}
	int min = -max;
	int offsetRight = (int)std::round(morphPercent*(max - min) + min);
	if (offsetRight == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetRight = -1;
		else
			offsetRight = 1;
	}

	int endy = std::max(pointsSrcRight[5].y, pointsSrcRight[9].y) + (2 * max);
	rectRight.height = (endy < eyePointY ? endy : eyePointY) + 1;
	if (checkBoundsForAffine(rectRight, inputImage.cols, inputImage.rows))
		return;
	pointsSrcRight[18].x = pointsSrcRight[4].x; pointsSrcRight[18].y = rectRight.height - 1;
	pointsSrcRight[19].x = pointsSrcRight[3].x; pointsSrcRight[19].y = rectRight.height - 1;
	pointsSrcRight[20].x = pointsSrcRight[2].x; pointsSrcRight[20].y = rectRight.height - 1;
	pointsSrcRight[21].x = pointsSrcRight[1].x; pointsSrcRight[21].y = rectRight.height - 1;
	pointsSrcRight[22].x = pointsSrcRight[0].x; pointsSrcRight[22].y = rectRight.height - 1;
	pointsSrcRight[23].x = 0;	                pointsSrcRight[23].y = pointsSrcRight[9].y;

	paddX = (int)((fldPoints[26 * 2] - fldPoints[22 * 2])*0.1);
	slope = (double)(fldPoints[72 * 2 + 1] - fldPoints[74 * 2 + 1]) / (fldPoints[22 * 2] - fldPoints[74 * 2]);
	cv::Rect rectLeft;
	rectLeft.x = fldPoints[22 * 2] - paddX;
	rectLeft.y = (int)(slope*(fldPoints[22 * 2] - fldPoints[74 * 2])) + fldPoints[74 * 2 + 1];
	rectLeft.width = fldPoints[26 * 2] + paddX - rectLeft.x + 1;
	eyePointY = (std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (int)((std::min(fldPoints[43 * 2 + 1], fldPoints[44 * 2 + 1]) - (fldPoints[23 * 2 + 1] + fldPoints[24 * 2 + 1] + fldPoints[25 * 2 + 1]) / 3)*0.15)) - rectLeft.y;
	cv::Point pointsSrcLeft[length];
	pointsSrcLeft[0].x = fldPoints[26 * 2] - rectLeft.x; pointsSrcLeft[0].y = fldPoints[26 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[1].x = fldPoints[25 * 2] - rectLeft.x; pointsSrcLeft[1].y = fldPoints[25 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[2].x = fldPoints[24 * 2] - rectLeft.x; pointsSrcLeft[2].y = fldPoints[24 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[3].x = fldPoints[23 * 2] - rectLeft.x; pointsSrcLeft[3].y = fldPoints[23 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[4].x = fldPoints[22 * 2] - rectLeft.x; pointsSrcLeft[4].y = fldPoints[22 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[5].x = pointsSrcLeft[4].x;             pointsSrcLeft[5].y = (eyePointY + pointsSrcLeft[4].y) / 2;
	pointsSrcLeft[6].x = pointsSrcLeft[3].x;             pointsSrcLeft[6].y = (eyePointY + pointsSrcLeft[3].y) / 2;
	pointsSrcLeft[7].x = pointsSrcLeft[2].x;             pointsSrcLeft[7].y = (eyePointY + pointsSrcLeft[2].y) / 2;
	pointsSrcLeft[8].x = pointsSrcLeft[1].x;             pointsSrcLeft[8].y = (eyePointY + pointsSrcLeft[1].y) / 2;
	pointsSrcLeft[9].x = pointsSrcLeft[0].x;             pointsSrcLeft[9].y = (eyePointY + pointsSrcLeft[0].y) / 2;
	//support coords
	pointsSrcLeft[10].x = rectLeft.width - 1; pointsSrcLeft[10].y = pointsSrcLeft[0].y;
	pointsSrcLeft[11].x = pointsSrcLeft[0].x; pointsSrcLeft[11].y = (int)(slope*(fldPoints[26 * 2] - fldPoints[74 * 2])) + fldPoints[74 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[12].x = pointsSrcLeft[1].x; pointsSrcLeft[12].y = (int)(slope*(fldPoints[25 * 2] - fldPoints[74 * 2])) + fldPoints[74 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[13].x = pointsSrcLeft[2].x; pointsSrcLeft[13].y = (int)(slope*(fldPoints[24 * 2] - fldPoints[74 * 2])) + fldPoints[74 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[14].x = pointsSrcLeft[3].x; pointsSrcLeft[14].y = (int)(slope*(fldPoints[23 * 2] - fldPoints[74 * 2])) + fldPoints[74 * 2 + 1] - rectLeft.y;
	pointsSrcLeft[15].x = pointsSrcLeft[4].x; pointsSrcLeft[15].y = 0;
	pointsSrcLeft[16].x = 0;                  pointsSrcLeft[16].y = pointsSrcLeft[4].y;
	pointsSrcLeft[17].x = 0;                  pointsSrcLeft[17].y = pointsSrcLeft[5].y;
	int morphSpaceLeft = std::min(std::min(pointsSrcLeft[0].y - pointsSrcLeft[11].y, pointsSrcLeft[1].y - pointsSrcLeft[12].y), pointsSrcLeft[2].y - pointsSrcLeft[13].y) + 1;
	if (morphSpaceLeft < thresh)
		return;
	max = maxEyebrowLeft;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(0.1*morphSpaceLeft);
		if (max == 0)
			max = 1;
		maxEyebrowLeft = max;
	}
	min = -max;
	int offsetLeft = (int)std::round(morphPercent*(max - min) + min);
	if (offsetLeft == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetLeft = -1;
		else
			offsetLeft = 1;
	}
	endy = std::max(pointsSrcLeft[5].y, pointsSrcLeft[9].y) + (2 * max);
	rectLeft.height = (endy < eyePointY ? endy : eyePointY) + 1;
	if (checkBoundsForAffine(rectLeft, inputImage.cols, inputImage.rows))
		return;
	pointsSrcLeft[18].x = pointsSrcLeft[4].x; pointsSrcLeft[18].y = rectLeft.height - 1;
	pointsSrcLeft[19].x = pointsSrcLeft[3].x; pointsSrcLeft[19].y = rectLeft.height - 1;
	pointsSrcLeft[20].x = pointsSrcLeft[2].x; pointsSrcLeft[20].y = rectLeft.height - 1;
	pointsSrcLeft[21].x = pointsSrcLeft[1].x; pointsSrcLeft[21].y = rectLeft.height - 1;
	pointsSrcLeft[22].x = pointsSrcLeft[0].x; pointsSrcLeft[22].y = rectLeft.height - 1;
	pointsSrcLeft[23].x = rectLeft.width - 1; pointsSrcLeft[23].y = pointsSrcLeft[9].y;

	cv::Point pointsDstRight[length], pointsDstLeft[length];
	for (int i = 0; i < length; i++)
	{
		pointsDstRight[i].x = pointsSrcRight[i].x; pointsDstRight[i].y = pointsSrcRight[i].y;
		pointsDstLeft[i].x = pointsSrcLeft[i].x; pointsDstLeft[i].y = pointsSrcLeft[i].y;
	}
	int triSets[] = { 0,10,11, 0,1,11, 1,11,12, 1,2,12, 2,12,13, 2,3,13, 3,13,14, 3,4,14, 4,14,15, 4,16,15, 4,16,17,
		4,5,17, 5,4,3, 5,6,3, 6,3,2, 6,7,2, 7,2,1, 7,8,1, 8,1,0, 8,9,0, 9,0,10, 9,23,10,
		9,23,22, 22,21,9, 21,9,8, 21,20,8, 20,8,7, 20,19,7, 19,7,6, 19,18,6, 18,6,5, 18,17,5 };

	//adding offsets to Y positions to make eyebrow lift/drop
	pointsDstRight[0].y -= offsetRight;
	pointsDstRight[1].y -= offsetRight;
	pointsDstRight[2].y -= offsetRight;
	pointsDstRight[3].y -= offsetRight;
	pointsDstRight[4].y -= offsetRight;
	pointsDstRight[5].y -= offsetRight;
	pointsDstRight[6].y -= offsetRight;
	pointsDstRight[7].y -= offsetRight;
	pointsDstRight[8].y -= offsetRight;
	pointsDstRight[9].y -= offsetRight;

	pointsDstLeft[0].y -= offsetLeft;
	pointsDstLeft[1].y -= offsetLeft;
	pointsDstLeft[2].y -= offsetLeft;
	pointsDstLeft[3].y -= offsetLeft;
	pointsDstLeft[4].y -= offsetLeft;
	pointsDstLeft[5].y -= offsetLeft;
	pointsDstLeft[6].y -= offsetLeft;
	pointsDstLeft[7].y -= offsetLeft;
	pointsDstLeft[8].y -= offsetLeft;
	pointsDstLeft[9].y -= offsetLeft;

	cv::Rect* rects[] = { &rectRight,&rectLeft };
	cv::Point* pointsSrc[] = { pointsSrcRight,pointsSrcLeft };
	cv::Point* pointsDst[] = { pointsDstRight,pointsDstLeft };
	for (int j = 0; j < 2; j++)
	{
		cv::Mat cropImage = inputImage(*rects[j]).clone();
		Utilities_Live::saveImage(cropImage, "input.png");
		cv::Mat morphedImage = cropImage.clone();
		cv::Mat maskImage, tempImage;
		for (int i = 0; i < 96; i = i + 3)
		{
			affineTransformation(pointsSrc[j], pointsDst[j], triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
		}
		morphedImage.copyTo(inputImage(*rects[j]));
	}
}

void  Morphing::AreaMorphs_Live::lipCornerMorph(cv::Mat& inputImage, const int fldPoints[], float morphPercent, bool drawTriangles)
{
	const int thresh = 0;//min to do the operation
	int morphSpaceRight = fldPoints[48 * 2 + 1] - fldPoints[49 * 2 + 1] + 1;
	int morphSpaceLeft = fldPoints[54 * 2 + 1] - fldPoints[53 * 2 + 1] + 1;
	int morphSpaceMin = (int)((fldPoints[54 * 2] - fldPoints[48 * 2] + 1)*0.075);
	if (morphSpaceRight < morphSpaceMin)
		morphSpaceRight = morphSpaceMin;
	if (morphSpaceLeft < morphSpaceMin)
		morphSpaceLeft = morphSpaceMin;
	if (morphSpaceRight < thresh || morphSpaceLeft < thresh)
		return;
	int max = maxLipCornerRight;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceRight*0.5f);
		if (max == 0)
			max = 1;
		maxLipCornerRight = max;
	}
	int min = -max;
	int offsetRight = (int)std::round(morphPercent*(max - min) + min);
	if (offsetRight == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetRight = -1;
		else
			offsetRight = 1;
	}
	cv::Rect rectRight;
	rectRight.x = fldPoints[48 * 2] - ((fldPoints[49 * 2] + fldPoints[59 * 2]) / 2 - fldPoints[48 * 2]);
	rectRight.y = std::min(fldPoints[48 * 2 + 1], fldPoints[49 * 2 + 1]) - (int)(1.5*max);
	rectRight.width = (fldPoints[49 * 2] + fldPoints[59 * 2]) / 2 - rectRight.x + 1;
	rectRight.height = fldPoints[59 * 2 + 1] + (int)(1.5*max) - rectRight.y + 1;
	if (checkBoundsForAffine(rectRight, inputImage.cols, inputImage.rows))
		return;
	const int length = 5;
	cv::Point pointsSrcRight[length];
	pointsSrcRight[0].x = fldPoints[48 * 2] - rectRight.x; pointsSrcRight[0].y = fldPoints[48 * 2 + 1] - rectRight.y;
	//support coords
	pointsSrcRight[1].x = 0;                   pointsSrcRight[1].y = 0;
	pointsSrcRight[2].x = rectRight.width - 1; pointsSrcRight[2].y = 0;
	pointsSrcRight[3].x = rectRight.width - 1; pointsSrcRight[3].y = rectRight.height - 1;
	pointsSrcRight[4].x = 0;                   pointsSrcRight[4].y = rectRight.height - 1;

	max = maxLipCornerLeft;
	if (!Utilities_Live::isMorphStable)
	{
		max = (int)std::round(morphSpaceLeft*0.5f);
		if (max == 0)
			max = 1;
		maxLipCornerLeft = max;
	}
	min = -max;
	int offsetLeft = (int)std::round(morphPercent*(max - min) + min);
	if (offsetLeft == 0 && morphPercent != 0.5)
	{
		if (morphPercent < 0.5)
			offsetLeft = -1;
		else
			offsetLeft = 1;
	}
	cv::Rect rectLeft;
	rectLeft.x = (fldPoints[53 * 2] + fldPoints[55 * 2]) / 2;
	rectLeft.y = std::min(fldPoints[53 * 2 + 1], fldPoints[54 * 2 + 1]) - (int)(1.5*max);
	rectLeft.width = fldPoints[54 * 2] + (fldPoints[54 * 2] - (fldPoints[53 * 2] + fldPoints[55 * 2]) / 2) - rectLeft.x + 1;
	rectLeft.height = fldPoints[55 * 2 + 1] + (int)(1.5*max) - rectLeft.y + 1;
	if (checkBoundsForAffine(rectLeft, inputImage.cols, inputImage.rows))
		return;
	cv::Point pointsSrcLeft[length];
	pointsSrcLeft[0].x = fldPoints[54 * 2] - rectLeft.x; pointsSrcLeft[0].y = fldPoints[54 * 2 + 1] - rectLeft.y;
	//support coords
	pointsSrcLeft[1].x = rectLeft.width - 1; pointsSrcLeft[1].y = 0;
	pointsSrcLeft[2].x = 0;                  pointsSrcLeft[2].y = 0;
	pointsSrcLeft[3].x = 0;                  pointsSrcLeft[3].y = rectLeft.height - 1;
	pointsSrcLeft[4].x = rectLeft.width - 1; pointsSrcLeft[4].y = rectLeft.height - 1;

	cv::Point pointsDstRight[length], pointsDstLeft[length];
	for (int i = 0; i < length; i++)
	{
		pointsDstRight[i].x = pointsSrcRight[i].x; pointsDstRight[i].y = pointsSrcRight[i].y;
		pointsDstLeft[i].x = pointsSrcLeft[i].x; pointsDstLeft[i].y = pointsSrcLeft[i].y;
	}
	int triSets[] = { 0,1,2, 0,2,3, 0,3,4, 0,4,1 };

	//adding offsets to Y positions to make lipcorner rising/sagging
	pointsDstRight[0].y -= offsetRight;
	pointsDstLeft[0].y -= offsetLeft;

	cv::Rect* rects[] = { &rectRight,&rectLeft };
	cv::Point* pointsSrc[] = { pointsSrcRight,pointsSrcLeft };
	cv::Point* pointsDst[] = { pointsDstRight,pointsDstLeft };
	for (int j = 0; j < 2; j++)
	{
		cv::Mat cropImage = inputImage(*rects[j]).clone();
		Utilities_Live::saveImage(cropImage, "input.png");
		cv::Mat morphedImage = cropImage.clone();
		cv::Mat maskImage, tempImage;
		for (int i = 0; i < 12; i = i + 3)
		{
			affineTransformation(pointsSrc[j], pointsDst[j], triSets, i, cropImage, maskImage, tempImage, morphedImage, drawTriangles);
		}
		morphedImage.copyTo(inputImage(*rects[j]));
	}
}

