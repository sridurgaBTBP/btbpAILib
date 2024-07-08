#include "ImageTransforms_Live.h"
#include "Utilities_Live.h"
#include <opencv2/imgproc.hpp>

void Morphing::ImageTransforms_Live::affineTransformation(cv::Point* pointsSrc, cv::Point* pointsDst, int* triSets, int index, const cv::Mat& cropImage, cv::Mat& maskImage, cv::Mat& tempImage, cv::Mat& morphedImage, bool drawTriangles)
{
	std::vector<std::vector<cv::Point>> pts(1);
	pts[0].push_back(pointsSrc[triSets[index]]);
	pts[0].push_back(pointsSrc[triSets[index + 1]]);
	pts[0].push_back(pointsSrc[triSets[index + 2]]);
	maskImage = cv::Mat::zeros(cropImage.rows, cropImage.cols, CV_8UC1);
	cv::drawContours(maskImage, pts, 0, cv::Scalar(255), cv::FILLED);
#ifdef  TESTING
	if (helperImage.rows != cropImage.rows || helperImage.cols != cropImage.cols)
		helperImage = cv::Mat::zeros(cropImage.rows, cropImage.cols, CV_8UC1);
	cv::drawContours(helperImage, pts, 0, cv::Scalar(255), CV_FILLED);
	Utilities_Live::saveImage(helperImage, "processVerification.png");
#endif //TESTING
	tempImage = cv::Mat::zeros(cropImage.rows, cropImage.cols, cropImage.type());
	cropImage.copyTo(tempImage, maskImage);
	Utilities_Live::saveImage(tempImage, "triangle.png");
	cv::Point2f src[3] = { pts[0][0], pts[0][1], pts[0][2] };
	cv::Point2f dst[3] = { pointsDst[triSets[index]],pointsDst[triSets[index + 1]], pointsDst[triSets[index + 2]] };
	cv::Mat transformationMatrix = cv::getAffineTransform(src, dst);
	if (cv::countNonZero(transformationMatrix) > 0)
	{
		cv::warpAffine(tempImage, tempImage, transformationMatrix, tempImage.size(), cv::INTER_NEAREST);
		tempImage.copyTo(morphedImage, tempImage);
		if (drawTriangles)
		{
			pts[0][0] = dst[0];
			pts[0][1] = dst[1];
			pts[0][2] = dst[2];
			cv::drawContours(morphedImage, pts, 0, cv::Scalar(0, 160, 255), 1);
		}
	}
	Utilities_Live::saveImage(tempImage, "triangleTrasformed.png");
	Utilities_Live::saveImage(morphedImage, "morphed.png");
}

std::vector<Morphing::ObjectInfo> Morphing::ImageTransforms_Live::affineTransformation(const cv::Point2f* pointsSrc, const cv::Point2f* pointsDst, const int* triSets, int triSetsLength, std::vector<ObjectInfo> objectsInfo)
{
	for (int i = 0; i < triSetsLength; i = i + 3)
	{
		std::vector<cv::Point2f> pts;
		pts.push_back(pointsSrc[triSets[i]]);
		pts.push_back(pointsSrc[triSets[i + 1]]);
		pts.push_back(pointsSrc[triSets[i + 2]]);
		int n = (int)objectsInfo.size();
		for (int j = 0; j < n; j++)
		{
			ObjectInfo& polyInfo = objectsInfo[j];
			if (!polyInfo.isApplied && cv::pointPolygonTest(pts, polyInfo.centroid, false) >= 0)
			{
				cv::Point2f src[3] = { pts[0], pts[1], pts[2] };
				cv::Point2f dst[3] = { pointsDst[triSets[i]],pointsDst[triSets[i + 1]], pointsDst[triSets[i + 2]] };
				cv::Mat transformationMatrix = cv::getAffineTransform(src, dst);
				double* matrixPtr = (double*)transformationMatrix.data;
				std::vector<cv::Point2f>& points = polyInfo.polygon;
				int n1 = (int)points.size();
				cv::Point2f temp;
				for (int k = 0; k < n1; k++)
				{
					temp.x = (float)(matrixPtr[0] * points[k].x + matrixPtr[1] * points[k].y + matrixPtr[2]);
					temp.y = (float)(matrixPtr[3] * points[k].x + matrixPtr[4] * points[k].y + matrixPtr[5]);
					points[k] = temp;
				}
				polyInfo.isApplied = true;
			}
		}
	}
	return objectsInfo;
}

bool Morphing::ImageTransforms_Live::checkBoundsForAffine(cv::Rect& rect, int cols, int rows)
{
	if (rect.x < 0 || rect.width <= 0 || rect.x + rect.width > cols
		|| rect.y < 0 || rect.height <= 0 || rect.y + rect.height > rows)
		return true;
	else
		return false;
}

bool Morphing::ImageTransforms_Live::checkIQCForAffine(const int fldPoints[], float thresholds[])
{
	//head rotation
	double angle = std::atan2(fldPoints[45 * 2 + 1] - fldPoints[36 * 2 + 1], fldPoints[45 * 2] - fldPoints[36 * 2])*(180 / CV_PI);
	if (std::abs(angle) > thresholds[0])
		return false;

	//Face up and down
	int verticalTilt = (fldPoints[36 * 2 + 1] + fldPoints[45 * 2 + 1]) / 2 - (fldPoints[0 * 2 + 1] + fldPoints[16 * 2 + 1]) / 2;
	if (std::abs(verticalTilt) > thresholds[1] * (fldPoints[8 * 2 + 1] - fldPoints[27 * 2 + 1]))
		return false;

	//profile rotation
	int leftSideDistance = 0, rightSideDistance = 0;
	leftSideDistance += fldPoints[30 * 2] - fldPoints[2 * 2];
	leftSideDistance += fldPoints[33 * 2] - fldPoints[3 * 2];
	rightSideDistance += fldPoints[14 * 2] - fldPoints[30 * 2];
	rightSideDistance += fldPoints[13 * 2] - fldPoints[33 * 2];
	if (std::abs(rightSideDistance - leftSideDistance) > thresholds[2] * (leftSideDistance + rightSideDistance))
		return false;

	return true;
}

bool Morphing::ImageTransforms_Live::checkIQCForAffine(const float fldPoints[], float thresholds[])
{
	//head rotation
	double angle = std::atan2(fldPoints[45 * 2 + 1] - fldPoints[36 * 2 + 1], fldPoints[45 * 2] - fldPoints[36 * 2])*(180 / CV_PI);
	if (std::abs(angle) > thresholds[0])
		return false;

	//Face up and down
	float verticalTilt = (fldPoints[36 * 2 + 1] + fldPoints[45 * 2 + 1]) / 2 - (fldPoints[0 * 2 + 1] + fldPoints[16 * 2 + 1]) / 2;
	if (std::abs(verticalTilt) > thresholds[1] * (fldPoints[8 * 2 + 1] - fldPoints[27 * 2 + 1]))
		return false;

	//profile rotation
	float leftSideDistance = 0, rightSideDistance = 0;
	leftSideDistance += fldPoints[30 * 2] - fldPoints[2 * 2];
	leftSideDistance += fldPoints[33 * 2] - fldPoints[3 * 2];
	rightSideDistance += fldPoints[14 * 2] - fldPoints[30 * 2];
	rightSideDistance += fldPoints[13 * 2] - fldPoints[33 * 2];
	if (std::abs(rightSideDistance - leftSideDistance) > thresholds[2] * (leftSideDistance + rightSideDistance))
		return false;

	return true;
}
