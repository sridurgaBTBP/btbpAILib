#ifndef SKINCARE_FACIALREGIONS_LIVE_H
#define SKINCARE_FACIALREGIONS_LIVE_H

#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <fstream>
#include <thread>
#include <iostream>

namespace Skincare
{
	class FacialRegions_Live
	{
		/*variables*/
	public:
		int totalRegionPoints = 0;

		/*methods*/
	public:
		std::vector<std::pair<std::string, std::vector<cv::Point>>> facialZones_12Old(cv::Mat rgbImage, std::vector<cv::Point> FLDpoints, std::string dir);
		std::vector<std::pair<std::string, std::vector<cv::Point>>> facialZones(std::vector<cv::Point> FLDpoints, int no_of_regions);
	private:
		cv::Point line_intersection_intercept(cv::Point A, cv::Point B, cv::Point C, double m2);
		void getExtendedPoints(std::vector<cv::Point>& dataPoints, std::vector<cv::Point>& extendedPoints, std::vector<int> skipPoints = std::vector<int>(), bool isClockwise=true);
	};
}
#endif // SKINCARE_FACIALREGIONS_LIVE_H

