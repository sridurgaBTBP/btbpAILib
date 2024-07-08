#include "HairFeatures.h"
#include "Utilities_Live.h"
#include <dlib/clustering.h>
#include <dlib/rand.h>
#include <iostream>
#include <opencv2/imgproc.hpp>

void Skincare::HairFeatures::kkmeans_ex()
{
	// Here we declare that our samples will be 2 dimensional column vectors.  
	// (Note that if you don't know the dimensionality of your vectors at compile time
	// you can change the 2 to a 0 and then set the size at runtime)
	typedef dlib::matrix<double, 2, 1> sample_type;

	// Now we are making a typedef for the kind of kernel we want to use.  I picked the
	// radial basis kernel because it only has one parameter and generally gives good
	// results without much fiddling.
	typedef dlib::radial_basis_kernel<sample_type> kernel_type;


	// Here we declare an instance of the kcentroid object.  It is the object used to 
	// represent each of the centers used for clustering.  The kcentroid has 3 parameters 
	// you need to set.  The first argument to the constructor is the kernel we wish to 
	// use.  The second is a parameter that determines the numerical accuracy with which 
	// the object will perform part of the learning algorithm.  Generally, smaller values 
	// give better results but cause the algorithm to attempt to use more dictionary vectors 
	// (and thus run slower and use more memory).  The third argument, however, is the 
	// maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
	// it to control the runtime complexity.  
	dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 8);

	// Now we make an instance of the kkmeans object and tell it to use kcentroid objects
	// that are configured with the parameters from the kc object we defined above.
	dlib::kkmeans<kernel_type> test(kc);

	std::vector<sample_type> samples;
	std::vector<sample_type> initial_centers;

	sample_type m;

	dlib::rand rnd;

	// we will make 50 points from each class
	const long num = 50;

	// make some samples near the origin
	double radius = 0.5;
	for (long i = 0; i < num; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// add this sample to our set of samples we will run k-means 
		samples.push_back(m);
	}

	// make some samples in a circle around the origin but far away
	radius = 10.0;
	for (long i = 0; i < num; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// add this sample to our set of samples we will run k-means 
		samples.push_back(m);
	}

	// make some samples in a circle around the point (25,25) 
	radius = 4.0;
	for (long i = 0; i < num; ++i)
	{
		double sign = 1;
		if (rnd.get_random_double() < 0.5)
			sign = -1;
		m(0) = 2 * radius*rnd.get_random_double() - radius;
		m(1) = sign*sqrt(radius*radius - m(0)*m(0));

		// translate this point away from the origin
		m(0) += 25;
		m(1) += 25;

		// add this sample to our set of samples we will run k-means 
		samples.push_back(m);
	}

	// tell the kkmeans object we made that we want to run k-means with k set to 3. 
	// (i.e. we want 3 clusters)
	test.set_number_of_centers(3);

	// You need to pick some initial centers for the k-means algorithm.  So here
	// we will use the dlib::pick_initial_centers() function which tries to find
	// n points that are far apart (basically).  
	pick_initial_centers(3, initial_centers, samples, test.get_kernel());

	// now run the k-means algorithm on our set of samples.  
	test.train(samples, initial_centers);

	// now loop over all our samples and print out their predicted class.  In this example
	// all points are correctly identified.
	for (unsigned long i = 0; i < samples.size() / 3; ++i)
	{
		std::cout << test(samples[i]) << " ";
		std::cout << test(samples[i + num]) << " ";
		std::cout << test(samples[i + 2 * num]) << "\n";
	}

	// Now print out how many dictionary vectors each center used.  Note that 
	// the maximum number of 8 was reached.  If you went back to the kcentroid 
	// constructor and changed the 8 to some bigger number you would see that these
	// numbers would go up.  However, 8 is all we need to correctly cluster this dataset.
	std::cout << "num dictionary vectors for center 0: " << test.get_kcentroid(0).dictionary_size() << std::endl;
	std::cout << "num dictionary vectors for center 1: " << test.get_kcentroid(1).dictionary_size() << std::endl;
	std::cout << "num dictionary vectors for center 2: " << test.get_kcentroid(2).dictionary_size() << std::endl;

	//Utilities_Live::saveImage()
}

void Skincare::HairFeatures::kkmeans(cv::Mat& inputImage)
{
	typedef dlib::matrix<double, 3, 1> sample_type;
	typedef dlib::radial_basis_kernel<sample_type> kernel_type;
	dlib::kcentroid<kernel_type> kc(kernel_type(0.1), 0.01, 100);
	dlib::kkmeans<kernel_type> test(kc);

	std::vector<sample_type> samples;
	std::vector<sample_type> initial_centers;

	sample_type m;

	uchar*  inputPtr = inputImage.data;
	int inputStep = (int)inputImage.step;
	int length = inputImage.cols * 3;
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < length; j += 3)
		{
			m(0) = inputPtr[j];
			m(1) = inputPtr[j + 1];
			m(2) = inputPtr[j + 2];
			samples.push_back(m);
		}
		inputPtr += inputStep;
	}
	test.set_number_of_centers(3);
	pick_initial_centers(3, initial_centers, samples, test.get_kernel());
	test.train(samples, initial_centers);

	inputPtr = inputImage.data;
	int k = -1;
	for (int i = 0; i < inputImage.rows; i++)
	{
		for (int j = 0; j < length; j += 3)
		{
			unsigned long v = test(samples[++k]);
			inputPtr[j] = (uchar)(v * 80);
			inputPtr[j + 1] = (uchar)(v * 80);
			inputPtr[j + 2] = (uchar)(v * 80);
		}
		inputPtr += inputStep;
	}
	std::cout << "num dictionary vectors for center 0: " << test.get_kcentroid(0).dictionary_size() << std::endl;
	std::cout << "num dictionary vectors for center 1: " << test.get_kcentroid(1).dictionary_size() << std::endl;
	std::cout << "num dictionary vectors for center 2: " << test.get_kcentroid(2).dictionary_size() << std::endl;
}

void Skincare::HairFeatures::hairColor(cv::Mat& inputImage, cv::Mat& maskImage224, int foreheadPos, int color[], float coverage)
{
	cv::Mat maskContoursImage = maskImage224.clone();
	std::vector<std::vector<cv::Point>> regions;
    cv::findContours(maskContoursImage, regions, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	double regionArea, maxRegionArea = 0, maxRegionIndex = 0;
	for (int i = 0; i < regions.size(); i++)
	{
		regionArea = cv::contourArea(regions[i]);
		if (regionArea > maxRegionArea)
		{
			maxRegionArea = regionArea;
			maxRegionIndex = i;
		}
	}
	int minX = maskImage224.cols, maxX = 0, minY = maskImage224.rows, maxY = 0;
	for (int i = 0; i < regions.size(); i++)//removing all the objects except max object
	{
		cv::Rect regionRect = cv::boundingRect(regions[i]);
		if (cv::contourArea(regions[i]) < (0.1*maxRegionArea) || regionRect.y > foreheadPos)
		{
			cv::drawContours(maskImage224, regions, i, cv::Scalar(0), cv::FILLED);
		}
		else
		{
			if (regionRect.x < minX)
				minX = regionRect.x;
			if (regionRect.y < minY)
				minY = regionRect.y;
			if (regionRect.x + regionRect.width - 1 > maxX)
				maxX = regionRect.x + regionRect.width - 1;
			if (regionRect.y + regionRect.height - 1 > maxY)
				maxY = regionRect.y + regionRect.height - 1;

		}
	}
	cv::Rect rect = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
	int	kernalHair = (int)(rect.width*0.04* coverage);
	if (kernalHair % 2 == 0)
		kernalHair++;
	if (kernalHair < 3)
		kernalHair = 3;
	if (!Utilities_Live::rectanglePadding(rect, kernalHair, maskImage224.cols, maskImage224.rows))
		return;
	float ratio0, ratio1, ratio2;
	int maxChannel;
	if (color[0] == 0 && color[1] == 0 && color[2] == 0)//black color
	{
		ratio0 = ratio1 = ratio2 = 0;
		maxChannel = 0;//can be anything
	}
	else
	{
		if (color[0] >= color[1] && color[0] >= color[2])
		{
			ratio0 = 1;
			ratio1 = (float)color[1] / color[0];
			ratio2 = (float)color[2] / color[0];
			maxChannel = 0;
		}
		else
		{
			if (color[1] >= color[2])
			{
				ratio0 = (float)color[0] / color[1];
				ratio1 = 1;
				ratio2 = (float)color[2] / color[1];
				maxChannel = 1;
			}
			else
			{
				ratio0 = (float)color[0] / color[2];
				ratio1 = (float)color[1] / color[2];
				ratio2 = 1;
				maxChannel = 2;
			}
		}
	}
	float resizeFactor = (float)inputImage.cols / maskImage224.cols;
	cv::Rect fullResRect((int)(rect.x*resizeFactor), (int)(rect.y*resizeFactor), (int)(rect.width*resizeFactor), (int)(rect.height*resizeFactor));
	cv::Mat inputROIImage = inputImage(fullResRect);
	cv::Mat maskROIImage = maskImage224(rect);
	Utilities_Live::saveImage(maskROIImage, "maskbefore.png");
	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(kernalHair, kernalHair));
	cv::erode(maskROIImage, maskROIImage, element, cv::Point(kernalHair/2, 0), 1);
	Utilities_Live::saveImage(maskROIImage, "maskeroded.png");
	cv::Mat inputROIresizeImage;
	cv::resize(inputROIImage, inputROIresizeImage, cv::Size(maskROIImage.cols, maskROIImage.rows));
	int moustache = inputROIresizeImage.rows;
	float RGB[3] = {};
	Utilities_Live::averageRGB(inputROIresizeImage, maskROIImage, moustache, RGB, 0.6f, 0.1f, "SkinLab");
	Utilities_Live::saveImage(inputROIresizeImage, "hairbefore.png");
	float intensityFactor = (color[maxChannel] / RGB[maxChannel]) - 1;
	intensityFactor *= 0.5;
	if (intensityFactor > 1)
		intensityFactor = 1;
	cv::blur(maskROIImage, maskROIImage, cv::Size(kernalHair, kernalHair));
	Utilities_Live::saveImage(maskROIImage, "mask.png");
	cv::resize(maskROIImage, maskROIImage, cv::Size(inputROIImage.cols, inputROIImage.rows));
	Utilities_Live::saveImage(maskROIImage, "maskresized.png");

	uchar* inputPtr = inputROIImage.data;
	const uchar* maskPtr = maskROIImage.data;
	int inputStep = (int)inputROIImage.step;
	int maskStep = (int)maskROIImage.step;

	//Simulate hair color 
	float  R, G, B, P, sk;
	float resValue = (1.f / 255)*coverage;
	int length = inputROIImage.cols * 3;
	float I = 1 + intensityFactor;
	for (int i = 0; i < inputROIImage.rows; i++)
	{
		int k = -1;
		for (int j = 0; j < length; j += 3)
		{
			sk = maskPtr[++k] * resValue;
			if (sk > 0)
			{
				R = inputPtr[j];
				G = inputPtr[j + 1];
				B = inputPtr[j + 2];
				P = inputPtr[j + maxChannel];
				R = R * (1 - sk) + P * ratio0*I  * sk;
				G = G * (1 - sk) + P * ratio1*I  * sk;
				B = B * (1 - sk) + P * ratio2*I  * sk;
				if (R > 200)
					R = 200;
				if (G > 200)
					G = 200;
				if (B > 200)
					B = 200;
				inputPtr[j] = (uchar)R;
				inputPtr[j + 1] = (uchar)G;
				inputPtr[j + 2] = (uchar)B;
			}
		}
		maskPtr += maskStep;
		inputPtr += inputStep;
	}
}
