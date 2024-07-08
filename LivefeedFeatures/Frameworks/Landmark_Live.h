//
//  Landmark_Live.h
//  Landmark_Live
//
//  Created by Mac5Dev1 on 09/03/18.
//  Copyright © 2018 Mac5Dev1. All rights reserved.
//
#import <opencv2/opencv.hpp>
#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>
#import <UIKit/UIKit.h>
#ifdef __cplusplus
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <stdio.h>
#include <stdint.h>

#include <string>
#include <fstream>
#include <thread>
#import <mach/mach_time.h>
#endif

@interface Landmark_Live : NSObject
{
    
}

//Internal varibales
@property (assign) BOOL isfldinitialized;

//public variables need to set from outside
@property (nonatomic) NSString* tagName;//for printing messages
@property (nonatomic) int fps;//For FPS calculation syncing with main library

@property (nonatomic) BOOL stillImageInitialization;//To set BRF detection on still image

//
//settings from outside
@property (nonatomic) bool useBRF;//To use brf fld
@property (nonatomic) double ProfileDeterminant;//1-F,2-L,3-R //To set profile determinant percentage
@property (nonatomic) double stabilityFactor;//To set stability factor
@property (nonatomic) int Profile;//1-F,2-L,3-R--//To set profile
@property (nonatomic) bool isFaceGuideDisplayed;//Used only for clinical app and in default it will be false..
//
//

@property (nonatomic) bool isFrontCapture;//To set front/back camera for previous rotation
@property (nonatomic) bool drawLandMarks;
@property (nonatomic) bool DlibStability;

@property (nonatomic) UIImage* LandMarksImage;
@property (nonatomic) UIImage* FaceRecogChipImage;

//Public variables return to outside
@property (nonatomic) int StatusCode;//for IQC checking

//private
-(void) LoadModel:(NSString*)trainedModelPath;
-(void) n_calculatingFPS;//For direct Auto capture

//public direct Auto capture methods
//Have to initialize 
-(int)performIQConCapture:(NSString*)imgInFile IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals rotation:(bool)isRotationNeeded;
-(int)performIQConUIImage:(UIImage*)imageUI IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals rotation:(bool)isRotationNeeded;
-(int)performExpressiononCapture:(CMSampleBufferRef)sampleBuffer IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals;
-(int)performExpressiononUIImage:(UIImage*)sampleBuffer IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals;
//

-(int)performAutocapture:(CMSampleBufferRef)sampleBuffer IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals;

-(void) setFacerecognormalization:(bool) isNeeded;//for face recog chip generation
-(void) initializingFLD:(int) imageWidth arg:(int)imageHeight;
-(void) initializingFLDwithPath:(int) imageWidth arg:(int)imageHeight dlibDatFilePath:(NSString*)dlibDatFilePath;

//public setting attributes
-(void) n_GChannelLipsNeeded:(bool) isNeeded;//for makeup features

#ifdef __cplusplus
-(void)Arraytovector:(int*) LandmarkPoints arg1:(int) ArraySize arg2:(std::vector<cv::Point>&) FLDPoints;
-(void) VectortoArray:(std::vector<cv::Point>&) FRDPoints arg1:(int*) LandmarkPoints;
-(void) initializingBRF:(int) imageWidth arg:(int)imageHeight;//for Makeup and morphing
-(void) performBRFLandmarkDetection:(uint8_t*) imageData arg1:(int[]) fldPoints;
-(NSString *) n_JoinLandMarks:(cv::Mat &)inputFrame j_LandMarks:(int*)landMarkPoints para1:(bool)isBoundsChecked para2:(int *)rectCoordinates para3:(double) Value ;

-(void) n_FetchFaceRegions:(cv::Mat &)liveImage j_LandMarks:(int*)landMarkPoints para3:(double) ProfileDeterminamt;

-(void) performRotation:(cv::Mat&)input arg1:(cv::Mat&)output arg2:(int)rotationAngle arg3:(int) cameraId;

//public splitted auto capture methods
-(void) landmarkdetectiononcamerabuffer:(CMSampleBufferRef)sampleBuffer arg1:(cv::Mat&) resizeImage;
-(void) landmarkdetectiononUIimage:(UIImage*)uiImage arg1:(cv::Mat&) resizeImage;
//resizeImage for landmark detection
//rects for obtaining IOS rect points
//width and height for IOS rect points
//feature resize factor for IOS rect points
//base buffer is a pointer used for brf points detection
-(void) detectingLandmarks:(cv::Mat&)resizeImage;
-(int) profiledeterminant;
-(int) performIQConMat:(cv::Mat&)resizeImage IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals eyeCheck:(int)eyeCheckStatus;

-(int) performIQConMatforEye:(cv::Mat&)resizeImage IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals eyeCheck:(int)eyeCheckStatus disableEyeStatus:(bool)disableEyeStatus disableLightIQC:(bool)disableLightIQC;
-(int) OpenorClosedEyeIQCCheck:(cv::Mat&)resizeImage;
-(void) placingmatontocamerabuffer:(CMSampleBufferRef)sampleBuffer arg1:(cv::Mat&) resizeImage;

//public getting attributes
-(void) copinglandmarks:(int [])fldPoints;
-(void) copingFloatlandmarks:(float [])fldPoints;
-(void) copingrectcoordinates:(int[])coordinates;
-(std::vector<std::vector<cv::Point>>) copingregions;
-(bool) copingstablity:(bool) isFacerecog;

//releasing
-(void) n_releaseMemory:(int) mode;

//resizing
-(void) imageResizing:(NSString*) ImagePath;
-(UIImage*) ResizeUIImage:(UIImage*)UIimage resizeto:(double)resolutioninMP;

//setting and getting brfFace initialization
-(void) setBRFInitialization:(bool) set;
-(bool) getBRFInitialization;
/*Mouth Expressions*/
-(int) PerformMouthExpression:(cv::Mat&)Image j_LandMarks:(std::vector<cv::Point>&) FLDPoints;

/**
 ConvertUIImagetocvMat args description

 @param imageUI captured image in UIImage format
 @return BGR formate mat
 */
-(cv::Mat) ConvertUIImagetocvMat:(UIImage*) imageUI;
/**
 ConvertbufferUIImagetocvMat args description

 @param sampleBufferUI frame buffer in UIImage format
 @return BGR formate mat
 */
-(cv::Mat) ConvertbufferUIImagetocvMat:(UIImage*)sampleBufferUI;

/**
 ConvertMattoUIImage args description

 @return UIImage from open cv mat
 */
-(UIImage*) ConvertMattoUIImage:(cv::Mat) cvMat;
#endif

@end
