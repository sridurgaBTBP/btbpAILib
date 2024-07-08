//
//  Morphing_Live.h
//  Morphing_Live
//
//  Created by Mac5Dev1 on 17/03/18.
//  Copyright © 2018 Mac5Dev1. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import <CoreMedia/CoreMedia.h>
#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
#endif

@interface Morphing_Live : NSObject
{
    
}

@property (nonatomic) NSString* tagName;
@property (assign) BOOL isfldinitialized;

//landmark live variables
@property (nonatomic) bool isBRFAutocapture;
@property (nonatomic) double stabilityFactor;
@property (nonatomic) int Profile;//1-F,2-L,3-R
@property (nonatomic) double ProfileDeterminant;//1-F,2-L,3-R

//morphing feature variables
@property (nonatomic) bool isBlink;
@property (nonatomic) bool isLandmarks;
@property (nonatomic) bool isMask;
@property (nonatomic) bool isDistortion;
@property (nonatomic) bool isLipMorphing;
@property (nonatomic) bool isLipCornerMorphing;
@property (nonatomic) bool isEyeMorphing;
@property (nonatomic) bool isEyebrowMorphing;
@property (nonatomic) bool isNoseMorphing;
@property (nonatomic) bool isJawMorphing;
@property (nonatomic) bool isEyebrowShapeMorphing;

//morphing intensity variables
@property (nonatomic) float lipMorphingIntensity;
@property (nonatomic) float lipCornerMorphingIntensity;
@property (nonatomic) float eyeMorphingIntensity;
@property (nonatomic) float eyebrowMorphingIntensity;
@property (nonatomic) float noseMorphingIntensity;
@property (nonatomic) float jawMorphingIntensity;
@property (nonatomic) float eyebrowshapeMorphingIntensity;
@property (nonatomic) float eyebrowshapeIndex;

//skincare feature variables
@property (nonatomic) bool isComplexion;
@property (nonatomic) bool isForeheadComplexion;
@property (nonatomic) bool isUnderEyeComplexion;
@property (nonatomic) bool isNasolabialComplexion;
@property (nonatomic) bool isLipHealth;

//skincare intensity variables
@property (nonatomic) float complexionIntensity;
@property (nonatomic) float foreheadComplexionIntensity;
@property (nonatomic) float underEyeComplexionIntensity;
@property (nonatomic) float nasolabialComplexionIntensity;
@property (nonatomic) float lipHealthIntensity;

@property (nonatomic) int foundationColorNumber;

//private
-(void) assigningLandmarkvariales;
#ifdef __cplusplus
-(bool) performMorphingCommon:(cv::Mat&) inputImage arg1:(int[]) fldPoints;
#endif

//public methods
-(void) initializingFLDMorphing:(int) imageWidth arg:(int)imageHeight;
-(void) n_calculatingFPS;
-(void) setFoundationColors:(NSMutableArray*)colors;
-(bool) performMorphingLive:(CMSampleBufferRef)sampleBuffer;
-(UIImage*) performMorphingStill;
-(bool) initializingstillmat:(NSString*) captureImagePath;
-(void) releaseMat;

-(void) initializingBRFLive:(int) imageWidth arg:(int)imageHeight;


//setting and getting brfFace initialization
-(void) setBRFInitialization:(bool) set;
-(bool) getBRFInitialization;



@end
