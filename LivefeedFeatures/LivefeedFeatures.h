//
//  Makeupfeatures_Live.h
//  Makeupfeatures_Live
//
//  Created by Mac5Dev1 on 13/03/18.
//  Copyright © 2018 Mac5Dev1. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <CoreMedia/CoreMedia.h>
#import <UIKit/UIKit.h>
#ifdef __cplusplus
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <fstream>
#include <thread>
#endif


@interface LivefeedFeatures : NSObject
{
    
}

@property (nonatomic) NSMutableArray * landMarkPointsonFullRezFrame;
/**
 facial4RegionPoints
 format :
 foreheadregion  = facial4RegionPoints[0];
 CheekLeft=facial4RegionPoints[1];
 CheekRight=facial4RegionPoints[2];
 Chin=facial4RegionPoints[3];
 */
@property (nonatomic) NSMutableArray * facial4RegionPoints; //0->FH,1->CheekLeft,2->CheekRight,3->Chin


/**
 an array of x1,y1,x2,y2,x3,y3.....etc - to draw point on the view
 */
@property (nonatomic) NSMutableArray* drawingPoints;

@property (nonatomic) NSString* tagName;
@property (assign) BOOL isfldinitialized;

//landmark live variables
@property (nonatomic) bool useBRF;
@property (nonatomic) double stabilityFactor;
@property (nonatomic) int Profile;//1-F,2-L,3-R
@property (nonatomic) double ProfileDeterminant;//1-F,2-L,3-R
@property (nonatomic) bool isFrontCapture;
//Make up variables
@property (nonatomic) int StatusCode;
//feature triggering variables
@property (nonatomic) bool isFeatureSwitch;
@property (nonatomic) bool isAutoCapture;
@property (nonatomic) bool isLandmarksonFace;
@property (nonatomic) bool isLandmarksFetching;
@property (nonatomic) bool is7regionsFetching;
@property (nonatomic) bool isFaceBounds;
@property (nonatomic) bool isEVDamage;
@property (nonatomic) bool isBlochiness;
@property (nonatomic) bool isHighSkinPigmentation;
@property (nonatomic) bool isAverageContrast;
@property (nonatomic) bool isTeethWhitening;
@property (nonatomic) bool isFairness;
@property (nonatomic) bool isRednesslive;
@property (nonatomic) bool isBlackBG;
@property (nonatomic) bool isGrayImage;
@property (nonatomic) bool isNoGreenROI;
@property (nonatomic) bool isFaceShape;
@property (nonatomic) bool isHairFrizz;
@property (nonatomic) bool isFoundation;
@property (nonatomic) bool isLipStick;
@property (nonatomic) bool isBeautyFace;

//variables for trackbar values for each live feed feature
@property (nonatomic) int ColorPercent;
@property (nonatomic) double ClipLimit;
@property (nonatomic) double fairtreatPert;
@property (nonatomic) double redtreatPert;
@property (nonatomic) double teethWhitePert;
@property (nonatomic) double foundCoveragePert;
@property (nonatomic) bool isMatchFoundation;
@property (nonatomic) double lipCoveragePert;
@property (nonatomic) double lipglosssPert;

/**
 how much feathering woud like to have- new param
 */
@property (nonatomic) double liplinerPert;
@property (nonatomic) NSMutableArray* foundationColor;
@property (nonatomic) NSMutableArray* lipColor;
@property (nonatomic) double beautytreatPert;

//feature return variables
@property (nonatomic) int matchedFoundationColor;
@property (nonatomic) int StatusCode_Side;
@property (nonatomic) NSString* FPS;


//morphing feature variables
@property (nonatomic) bool isBlink;
@property (nonatomic) bool isMorphtriangles;
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

//private
#ifdef __cplusplus
-(void) assigningLandmarkvariales;
-(bool) performMorphingCommon:(cv::Mat&) inputImage arg1:(int[]) fldPoints;
-(bool) performMorphingLive:(cv::Mat&) inputImage arg1:(int[]) fldPoints;
-(NSMutableArray*)FeatureProcessingOnMatImage:(NSMutableArray *)IQC_ThreshVals landmarkPoints:(int *)landmarkPoints resizeImage:(cv::Mat &)resizeImage;
#endif

-(void) TestMethod:(UIImage*)LiveFrame FaceLandMarks:(NSMutableArray*)MediaPipeLandmarks ;


//public
//makeup
-(void) initializingFLD:(int)imageWidth imageHeight:(int)imageHeight datFilePath:(NSString*)datFilePath;
-(void) setFoundationColors:(NSMutableArray*)cofoundationColorNSArraylors;
-(NSMutableArray*)performLiveFeatures:(CMSampleBufferRef)sampleBuffer IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals;
-(NSMutableArray*)performLiveFeaturesonUIImage:(UIImage*)uiImage IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals;

-(void)n_BeautyFaceonstillImage:(NSString *)InputPath param1:(int)CameraRotationAngle param2:(int)CameraId param3:(NSString*)Outputpath param4:(double)treatpercent;
-(int)performIQConCapture:(NSString*)imgInFile IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals rotation:(bool)isRotationNeeded;
-(void) n_releaseMemory:(int) mode;
//morphing
-(void) n_calculatingFPS;
-(UIImage*) performMorphingStill;
-(bool) initializingstillmat:(NSString*) captureImagePath;
-(void) initializingBRFLive:(int) imageWidth arg:(int)imageHeight;
-(UIImage*)drawHairColor:(UIImage*)uiImage  uiMask:(UIImage*)uiMask color:(NSMutableArray*)hairColor coverage:(CGFloat) coverage;
//ARUI
/**
 loadObjectsInfo

 @param objects objects from server
 @param objectsFldPoints fld points of last frame before capture
 */
-(void) loadObjectsInfo:(NSMutableArray*)objects objectsFldPoints:(NSMutableArray*)objectsFldPoints;
/**
 objectsOverlapping

 @param fldPoints fld points of live frame
 @param getFacialRegions to get facial regions 12
 @return the morphed objects points will be returned
 */
-(NSMutableArray*) objectsOverlapping:(NSMutableArray*)fldPoints getFacialRegions:(bool)getFacialRegions threshold:(double)threshold;

//FoundationMatching

-(NSMutableArray*)performFoundationShadeMatch:(NSString *)inputPath landmarksArr:(NSMutableArray*)landmarksArr colors:(NSMutableArray*)colors skinLab:(NSString *)skinLab;

-(UIImage*) performFoundationSimulation:(int)index coverage:(CGFloat)coverage skinLab:(NSString *)skinLab;
/**
 DermaRegions this will be assigned to values when you sent getFacialRegions true to the objectsOverlapping method
 */
@property (nonatomic) NSMutableArray* DermaRegions;
-(void) releaseMat;

/**
 setSplitFaceParams
 @param viewType -> 0:split face !=0:split screen
 @param params ->
        if split face param[0] - 0:right side effect on ; !=0:left side effect on
        else param[0] - cursor x Cordinate
        if split screen ON <if split screen OFF then no need of these below two params>
            param[1] - window width
            param[2] - view width
 */
-(void) setSplitFaceParams:(int) viewType params:(NSMutableArray*)params isSplitNeeded:(bool)isSplitNeeded;

@end

//default colors for foundation
//int foundationShades_RF[] =
//{  234, 190, 176,
//    213, 162, 134,
//    197, 139, 104,
//    176, 137, 110,
//    152, 111, 84,
//    119, 87, 65
//};


/*
 add(num1: 1,num2: 2)
 add(1, xyz: 2)
 func add(_ num1: Int, xyz num2: Int) {
    num1 + num2
 }
 
 */

