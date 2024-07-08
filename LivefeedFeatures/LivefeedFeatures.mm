//
//  Makeupfeatures_Live.m
//  Makeupfeatures_Live
//
//  Created by Mac5Dev1 on 13/03/18.
//  Copyright © 2018 Mac5Dev1. All rights reserved.
//

#import <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/ios.h>
#import "Landmark_Live.h"
#import "LivefeedFeatures.h"
#include "BeautyFace_Live.h"
#include "BlueLens_Live.h"
//#include "Fairness_Live.h"
#include "HairDetection_Live.h"
#include "MakeupFeatures_Live.h"
#include "Redness_Live.h"
#include "TeethWhitening_Live.h"
#include "Utilities_Live.h"
#include "AreaMorphs_Live.h"
#include "OverlayMorphs_Live.h"
//#include "MorphingUtilities_Live.h"
#include "HairFeatures.h"
#include "FacialRegions_Live.h"
#include "FoundationSimulation_Live.h"

#define LogEnable

@class Landmark_Live;
@interface LivefeedFeatures()
{
    Landmark_Live *fld_L;
}
@property (nonatomic,readwrite,retain)Landmark_Live* fld_L;
@end

Beauty::BeautyFace_Live bfl;
Skincare::BlueLens_Live bl;
Beauty::MakeupFeatures_Live makeup;
Skincare::HairDetection_Live hd;
Skincare::Redness_Live redness;
Skincare::TeethWhitening_Live tw;
Morphing::OverlayMorphs_Live overlayMorphs;
Skincare::FacialRegions_Live facialRegions;
Beauty::FoundationSimulation_Live foundationMatching;

std::string tagLive;

@implementation LivefeedFeatures
{
    int fpsLive;
    int cntFrames;
    NSDate *startTime;
    bool doOperation;
    bool StartedMorph;
    int  objectsLength;
    
    int n_viewType; int n_params[3]; bool n_isSplitFaceNeeded;
}

@synthesize fld_L;
-(id) init
{
    if((self = [super init])!=nil)
    {
        fld_L = [[Landmark_Live alloc]init];
    }
    return self;
}

-(void) assigningLandmarkvariables
{
#ifdef LogEnable
    std::cout << tagLive << "::" << "start Landmark vars Assing" << std::endl;
#endif
    fld_L.useBRF = _useBRF;
    fld_L.DlibStability =true;
    fld_L.stabilityFactor = _stabilityFactor;
    fld_L.stillImageInitialization =false;
    fld_L.Profile = _Profile;//used for IQC checking
    fld_L.ProfileDeterminant = _ProfileDeterminant;//used for IQC checking
    [fld_L setFacerecognormalization:false];//used for Face recog
    [fld_L n_GChannelLipsNeeded:false];//used for makeup features
#ifdef LogEnable
    std::cout << tagLive << "::" << "end Landmark vars Assing" << std::endl;
#endif
}

-(void) initializingFLD:(int)imageWidth imageHeight:(int)imageHeight datFilePath:(NSString*)datFilePath
{
    if(self.isfldinitialized == YES)
    {
#ifdef LogEnable
        std::cout << tagLive << "::" << "Fld is already initialized" << std::endl;
#endif
        return;
    }
    
    fld_L.tagName = _tagName;
    tagLive = [_tagName UTF8String];//assigning tagMF names
    [self assigningLandmarkvariables];
    fld_L.stillImageInitialization =false;
    [fld_L initializingFLDwithPath:imageWidth arg:imageHeight dlibDatFilePath:datFilePath];
    self.isfldinitialized = fld_L.isfldinitialized;
    int defaultVal = 0;
    _FPS  = [NSString stringWithFormat:@"FPS = %d", defaultVal];
    NSLog(@"Frame rate per second : %3@ ",_FPS);
    
    fpsLive=30;
    cntFrames=0;
    doOperation= true;
    startTime = [NSDate date];
    StartedMorph =false;
}

-(void) n_releaseMemory:(int) mode
{
    if(mode == 0)
    {
#ifdef LogEnable
        std::cout << tagLive << ":" <<"Releasing started"<< self.isfldinitialized << std::endl;
#endif
        bfl.releaseMemory();
        
        
        hd.releaseMemory();
        redness.releaseMemory();
        tw.releaseMemory();
#ifdef LogEnable
        std::cout << tagLive << ":" <<"Releasing ended"<< self.isfldinitialized << std::endl;
#endif
    }
}

//region Data type conversions
NSMutableArray* ConvertinttoNSMutableArray(int* Array ,int Size)
{
    NSMutableArray *array =  [[NSMutableArray alloc] init];
    for( int index=0; index <Size; ++index )
    {
        [array addObject: [NSNumber numberWithInt: Array[index]]];
    }
    return array;
}
//endregion

void n_BeautyFace(cv::Mat & liveImage ,int * landMarkPoints ,double beautyTreatedOffset)
{
    bfl.performBeautyFace(liveImage, landMarkPoints,beautyTreatedOffset);
}

void n_BlueLens (cv::Mat & liveImage, int * landMarkPoints, bool isBlueBlackChecked, bool isBlueImage, int Channel ,int colorPercent ,double clipLimit)
{
    bl.performBlueLens(liveImage, landMarkPoints, isBlueBlackChecked, isBlueImage, Channel, colorPercent, clipLimit);
}

NSMutableArray* n_Fairness(cv::Mat & liveImage ,int * landMarkPoints ,double fairnessOffset)
{
    makeup.complexion(liveImage, 1, landMarkPoints, true, fairnessOffset, true, fairnessOffset);
    //int Size = sizeof(makeup.feedback)/sizeof(makeup.feedback[0]);
    NSMutableArray *FairnessValues;// = ConvertinttoNSMutableArray(makeup.feedback ,Size);
    
    return FairnessValues;
}

NSMutableArray* n_HairFrizzDetection(cv::Mat & liveImage,int * landMarkPoints)
{
    hd.detectHair(liveImage, landMarkPoints);
    int Size = sizeof(hd.measurements)/sizeof(hd.measurements[0]);
    NSMutableArray *HairFrizzValues = ConvertinttoNSMutableArray(hd.measurements ,Size);
    return HairFrizzValues;
}


NSMutableArray* n_Redness(cv::Mat & liveImage ,int * landMarkPoints ,bool isLocalRedness ,double rednessOffset)
{
    redness.performRedness(liveImage, landMarkPoints,isLocalRedness,rednessOffset);
    int Size = sizeof(redness.rednessvalues)/sizeof(redness.rednessvalues[0]);
    NSMutableArray *RednessValues = ConvertinttoNSMutableArray(redness.rednessvalues,Size);
    return RednessValues;
}

NSMutableArray * n_TeethWhitening(cv::Mat & Frame ,int * landMarkPoints ,bool isWithoutGreenROI ,int percentagMFe)
{
    double value = percentagMFe / 100.0;
    tw.doTeethWhitening(Frame, landMarkPoints,isWithoutGreenROI,value);
    int Size = sizeof(tw.measurements)/sizeof(tw.measurements[0]);
    NSMutableArray *Measurments = ConvertinttoNSMutableArray(tw.measurements ,Size);
    return Measurments;
}

int n_Foundation(cv::Mat& inputImage ,int* fldPoints,int* color ,float coverage ,bool isMatchFoundation)
{
    Beauty::MakeupFeatures_Live makeup;
    int matchedIndex = makeup.foundation(inputImage, fldPoints, color, coverage, isMatchFoundation);
    return matchedIndex;
}

void n_lipstick(cv::Mat& inputImage, int* fldPoints, int* color, float coverage, float glossValue, float lipLinerValue)
{
    Beauty::MakeupFeatures_Live makeup;
    //makeup.lipstick(inputImage, fldPoints, color, coverage, 0.7, glossValue);
    makeup.lipstick(inputImage, 1, fldPoints, color, coverage, 0.7, glossValue, lipLinerValue);
}

-(void) setFoundationColors:(NSMutableArray*)colors
{
    int colorsNumber = (int)colors.count;
    
#ifdef LogEnable
    std::cout << tagLive << "::" << "Foundation colors assignment started and colors count is" << colorsNumber  <<std::endl;
#endif
    
    int* foundationColors=new int[colorsNumber];
    int i=0;
    for(i=0;i<colorsNumber;i++)
    {
        foundationColors[i]= (int)[[colors objectAtIndex:i] integerValue];
    }
    Beauty::MakeupFeatures_Live::setFoundationShades(foundationColors,colorsNumber);
    
    if(foundationColors!=NULL)
    {
        delete []foundationColors;
        foundationColors =NULL;
    }
#ifdef LogEnable
    std::cout << tagLive << "::" << "Foundation colors assignment ended" <<std::endl;
#endif
}


- (void)ApplySplitFace:(int *)landmarkPoints originalFrame:(const cv::Mat &)originalFrame outputImage:(cv::Mat &)resizeImage {
    int out_Points[6];
    
    Utilities_Live::splitViews(originalFrame, resizeImage, landmarkPoints, n_viewType, n_params, out_Points);
    
    _drawingPoints = [NSMutableArray arrayWithCapacity:6];
    
    for(int k=0;k<6;k++)
    {
        [_drawingPoints addObject:[NSNumber numberWithInt: out_Points[k]]];
    }
}

-(void) TestMethod:(UIImage*)LiveFrame FaceLandMarks:(NSMutableArray*)MediaPipeLandmarks
{
    std::cout << "MediaPipe" << "::" << "FCall from media pipe framework.."<<std::endl;
}


-(NSMutableArray*)FeatureProcessingOnMatImage:(NSMutableArray *)IQC_ThreshVals landmarkPoints:(int *)landmarkPoints resizeImage:(cv::Mat &)resizeImage
{
    NSMutableArray* _Measurements = [[NSMutableArray alloc] init];
    
    cv::Mat originalFrame ;
    
    clock_t start=clock();
    if (landmarkPoints[150] == 1)
    {
        _StatusCode = 2012;
#ifdef LogEnable
        std::cout << tagLive << "::" << "Face is not detected.Please make yourself align with the camera and closer to camera,statuscode"<< _StatusCode<<std::endl;
#endif
    }
    else
    {
        if(_isLandmarksFetching == true)
        {
#ifdef LogEnable
            std::cout<<tagLive<<"landmarks fetching started:: "<<std::endl;
#endif
            _landMarkPointsonFullRezFrame = [NSMutableArray arrayWithCapacity:151];
            for(int i =0;i<151;i++)
            {
#ifdef LogEnable
                std::cout<<tagLive<<"FLD Points at:: " << i <<":"<<landmarkPoints[i]<<std::endl;
#endif
                
                NSNumber* landmarkPt = [NSNumber numberWithInt:landmarkPoints[i]];
                [_landMarkPointsonFullRezFrame addObject:landmarkPt];
            }
#ifdef LogEnable
            std::cout<<tagLive<<"landmarks fetching Ended:: "<<std::endl;
#endif
        }
        if(_is7regionsFetching == true)
        {
#ifdef LogEnable
            std::cout<<tagLive<<"4 regions fetching started:: "<<std::endl;
#endif
            std::vector<cv::Point>FLDPoints(75,cv::Point(0,0));
            for (int i = 0; i < (150 / 2); i++)
            {
                FLDPoints[i].x = landmarkPoints[i * 2];
                FLDPoints[i].y = landmarkPoints[i * 2 + 1];
            }
            
            std::vector<std::pair<std::string, std::vector<cv::Point>>> Facial4Regions;
            Facial4Regions = facialRegions.facialZones(FLDPoints, 4);
            
            _facial4RegionPoints = [[NSMutableArray alloc] initWithCapacity:Facial4Regions.size()];
#ifdef LogEnable
            std::cout<<tagLive<<"_facial4RegionPoints:: " <<Facial4Regions.size() <<std::endl;
#endif
            for (int i = 0; i < Facial4Regions.size(); i++)
            {
                std::vector<cv::Point> Facial4RegionsL = Facial4Regions[i].second;
#ifdef LogEnable
                std::cout<<tagLive<<"Facial4RegionsL:: " <<Facial4RegionsL.size() <<std::endl;
#endif
                NSMutableArray *inner = [[NSMutableArray alloc] initWithCapacity:Facial4RegionsL.size()*2];
                
                for (int j = 0; j < Facial4RegionsL.size(); j++)
                {
                    [inner addObject:[NSNumber numberWithFloat: Facial4RegionsL[j].x]];
                    [inner addObject:[NSNumber numberWithFloat: Facial4RegionsL[j].y]];
                    
#ifdef LogEnable
                    std::cout<<tagLive<<"Facial4RegionsL(x,y):: " <<Facial4RegionsL[j].x << "," << Facial4RegionsL[j].y<<std::endl;
#endif
                }
                [_facial4RegionPoints addObject:inner];
            }
            
#ifdef LogEnable
            std::cout<<tagLive<<"4 regions fetching Ended:: "<<std::endl;
#endif
        }
        
        //displaying landmarks
        if(_isLandmarksonFace == true || _isFaceBounds == true)
        {
            bool isboundsChecked = false;
            if(_isLandmarksonFace == true)
            {
                isboundsChecked = true;
            }
            int rectcoordinates[4];//assigning profile and coordinates
            [fld_L copingrectcoordinates:rectcoordinates];
            fld_L.Profile = _Profile;
#ifdef LogEnable
            std::cout << tagLive << "::" << "Join landmarks started" <<std::endl;
#endif
            if(_isFaceBounds == true)
            {
                cv::Mat resizeImageTemp = resizeImage.clone();
                [fld_L n_JoinLandMarks:resizeImageTemp j_LandMarks:landmarkPoints para1
                                      :isboundsChecked para2:rectcoordinates para3:_ProfileDeterminant];
                double alpha = _ColorPercent / 100.0;
                cv::addWeighted(resizeImage, 1-alpha, resizeImageTemp, alpha, 0, resizeImage);
            }
            else
            {
                [fld_L n_JoinLandMarks:resizeImage j_LandMarks:landmarkPoints para1
                                      :isboundsChecked para2:rectcoordinates para3:_ProfileDeterminant];
            }
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Join landmarks ended" <<std::endl;
#endif
        }
        
        if (_isAutoCapture ==true)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "IQC Checking started" <<std::endl;
#endif
            
            _StatusCode = [fld_L performIQConMat:resizeImage IQC_ThreshVals:IQC_ThreshVals eyeCheck:2];
            _StatusCode_Side = [fld_L profiledeterminant];
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "IQC Checking ended,StatusCode:" << _StatusCode << ",StatusCodeSide:" << _StatusCode_Side <<std::endl;
#endif
        }
        
        originalFrame = resizeImage.clone();
        //scale changes
        //new value = ((old value - oldscale min) * (newscalemax - newscalemin)/(oldscalemax - oldscalemin)) + newscalemin
                
        //[self n_SplitFace:_isSplit];//for spliting mask
        if(_isEVDamage == true || _isBlochiness == true || _isHighSkinPigmentation == true ||  _isAverageContrast == true)
        {
            bool isBlack = _isBlackBG;
            bool isBlueImage = _isGrayImage;
            int channel = 2;
            if(_isHighSkinPigmentation == true)
            {
                channel = 0;
            }
            if(_isBlochiness == true)
            {
                channel = 1;
            }
            if(_isAverageContrast == true)
            {
                channel = -1;
            }
            int colorPercent1=_ColorPercent;
            double clipLimit1 =_ClipLimit;
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "BlueLens started,cliplimit:"<< clipLimit1 << ",ColorPercent:"<< colorPercent1<<std::endl;
#endif
            n_BlueLens(resizeImage ,landmarkPoints,isBlack ,isBlueImage ,channel ,colorPercent1 ,clipLimit1);
            
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "BlueLens ended"<<std::endl;
#endif
        }
        if(_isFairness==true)
        {
            double max = 8;
            double fairnessValue = _fairtreatPert * max;
            
#ifdef LogEnable
            std::cout << tagLive << "::" << " Fairness started,fairtreat:"<< _fairtreatPert << ",FairnessValue:"
            << fairnessValue <<std::endl;
#endif
            //lpercentagMFe offset , its a trackbar value sent from GUI
            _Measurements = n_Fairness(resizeImage ,landmarkPoints ,fairnessValue);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Complexion ended"<<std::endl;
#endif
        }
        if(_isRednesslive==true)
        {
            double max = 0.25;
            double redValue = _redtreatPert * max;
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Redness started,redtreat:"<< _redtreatPert << ",ChangedRedvalue:"<< redValue <<std::endl;
#endif
            _Measurements = n_Redness(resizeImage,landmarkPoints ,false ,redValue);
            
#ifdef LogEnable
            std::cout << tagLive << "::" <<"Redness ended"<<std::endl;
#endif
        }
        if(_isTeethWhitening == true)
        {
            double max = 100;
            double teethWhiteVal = _teethWhitePert * max;
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Teeth whitening started,teethwhitepert:"<< _teethWhitePert  << ",teethWhiteVal:" <<teethWhiteVal <<std::endl;
#endif
            _Measurements = n_TeethWhitening(resizeImage,landmarkPoints ,_isNoGreenROI ,teethWhiteVal);
#ifdef LogEnable
            std::cout << tagLive << "::" << "Teeth whitening ended"<<std::endl;
#endif
        }
        if(_isHairFrizz == true)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "Hair frizz started"<<std::endl;
#endif
            _Measurements = n_HairFrizzDetection(resizeImage ,landmarkPoints);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Hair frizz ended"<<std::endl;
#endif
        }
        if(_isFoundation)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "new foundation started,Coverage:"<< _foundCoveragePert << std::endl;
#endif
            int foundationColor[3];
            for(int i=0;i<3;i++)
            {
                foundationColor[i] = (int)[[_foundationColor objectAtIndex:i] integerValue];
            }
            _matchedFoundationColor=n_Foundation(resizeImage,landmarkPoints ,foundationColor ,_foundCoveragePert ,_isMatchFoundation);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "new foundation ended"<<std::endl;
#endif
        }
        if(_isLipStick)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "new lipstick started,cover:"<<_lipCoveragePert << ",gloss:" << _lipglosssPert << std::endl;
#endif
            int lipstickColor[3];
            for(int i=0;i<3;i++)
            {
                lipstickColor[i] = (int)[[_lipColor objectAtIndex:i] integerValue];
            }
            n_lipstick(resizeImage,landmarkPoints ,lipstickColor ,_lipCoveragePert ,_lipglosssPert, _liplinerPert);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "new lipstick ended"<<std::endl;
#endif
        }
        if(_isBeautyFace == true)
        {
#ifdef LogEnable
            //            std::cout << tagLive << "::" << "beauty face started,beautytreatPert:"<< _beautytreatPert << ",beautyVal:"<< beautyVal <<std::endl;
#endif
            n_BeautyFace(resizeImage,landmarkPoints ,_beautytreatPert);
#ifdef LogEnable
            std::cout << tagLive << "::" << "beauty face  ended"<<std::endl;
#endif
        }
        
        //Utilities_Live::
        
        [self performMorphingLive:resizeImage arg1:landmarkPoints];
    }
    //clock_t end=clock();
#ifdef LogEnable
    //    std::cout << tagLive << "::" <<"Time to complete live feed features: "<<(double)(end-start)/ CLOCKS_PER_SEC<<"....secs."<<std::endl;
#endif
    //spilt function call if split true
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:landmarkPoints originalFrame:originalFrame outputImage:resizeImage];
    }
    return _Measurements;
}

-(NSMutableArray*)performLiveFeatures:(CMSampleBufferRef)sampleBuffer IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals
{
    _StatusCode = 2012;
    [self n_calculatingFPS];
    [self assigningLandmarkvariables];
     NSMutableArray* _Measurements = [[NSMutableArray alloc] init];
   
    if(_isTeethWhitening == true || _isAutoCapture == true)
    {
        [fld_L n_GChannelLipsNeeded:false];
    }
    else
    {
        [fld_L n_GChannelLipsNeeded:true];
    }
    
    int landmarkPoints[151];
    cv::Mat resizeImage;
    [fld_L landmarkdetectiononcamerabuffer:sampleBuffer arg1:resizeImage];
    //landmark coping
  
    [fld_L copinglandmarks:landmarkPoints];
    
    _Measurements = [self FeatureProcessingOnMatImage:IQC_ThreshVals landmarkPoints:landmarkPoints resizeImage:resizeImage];
    
    //back to camera buffer
    [fld_L placingmatontocamerabuffer:sampleBuffer arg1:resizeImage];
    
#ifdef LogEnable
    std::cout << tagLive << "::" << "Status code at auto capture"<< _StatusCode<<std::endl;
#endif
    
    return _Measurements;
}

-(NSMutableArray*)performLiveFeaturesonUIImage:(UIImage*)uiImage IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals
{
    _StatusCode = 2012;
    [self n_calculatingFPS];
    [self assigningLandmarkvariables];
    NSMutableArray* _Measurements = [[NSMutableArray alloc] init];
    
    if(_isTeethWhitening == true || _isAutoCapture == true)
    {
        [fld_L n_GChannelLipsNeeded:false];
    }
    else
    {
        [fld_L n_GChannelLipsNeeded:true];
    }
    
    int landmarkPoints[151];
    cv::Mat resizeImage;
    [fld_L landmarkdetectiononUIimage:uiImage arg1:resizeImage];
    //landmark coping
    
    [fld_L copinglandmarks:landmarkPoints];
    
    _Measurements = [self FeatureProcessingOnMatImage:IQC_ThreshVals landmarkPoints:landmarkPoints resizeImage:resizeImage];
    
#ifdef LogEnable
    std::cout << tagLive << "::" << "Status code at auto capture"<< _StatusCode<<std::endl;
#endif
    
    return _Measurements;
}

-(void)n_BeautyFaceonstillImage:(NSString *)InputPath param1:(int)CameraRotationAngle param2:(int)CameraId param3:(NSString*)Outputpath param4:(double)treatpercent
{
    std::string inputImagePath = [InputPath UTF8String];
    std::string outputImagePath = [Outputpath UTF8String];
    
    Beauty::BeautyFace_Live beauty;
    cv::Mat inputImage = cv::imread(inputImagePath);
    int featureResizeFactor=std::sqrt((double) (inputImage.rows*inputImage.cols)/(540*720));
    cv::Mat preProcessingInputImage;
    cv::Mat rotateImage;
    int width=inputImage.cols/featureResizeFactor, height=inputImage.rows/featureResizeFactor;
    cv::resize(inputImage, preProcessingInputImage, cv::Size(width, height));
    //std::cout << tagLive << "::" <<"BeautyFace: Camera Rotation Angle:" << CameraRotationAngle << std::endl;
    [fld_L performRotation:preProcessingInputImage arg1:rotateImage arg2:CameraRotationAngle arg3:CameraId];
    
    //landmark detection
    [fld_L detectingLandmarks:rotateImage];
    //landmark coping
    int landmarkPoints[151];
    //std::cout << tagLive << "::" << "Landmark live coping started"<<std::endl;
    [fld_L copinglandmarks:landmarkPoints];
    //std::cout << tagLive << "::" << "Landmark live coping ended"<<std::endl;
    
    std::cout<<tagLive<<"Started logging FLD Points::"<<std::endl;
    //    for(int i=0; i<=150;i++)
    //    {
    //        std::cout<<tagLive<<"n_BeautyFaceonstillImage FLD Points:: "<<landmarkPoints[i]<<std::endl;
    //    }
    inputImage =rotateImage.clone();
    if(landmarkPoints[150]!=1)
    {
        beauty.performBeautyFace(rotateImage, landmarkPoints, treatpercent);
        cv::imwrite(outputImagePath, rotateImage);
    }
    else
    {
        cv::imwrite(outputImagePath, rotateImage);
    }
    
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:landmarkPoints originalFrame:inputImage outputImage:rotateImage];
    }
    
    rotateImage.release();
    inputImage.release();
    preProcessingInputImage.release();
}

-(int)performIQConCapture:(NSString*)imgInFile IQC_ThreshVals:(NSMutableArray*)IQC_ThreshVals rotation:(bool)isRotationNeeded
{
    _StatusCode = [fld_L performIQConCapture:imgInFile IQC_ThreshVals:IQC_ThreshVals rotation:isRotationNeeded];
    return _StatusCode;
    
}

-(void) initializingBRFLive:(int) imageWidth arg:(int)imageHeight
{
    [self assigningLandmarkvariables];
    fld_L.stillImageInitialization =false;
    [fld_L initializingBRF:imageWidth arg:imageHeight];
    
}

-(void)n_calculatingFPS
{
    NSTimeInterval timeInterval = fabs([startTime timeIntervalSinceNow]);
    cntFrames++;
    if(timeInterval>1)//1 sec
    {
        startTime=[NSDate date];
        fpsLive=cntFrames;
        cntFrames=0;
        Utilities_Live::fps=fpsLive;
    }
    
    _FPS = [NSString stringWithFormat:@"FPS = %d", fpsLive];
    
    //std::cout << tagLive << "::" << "FPS_" << fpsLive << std::endl;
    
    if(_isBlink)
    {
        if(timeInterval>1)//1 sec
        {
            doOperation=!doOperation;
        }
    }
    else
    {
        doOperation=true;
    }
}
-(void) loadObjectsInfo:(NSMutableArray*)objects objectsFldPoints:(NSMutableArray*)objectsFldPoints
{
    if(!_useBRF)
    {
        std::cout<<"This feature needs brf as we need float fld points"<<std::endl;
        return;
    }
    std::vector<std::vector<cv::Point2f>> contours;
    objectsLength = (int)objects.count;
    std::cout << tagLive << "::" << "AR Points count: " << objectsLength  <<std::endl;

    for (int i = 0; i < objectsLength;)
    {
        std::vector<cv::Point2f> contour;
        int n = i + (int)[[objects objectAtIndex:i] floatValue];
        for (i = i + 1; i < n; i = i + 2)
        {
            contour.push_back(cv::Point2f((float)[[objects objectAtIndex:i] floatValue], (float)[[objects objectAtIndex:(i+1)] floatValue]));
        }
        contours.push_back(contour);
    }
    std::cout << tagLive << "::" << "AR objects count: " << contours.size()  <<std::endl;
    
    const int fldLength = 150;
    float fldPoints[fldLength];
    for (int i = 0; i < fldLength;i++)
    {
        fldPoints[i]=(float)[[objectsFldPoints objectAtIndex:i] floatValue];
    }
    overlayMorphs.loadObjectsInfo(contours, fldPoints);
}

-(NSMutableArray*) objectsOverlapping:(NSMutableArray*)fldPoints getFacialRegions:(bool)getFacialRegions threshold:(double)threshold
{
    NSMutableArray *ARObjects;
    
    if(!_useBRF)
    {
        std::cout<<"This feature needs brf as we need float fld points"<<std::endl;
        return ARObjects;
    }
    const int fldLength = 150;
    float landmarkPoints[fldLength];
    for (int i = 0; i < fldLength;i++)
    {
        landmarkPoints[i]=(float)[[fldPoints objectAtIndex:i] floatValue];
    }
    
    if ((int)landmarkPoints[150] == 1)
    {
#ifdef LogEnable
        std::cout << tagLive << "::" << "Face is not detected.Please make yourself align with the camera and closer to camera,statuscode"<< _StatusCode<<std::endl;
#endif
         _DermaRegions = [NSMutableArray arrayWithCapacity:0];
        return ARObjects;
    }
    else
    {
        if(getFacialRegions)
        {
            std::vector<cv::Point>FLDPoints(75,cv::Point(0,0));
            for (int i = 0; i < (150 / 2); i++)
            {
                FLDPoints[i].x = landmarkPoints[i * 2];
                FLDPoints[i].y = landmarkPoints[i * 2 + 1];
            }
            std::vector<std::pair<std::string, std::vector<cv::Point>>> Facial5Regions;
            Facial5Regions = facialRegions.facialZones(FLDPoints, 5);
            
            std::vector<std::pair<std::string, std::vector<cv::Point>>> dermaRegions;
            _DermaRegions = [[NSMutableArray alloc] initWithCapacity:Facial5Regions.size()];
#ifdef LogEnable
            std::cout<<tagLive<<"_DermaRegions:: " <<Facial5Regions.size() <<std::endl;
#endif
            for (int i = 0; i < Facial5Regions.size(); i++)
            {
                std::vector<cv::Point> Facial5RegionsL = Facial5Regions[i].second;
#ifdef LogEnable
                std::cout<<tagLive<<"Facial5RegionsL:: " <<Facial5RegionsL.size() <<std::endl;
#endif
                NSMutableArray *inner = [[NSMutableArray alloc] initWithCapacity:Facial5RegionsL.size()*2];
                
                for (int j = 0; j < Facial5RegionsL.size(); j++)
                {
                    [inner addObject:[NSNumber numberWithFloat: Facial5RegionsL[j].x]];
                    [inner addObject:[NSNumber numberWithFloat: Facial5RegionsL[j].y]];
                    
#ifdef LogEnable
                    std::cout<<tagLive<<"Facial5RegionsL(x,y):: " <<Facial5RegionsL[j].x << "," << Facial5RegionsL[j].y<<std::endl;
#endif
                }
                
                [_DermaRegions addObject:inner];
            }
        }
        
        std::vector<Morphing::ObjectInfo> objectsInfoDst = overlayMorphs.objectsOverlapping(landmarkPoints, threshold);
        if (objectsInfoDst.empty())
        {
            return ARObjects;
        }
        int n = (int)objectsInfoDst.size();
        ARObjects = [[NSMutableArray alloc] init];
        ARObjects = [NSMutableArray arrayWithCapacity:objectsLength];
        
        for (int i = 0; i < n; i++)
        {
            std::vector<cv::Point2f> &contour = objectsInfoDst[i].polygon;
            int n1 = (int)contour.size();
            [ARObjects addObject:[NSNumber numberWithFloat: (n1 * 2)]];
            for (int j = 0; j < n1; j++)
            {
                [ARObjects addObject:[NSNumber numberWithFloat: contour[j].x]];
                [ARObjects addObject:[NSNumber numberWithFloat: contour[j].y]];
            }
        }
    }
    return ARObjects;
}

-(bool) performMorphingCommon:(cv::Mat&) inputImage arg1:(int[]) fldPoints
{
#ifdef LogEnable
    //    std::cout<<tagLive<<"Started logging FLD Points::"<<std::endl;
    //    for(int i=0; i<=150;i++)
    //    {
    //        std::cout<<tagLive<<"FLD Points:: "<<fldPoints[i]<<std::endl;
    //    }
    
    std::cout<<tagLive<<"Ended logging FLD Points::"<<std::endl;
#endif
    Beauty::MakeupFeatures_Live makeup;
    
    cv::Mat OriginalFrame = inputImage.clone();
    
    if(_isComplexion)
    {
#ifdef LogEnable
        std::cout << tagLive << "::" << "Complexion started" <<std::endl;
#endif
        makeup.complexion(inputImage, 1, fldPoints, true, _complexionIntensity, true, _complexionIntensity);
        
#ifdef LogEnable
        std::cout << tagLive << "::" << "Complexion ended" <<std::endl;
#endif
    }
    if(_isForeheadComplexion||_isUnderEyeComplexion||_isNasolabialComplexion)
    {
        float intensities[3]={_foreheadComplexionIntensity,_underEyeComplexionIntensity,_nasolabialComplexionIntensity};
        bool zones[3]={_isForeheadComplexion,_isUnderEyeComplexion,_isNasolabialComplexion};
        makeup.wrinkles(inputImage,fldPoints,zones,intensities);
    }
    
    if(_isLipHealth)
    {
#ifdef LogEnable
        std::cout << tagLive << "::" << "lip health started" <<std::endl;
#endif
        makeup.lipHealth(inputImage, fldPoints,_lipHealthIntensity);
#ifdef LogEnable
        std::cout << tagLive << "::" << "lip health ended" <<std::endl;
#endif
    }
    Morphing::AreaMorphs_Live areaMorphs;
    bool status=true;
    if(_isDistortion)
    {
#ifdef LogEnable
        std::cout << tagLive << "::" << "Morphing distortion started" <<std::endl;
#endif
        
        areaMorphs.noseMorph(inputImage, fldPoints, 0.35, _isMorphtriangles);
        areaMorphs.lipMorph(inputImage, fldPoints, 0.75, _isMorphtriangles);
        areaMorphs.eyeMorph(inputImage, fldPoints, 0.65, _isMorphtriangles);
        
#ifdef LogEnable
        std::cout << tagLive << "::" << "Morphing distortion ended" <<std::endl;
#endif
    }
    else
    {
        if(_isLipCornerMorphing)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "lip corner morphing started,intensity:"<<_lipCornerMorphingIntensity  <<std::endl;
#endif
            areaMorphs.lipCornerMorph(inputImage, fldPoints, _lipCornerMorphingIntensity, _isMorphtriangles);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "lip corner morphing ended" <<std::endl;
#endif
        }
        if(_isJawMorphing)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "Jaw morphing started,intensity" << _jawMorphingIntensity  <<std::endl;
#endif
            areaMorphs.jawlineMorph(inputImage, fldPoints, _jawMorphingIntensity, _isMorphtriangles);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Jaw morphing ended" <<std::endl;
#endif
        }
        if(_isNoseMorphing)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "Nose morphing started,intensity" <<_noseMorphingIntensity <<std::endl;
#endif
            
            areaMorphs.noseMorph(inputImage, fldPoints, _noseMorphingIntensity, _isMorphtriangles);
            
#ifdef LogEnable
            std::cout << tagLive << "::" << "Nose morphing ended" <<std::endl;
#endif
        }
        
        if(_isLipMorphing)
        {
#ifdef LogEnable
            std::cout << tagLive << "::" << "Lip morphing started,intensity" << _lipMorphingIntensity  <<std::endl;
#endif
            areaMorphs.lipMorph(inputImage, fldPoints, _lipMorphingIntensity, _isMorphtriangles);
#ifdef LogEnable
            std::cout << tagLive << "::" << "Lip morphing ended" <<std::endl;
#endif
        }
        if (_eyebrowMorphingIntensity < 0.5)
        {
            if(_isEyeMorphing)
            {
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye morphing started,intensity" <<_eyeMorphingIntensity  <<std::endl;
#endif
                
                areaMorphs.eyeMorph(inputImage, fldPoints, _eyeMorphingIntensity, _isMorphtriangles);
                
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye morphing ended" <<std::endl;
#endif
            }
            if(_isEyebrowMorphing)
            {
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye brow morphing started,intensity" << _eyebrowMorphingIntensity <<std::endl;
#endif
                areaMorphs.eyebrowMorph(inputImage, fldPoints, _eyebrowMorphingIntensity, _isMorphtriangles);
                
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye brow morphing ended" <<std::endl;
#endif
            }
            else if(_isEyebrowShapeMorphing)
            {
                Morphing::OverlayMorphs_Live overlayMorphs;
                status=overlayMorphs.eyebrowMorph(inputImage, fldPoints, _eyebrowshapeIndex, _eyebrowshapeMorphingIntensity, _isMorphtriangles);
            }
        }
        else
        {
            if(_isEyebrowMorphing)
            {
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye brow morphing started" <<std::endl;
#endif
                
                areaMorphs.eyebrowMorph(inputImage, fldPoints, _eyebrowMorphingIntensity, _isMorphtriangles);
                
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye brow morphing ended" <<std::endl;
#endif
            }
            else if(_isEyebrowShapeMorphing)
            {
                Morphing::OverlayMorphs_Live overlayMorphs;
                status= overlayMorphs.eyebrowMorph(inputImage, fldPoints, _eyebrowshapeIndex, _eyebrowshapeMorphingIntensity, _isMorphtriangles);
            }
            if(_isEyeMorphing)
            {
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye morphing started" <<std::endl;
                std::cout << tagLive << "::" << "Eye morphing intensity" <<_eyeMorphingIntensity<<std::endl;
#endif
                
                areaMorphs.eyeMorph(inputImage, fldPoints, _eyeMorphingIntensity, _isMorphtriangles);
                
#ifdef LogEnable
                std::cout << tagLive << "::" << "Eye morphing ended" <<std::endl;
#endif
            }
        }
        
    }
    
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:fldPoints originalFrame:OriginalFrame outputImage:inputImage];
    }
    
    return status;
}

cv::Mat inputImage;
cv::Mat liveImage;
-(bool) initializingstillmat:(NSString*) captureImagePath
{
    //UIImageToMat(captureImage,inputImage,false);
    
    std::string inputPath = [captureImagePath UTF8String];
    //std::cout << tagLive << "::" << "Inputpath" << inputPath <<std::endl;
    inputImage = cv::imread(inputPath,1);
    if(inputImage.data==0)
    {
        return false;
    }
    int cols = inputImage.cols,rows =inputImage.rows ;
    //std::cout << tagLive << "::" << "Still input image cols and rows"<<cols<< "," <<rows <<std::endl;
    //std::cout<<"inputImage: "<<cols<<","<<rows<<std::endl;
    double feature_resizeFactor = std::sqrt((double) (inputImage.rows * inputImage.cols) / (480 * 640));
    int resizeCols = (int)(cols/feature_resizeFactor);
    int resizeRows = (int)(rows/feature_resizeFactor);
    cv::resize(inputImage, liveImage, cv::Size(resizeCols,resizeRows));
    //std::cout<<"resizeImage: "<<resizeCols<<","<<resizeRows<<std::endl;
    NSString *modelFileName = [[NSBundle mainBundle] pathForResource:@"Faciallandmarkdetection_dlib" ofType:@"dat"];
    //dlib initialization
    [fld_L LoadModel:modelFileName];
    return true;
}

-(UIImage*) performMorphingStill
{
    if(liveImage.data == 0 )
    {
        //std::cout << tagLive << "::" <<  "CommonMat data is zero" <<std::endl;
        UIImage* dummyImage;
        return dummyImage;
    }
    
    //initializing BRF for every frame
    fld_L.stillImageInitialization =true;
    [fld_L initializingBRF:liveImage.cols arg:liveImage.rows];
    [self assigningLandmarkvariables];
    //std::cout<<tagLive<<"::"<<"Shared stability Morphing Still "<<Utilities_Live::isMorphStable<<std::endl;
    
    cv::Mat updatedImage;
    cv::cvtColor(liveImage,updatedImage,cv::COLOR_BGR2RGB);  //color conversion for live features to run face detection common for dlib and BRF
#ifdef LogEnable
    std::cout << tagLive << "::" << "fld detection on still started"<<std::endl;
#endif
    [fld_L detectingLandmarks:updatedImage];
    
    cv::Mat OriginalFrame = updatedImage.clone();
    
#ifdef LogEnable
    std::cout << tagLive << "::" <<"fld detection on still ended"<<std::endl;
#endif
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    std::string documentsPath = [documentsDirectory UTF8String];
    
    //std::ostringstream buffer;
    //buffer<<_lipMorphingIntensity;
    
    //std::string input =  buffer.str()+ "Input.jpg";//[imgInFile UTF8String];
    
    //std::string overallimagePath =documentsPath+"/"+input;
    //cv::imwrite(overallimagePath,updatedImage);
    
    int fldPoints[151];
    //landmark coping
    [fld_L copinglandmarks:fldPoints];
    
    //std::cout<<tagLive<<"Started logging FLD Points::"<<std::endl;
    //for(int i=0; i<=150;i++)
    //{
    //    std::cout<<tagLive<<"performMorphingStill FLD Points:: "<<fldPoints[i]<<std::endl;///
    //}
    
    clock_t start1=clock();
    if (fldPoints[150] == 1)
    {
        // std::cout << tagLive << "::" "Face is not detected.Please make yourself align with the camera and closer to camera"<<std::endl;
    }
    else
    {
        //saving image path
        [self performMorphingCommon:updatedImage arg1:fldPoints];
        //std::cout<<tagLive<< ":: LipPlumb  "<<_isLipMorphing <<","<<_lipMorphingIntensity<<std::endl;
        //std::cout<<tagLive<< ":: foreheadComplexion  "<<_isForeheadComplexion <<","<<_foreheadComplexionIntensity<<std::endl;
    }
    clock_t end1=clock();
    //std::cout << tagLive << "::" <<"Time to complete Morphing on still: "<<(double)(end1-start1)/ CLOCKS_PER_SEC<<"....secs."<<std::endl;
    
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:fldPoints originalFrame:OriginalFrame outputImage:updatedImage];
    }
    
    UIImage* captureImage = MatToUIImage(updatedImage);
    return captureImage;
}
-(void) releaseMat
{
    inputImage.release();
    liveImage.release();
}

-(bool) performMorphingLive:(cv::Mat&) inputImage arg1:(int[]) fldPoints
{
    Utilities_Live::isMorphStable = [fld_L copingstablity:false];
    //std::cout << tagLive << "::" << "Shared stability Morphing Live" <<  Utilities_Live::isMorphStable  <<std::endl;
    
    if(!StartedMorph)
    {
        Utilities_Live::isMorphStable =false;
        StartedMorph = true;
    }
    bool status=true;
    clock_t start1=clock();
    
    cv::Mat OriginalFrame = inputImage.clone();
    
    if (fldPoints[150] == 1)
    {
        //std::cout << tagLive << "::" "Face is not detected.Please make yourself align with the camera and closer to camera"<<std::endl;
        status=false;
    }
    else
    {
        if(doOperation)
        {
            status=[self performMorphingCommon:inputImage arg1:fldPoints];
            
        }
    }
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:fldPoints originalFrame:OriginalFrame outputImage:inputImage];
    }
    
    clock_t end1=clock();
    // std::cout << tagLive << "::" <<"Time to complete live feed features: "<<(double)(end1-start1)/ CLOCKS_PER_SEC<<"....secs."<<std::endl;
    return status;
}

cv::Mat foundationMatching_image;
int foundationMatching_landmarkPoints[150];
-(NSMutableArray*)performFoundationShadeMatch:(NSString *)inputPath landmarksArr:(NSMutableArray*)landmarksArr colors:(NSMutableArray*)colors skinLab:(NSString *)skinLab
{
    int noOfColors = (int)colors.count;
    int* foundationColors=new int[noOfColors];
    int i=0;
    for(i=0;i<noOfColors;i++)
    {
        foundationColors[i]= (int)[[colors objectAtIndex:i] integerValue];
    }
    foundationMatching.setFoundationShades(foundationColors,noOfColors);
    
    if(foundationColors!=NULL)
    {
        delete []foundationColors;
        foundationColors =NULL;
    }

    for (int i = 0; i < 150;i++)
    {
        foundationMatching_landmarkPoints[i]=(float)[[landmarksArr objectAtIndex:i] intValue];
    }
    std::string inputImagePath = [inputPath UTF8String];
    cv::Mat image = cv::imread(inputImagePath);
    double feature_resizeFactor = std::sqrt((double) (image.rows * image.cols) / (480 * 640));
    cv::resize(image, foundationMatching_image,cv::Size(std::round(image.cols / feature_resizeFactor),std::round(image.rows /feature_resizeFactor)));
    cv::cvtColor(foundationMatching_image, foundationMatching_image, cv::COLOR_BGR2RGB);
    const int recommandedShades=3;
    int matchedShadesIndex[recommandedShades];
    std::string skinLAB = [skinLab UTF8String];
    foundationMatching.colorMatchingProcess(foundationMatching_image, foundationMatching_landmarkPoints, matchedShadesIndex, skinLAB);
    NSMutableArray* matchedShadesArr = [NSMutableArray arrayWithCapacity:recommandedShades];
    for(int i =0;i<recommandedShades;i++)
    {
       [matchedShadesArr addObject:[NSNumber numberWithInt:(matchedShadesIndex[i])]];
    }
    return matchedShadesArr;
}

-(UIImage*) performFoundationSimulation:(int)index coverage:(CGFloat)coverage skinLab:(NSString *)skinLab {
    cv::Mat image=foundationMatching_image.clone();
    std::string skinLAB = [skinLab UTF8String];
    foundationMatching.foundation(image, foundationMatching_landmarkPoints, index, coverage, false, skinLAB);
    return MatToUIImage(image);
}

-(UIImage*)drawHairColor:(UIImage*)uiImage  uiMask:(UIImage*)uiMask color:(NSMutableArray*)hairColor coverage:(CGFloat) coverage {
    [self n_calculatingFPS];
    [self assigningLandmarkvariables];
    [fld_L n_GChannelLipsNeeded:true];
    
    int landmarkPoints[151];
    cv::Mat resizeImage;
    [fld_L landmarkdetectiononUIimage:uiImage arg1:resizeImage];
    //landmark coping
    //std::cout << tagLive << "::" << "Landmarks coping started"<<std::endl;
    [fld_L copinglandmarks:landmarkPoints];
    //std::cout << tagLive << "::" <<"Landmarks coping ended"<<std::endl;
    
    //std::cout<<tagLive<<"Started logging FLD Points::"<<std::endl;
    //for(int i=0; i<=150;i++)
    //{
    //    std::cout<<tagLive<<"drawHairColor FLD Points:: "<<landmarkPoints[i]<<std::endl;
    //}
    
    cv::Mat inputImage;
    UIImageToMat(uiImage, inputImage);
    if(inputImage.cols>inputImage.rows)
    {
        _StatusCode = 2012;
        //std::cout<< tagLive << "::" << "wrong frame"<<std::endl;
        return uiImage;
    }
    //NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    //NSString *documentsDirectory = [paths objectAtIndex:0];
    //std::string documentsPath = [documentsDirectory UTF8String];
    //cv::imwrite(documentsPath+"/"+"beforecolorcov.jpg",inputImage);
    cv::cvtColor(inputImage, inputImage, cv::COLOR_RGBA2RGB);
    //cv::imwrite(documentsPath+"/"+"aftercolorcov.jpg",inputImage);
    cv::Mat hairImage=inputImage(cv::Rect(0,(inputImage.rows-inputImage.cols)/2,inputImage.cols,inputImage.cols));
    if (landmarkPoints[150] == 1)
    {
        _StatusCode = 2012;
        //std::cout << tagLive << "::" << "Face is not detected.Please make yourself align with the camera and closer to camera,statuscode"<< _StatusCode<<std::endl;
    }
    else
    {
        _StatusCode=0;
        double feature_resizeFactor = std::sqrt((double) (inputImage.cols * inputImage.rows) / (480 * 640));
        for(int i=0;i<150;i++)
        {
            landmarkPoints[i]*=feature_resizeFactor;
        }
        cv::Mat maskImage;
        UIImageToMat(uiMask, maskImage);
        cv::Mat channels[4];
        cv::split(maskImage, channels);
        int color[3];
        for(int i=0;i<3;i++)
        {
            color[i]= (int)[[hairColor objectAtIndex:i] integerValue];
        }
        Skincare::HairFeatures hf;
        //cv::imwrite(documentsPath+"/"+"beforehair.jpg",hairImage);
        hf.hairColor(hairImage, channels[0], ((landmarkPoints[68*2+1]+landmarkPoints[74*2+1])/2-(inputImage.rows-inputImage.cols)/2)/((float)hairImage.cols/maskImage.cols), color, coverage);
        //cv::imwrite(documentsPath+"/"+"afterhair.jpg",hairImage);
    }
    if(n_isSplitFaceNeeded)
    {
        [self ApplySplitFace:landmarkPoints originalFrame:inputImage outputImage:hairImage];
    }
    
    UIImage *image=MatToUIImage(hairImage);
    return image;
}
- (void)setSplitFaceParams:(int)viewType params:(NSMutableArray *)params  isSplitNeeded:(bool)isSplitNeeded {
    
    n_viewType = viewType;
    
    for(int i=0;i< 3;i++)
    {
        n_params[i]=[[params objectAtIndex:i] doubleValue];
    }
    n_isSplitFaceNeeded = isSplitNeeded;
}


@end

