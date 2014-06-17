/*!
 * \file
 * \brief 
 * \author Mateusz Pruchniak
 */

#ifndef CLSIFT_HPP_
#define CLSIFT_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "Types/Features.hpp"

#include <opencv2/opencv.hpp>

#if (CV_MAJOR_VERSION == 2)
#if (CV_MINOR_VERSION > 3)
#include <opencv2/nonfree/features2d.hpp>
#endif
#endif


#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <ctype.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>


#define SIFT_MAX_NUMBER_KEYS		4000
#define DETECT_EXTREMA_OPENCL_KERNEL	"KernelDetectProcess"
#define DESC_EXTREMA_OPENCL_KERNEL	"KernelGenDescriptorProcess"
#define GPUBASE_OPENCL_SOURCE		"/home/mati/OpenCL/GPUCode.cl"
#define DETECT_EXTREMA_OPENCL_SOURCE	"/home/mati/OpenCL/DetectExtrema.cl"
#define SUBSTRACT_OPENCL_SOURCE		"/home/mati/OpenCL/Subtract.cl"
#define BLUR_OPENCL_SOURCE		"/home/mati/OpenCL/BlurGaussFilter.cl"
#define SIFT_MAX_NUMBER_KEYS		4000
#define MAX_KERNEL_SIZE			20
#define SIFT_MAX_NUMBER_KEYS		4000

#define SIFT_INTVLS			3
#define SIFT_SIGMA			1.6
#define SIFT_CONTR_THR			0.04
#define SIFT_CURV_THR			10
#define SIFT_IMG_DBL			1
#define SIFT_DESCR_WIDTH 		4
#define SIFT_DESCR_HIST_BINS 		8
#define SIFT_INIT_SIGMA 		0.5
#define SIFT_IMG_BORDER 		5
#define SIFT_MAX_INTERP_STEPS 		5
#define SIFT_ORI_HIST_BINS 		36
#define SIFT_ORI_SIG_FCTR 		1.5
#define SIFT_ORI_RADIUS 		3.0 * SIFT_ORI_SIG_FCTR
#define SIFT_ORI_SMOOTH_PASSES 		2
#define SIFT_ORI_PEAK_RATIO 		0.8
#define SIFT_DESCR_SCL_FCTR 		3.0
#define SIFT_DESCR_MAG_THR 		0.2
#define SIFT_INT_DESCR_FCTR 		512.0
#define FeatDetectionData(f) 		( (struct detection_data*)(f->feature_data) )
#define ROUND(x) 			( ( x - (int)x ) <= 0.5 ? (int)x :  (int)x + 1 )
#define FEATURE_OXFD_COLOR 		CV_RGB(255,255,0)
#define FEATURE_LOWE_COLOR 		CV_RGB(255,0,255)
#define FEATURE_MAX_D 			128
#define	SIFTCPU				0



namespace Processors {
namespace ClSIFT {

using namespace cv;

double SendTime;
double RecvTime;



typedef struct
{
	float	scx;
	float	scy;
	float	x;			/**< x coord */
	float	y;			/**< y coord */
	float	subintvl;
	float	intvl;	
	float	octv;
	float	scl;
	float	scl_octv;	/**< scale of a Lowe-style feature */
	float	ori;		/**< orientation of a Lowe-style feature */
	float	mag;
	float	desc[128];
} Keys;


struct detection_data
{
	int r;
	int c;
	int octv;
	int intvl;
	float subintvl;
	float scl_octv;
};



typedef struct feature
{
	double x;                    
	double y;                    
	double a;                    
	double b;                   
	double c;                      
	double scl;                  
	double ori;                   
	int d;                  
	double descr[FEATURE_MAX_D];  
	int type;                   
	int category;                 
	struct feature* fwd_match;  
	CvPoint2D64f img_pt;          
	CvPoint2D64f mdl_pt;           
	void* feature_data;        
} feature;


enum feature_type
{
	FEATURE_LOWE
};

class GPU
{
  private:

	cl_device_id device;
	
	
	cl_int GPUError;

	
	/*!
	 * Platforms are represented by a cl_platform_id, OpenCL framework allow an application to share resources and execute kernels on devices in the platform.
	 */
	cl_platform_id cpPlatform;

	GPU() 
	{
		printf("\n ----------- SINGLETON START --------------- \n");
		// Fetch the Platform and Device IDs; we only want one.
		cl_uint platforms;
		cl_uint devices;

		GPUError = clGetPlatformIDs(1, &cpPlatform, &platforms);
		if (GPUError != CL_SUCCESS) {
				printf("\n Error number %d", GPUError);
		}

		GPUError = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device, &devices);

		cl_context_properties properties[]={
		CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
		0};

		// Note that nVidia's OpenCL requires the platform property
		GPUContext = clCreateContext(properties, 1, &device, NULL, NULL, &GPUError);
		GPUCommandQueue = clCreateCommandQueue(GPUContext, device, 0, &GPUError);
		if (GPUError != CL_SUCCESS) {
				printf("\n Error number %d", GPUError);
		}
		
		numberOfBuffersIn  = 0;
		numberOfBuffersOut = 0;
		
		printf("\n ----------- SINGLETON END --------------- \n");
	}
	
	

	GPU(const GPU&);
	

  public:

	/*!
	 * OpenCL command-queue, is an object where OpenCL commands are enqueued to be executed by the device.
	 * "The command-queue is created on a specific device in a context [...] Having multiple command-queues allows applications to queue multiple independent commands without requiring synchronization." (OpenCL Specification).
	 */
	cl_command_queue GPUCommandQueue; 
	
	/*!
	 * Context defines the entire OpenCL environment, including OpenCL kernels, devices, memory management, command-queues, etc. Contexts in OpenCL are referenced by an cl_context object
	 */
	cl_context GPUContext; 
	
	cl_mem* buffersListIn;

	int numberOfBuffersIn;

	cl_mem* buffersListOut;

	int numberOfBuffersOut;
	
	bool CreateBuffersIn(int maxBufferSize, int numberOfBuffers);

	bool CreateBuffersOut(int maxBufferSize, int numberOfBuffers);

	cl_mem CreateBuffer(int size);

	cl_kernel CreateKernel(const char* kernel, cl_program GPUProgram);
	
	static GPU& getInstance()
	{
	  static GPU instance;
	  return instance;
	}
};



class GPUBase
{
	public:
		
	
	

	/*!
	 * Kernels are essentially functions that we can call from the host and that will run on the device
	 */
	cl_kernel GPUKernel;

	/*!
	 * Error code, only 0 is allowed.
	 */
	cl_int GPUError;	

	/*!
	 * Program is formed by a set of kernels, functions and declarations, and it's represented by an cl_program object.
	 */
	cl_program GPUProgram;
	  

	/*!
	 * Check error code.
	 */
	void CheckError(int code);

	

	/*!
	 * Work-group size - dim X.
	 */
	int iBlockDimX;                    

	/*!
	 * Work-group size - dim Y.
	 */
	int iBlockDimY;  

	/*!
	 * Image width.
	 */
	unsigned int imageWidth;   

	/*!
	 * Image height.
	 */
	unsigned int imageHeight; 

	/*!
	 * Global size of NDRange.
	 */
	size_t GPUGlobalWorkSize[2];

	char* kernelName;
	
	~GPUBase();

	GPUBase(char* source, char* KernelName);

	bool CreateBuffersIn(int maxBufferSize, int numberOfBuffers);

	bool CreateBuffersOut(int maxBufferSize, int numberOfBuffers);

	cl_mem CreateBuffer(int size);

	cl_kernel CreateKernel(const char* kernel, cl_program GPUProgram);

	bool SendImageToBuffers(int number, ... );

	bool ReceiveImageFromBuffers(int number, ... );

	size_t RoundUpGroupDim(int group_size, int global_size);

	char* LoadProgramSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

	
};


class PyramidProcess :
	public GPUBase
{

public:

	cl_mem cmBufPyramid;

	bool CreateBufferForPyramid(float size);

	bool ReleaseBufferForPyramid();

	bool ReceiveImageFromPyramid( IplImage* img, int offset);

	bool SendImageToPyramid(IplImage* img, int offset);

	/*!
	* Destructor.
	*/
	~PyramidProcess(void);

	/*!
	* Constructor.
	*/
	PyramidProcess(char* source, char* KernelName);
};

class Subtract :
	public PyramidProcess
{
public:

	bool Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct);

	/*!
	* Destructor.
	*/
	~Subtract(void);

	/*!
	* Constructor.
	*/
	Subtract();
};



class GaussFilter :
	public PyramidProcess
{

	int GetGaussKernelSize(double sigma, double cut_off=0.001);

public:

	/*!
	* Destructor.
	*/
	~GaussFilter(void);

	/*!
	* Constructor.
	*/
	GaussFilter();

	bool Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext);

	

};



class DetectExtrema :
	public GPUBase
{
private:

	cl_mem cmDevBufNumber;

	//int maxNumberKeys;

	cl_mem cmDevBufKeys;

	cl_mem cmDevBufCount;

	cl_kernel GPUKernelDesc;

public:

	/*!
	* Destructor.
	*/
	~DetectExtrema(void);

	/*!
	* Constructor.
	*/
	DetectExtrema();

	bool Process(cl_mem dogPyr, cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct, int OffsetNext,int* numExtr, float prelim_contrastThreshold, int intvl, int octv, Keys* keys);
};



class SiftGPU
{
private:
	/*-------------------------*/

	float contrastThreshold;

	int curvaturesThreshold;

	GaussFilter* gaussFilterGPU;
	
	Subtract* subtractGPU;
	
	DetectExtrema* detectExtremaGPU;

	int* imageWidthInPyramid;

	int* imageHeightInPyramid;

	IplImage** imgArray;

	int intvls;

	int octvs;

	float* sigmaList;

	int* sizeOfImages;

	int SizeOfPyramid;


	CvMemStorage* storage;
	
	int total;

	IplImage* CreateInitialImg( IplImage* img, int img_dbl, float sigma );
	IplImage* ConvertToGray32( IplImage* img );
	
	bool BuildGaussPyramid(IplImage* base);
	IplImage* Downsample( IplImage* img );
	int DetectAndGenerateDesc();
	feature* NewDesc( void );


public:

    	cv::Mat descriptors;
    	vector<cv::KeyPoint> keypoints; 

	SiftGPU(int _intvls, float _contrastThreshold, int _curvaturesThreshold);


	int DoSift(IplImage* img);

};




/*!
 * \class ClSIFT
 * \brief ClSIFT processor class.
 *
 * ClSIFT processor.
 */
class ClSIFT: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	ClSIFT(const std::string & name = "ClSIFT");

	/*!
	 * Destructor
	 */
	virtual ~ClSIFT();

	/*!
	 * Prepare components interface (register streams and handlers).
	 * At this point, all properties are already initialized and loaded to 
	 * values set in config file.
	 */
	void prepareInterface();

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();

	/*!
	 * Event handler function.
	 */
	void onNewImage();

	/// Event handler.
	Base::EventHandler <ClSIFT> h_onNewImage;

	/// Input data stream
	Base::DataStreamIn <Mat> in_img;

	/// Output data stream containing extracted features
	Base::DataStreamOut <Types::Features> out_features;

	/// Output data stream containing feature descriptors
	Base::DataStreamOut <cv::Mat> out_descriptors;

	SiftGPU* siftOpenCL;

	feature* features;

};






} //: namespace ClSIFT
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("ClSIFT", Processors::ClSIFT::ClSIFT)

#endif /* CLSIFT_HPP_ */

