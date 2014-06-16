/*!
 * \file
 * \brief
 * \author Mateusz Pruchniak
 */

#include <memory>
#include <string>

#include "ClSIFT.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#if CV_MAJOR_VERSION == 2
#if CV_MINOR_VERSION > 3
#include <opencv2/nonfree/features2d.hpp>
#endif
#elif CV_MAJOR_VERSION == 3
#include <opencv2/nonfree/features2d.hpp>
#endif

namespace Processors {
namespace ClSIFT {

ClSIFT::ClSIFT(const std::string & name) :
		Base::Component(name)  {

	siftOpenCL = new SiftGPU(SIFT_INTVLS, SIFT_CONTR_THR,SIFT_CURV_THR);


}

ClSIFT::~ClSIFT() {
}

void ClSIFT::prepareInterface() {
	// Register handlers with their dependencies.
	h_onNewImage.setup(this, &ClSIFT::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);
	addDependency("onNewImage", &in_img);

	// Input and output data streams.
	registerStream("in_img", &in_img);
	registerStream("out_features", &out_features);
	registerStream("out_descriptors", &out_descriptors);
}

bool ClSIFT::onInit() {

	return true;
}

bool ClSIFT::onFinish() {
	return true;
}

bool ClSIFT::onStop() {
	return true;
}

bool ClSIFT::onStart() {
	return true;
}

void ClSIFT::onNewImage()
{
	LOG(LTRACE) << "ClSIFT::onNewImage\n";
	try {
		// Input: a grayscale image.
		cv::Mat gray = in_img.read();
		//cv::Mat gray;
		//cvtColor(input, gray, COLOR_BGR2GRAY);

		cout << "--- START DoSift --- mat count of rows " <<  gray.rows << endl;
		IplImage* img = new IplImage(gray);
		
		cout << "--- START DoSift --- ipl count of rows " <<  img->height << endl;

		int n1 = siftOpenCL->DoSift(img);
		features = siftOpenCL->listOfPoints;
		
		cout << "!!! Detect " << n1 << " features" << endl;

		//-- Step 1: Detect the keypoints.
	    	cv::SiftFeatureDetector detector;
	    	std::vector<cv::KeyPoint> keypoints;
	    	//detector.detect(gray, keypoints);

		//-- Step 2: Calculate descriptors (feature vectors).
		//cv::SiftDescriptorExtractor extractor;
		Mat descriptors;
		//extractor.compute( gray, keypoints, descriptors);

		// Write results to outputs.
	    	Types::Features features(keypoints);
		out_features.write(features);
		out_descriptors.write(descriptors);
	} catch (...) {
		LOG(LERROR) << "ClSIFT::onNewImage failed\n";
	}
}




GPUBase::GPUBase(char* source, char* KernelName)
{
	
	printf("\n ----------- GPUBase START --------------- \n");
	kernelName = KernelName;
	size_t szKernelLength = 0;
	size_t szKernelLengthFilter = 0;
	size_t szKernelLengthSum = 0;
	char* SourceOpenCLShared;
	char* SourceOpenCL;
	iBlockDimX = 16;
	iBlockDimY = 16;

	
		// Load OpenCL kernel
	SourceOpenCLShared = LoadProgramSource(GPUBASE_OPENCL_SOURCE, "", &szKernelLength);

	printf("\n ----------- Load GPUBase CODE --------------- \n");

	SourceOpenCL = LoadProgramSource(source, "// My comment\n", &szKernelLengthFilter);
	
	printf("\n ----------- Load SOURCE CONSTR --------------- \n");

	szKernelLengthSum = szKernelLength + szKernelLengthFilter + 100;
	char* sourceCL = new char[szKernelLengthSum];
	
	
	strcpy(sourceCL,SourceOpenCLShared);
	strcat (sourceCL, SourceOpenCL);
	
	
	GPUProgram = clCreateProgramWithSource( GPU::getInstance().GPUContext , 1, (const char **)&sourceCL, &szKernelLengthSum, &GPUError);
	CheckError(GPUError);

	printf("\n ----------- AFTER clCreateProgramWithSource --------------- \n");


	// Build the program with 'mad' Optimization option
	char *flags = "-cl-unsafe-math-optimizations";

	GPUError = clBuildProgram(GPUProgram, 0, NULL, flags, NULL, NULL);
	CheckError(GPUError);
	
	GPUKernel = clCreateKernel(GPUProgram, kernelName, &GPUError);
	CheckError(GPUError);

	printf("\n ----------- GPUBase END --------------- \n");

}


bool GPUBase::CreateBuffersIn(int maxBufferSize, int numbOfBuffers)
{
	GPU::getInstance().CreateBuffersIn(maxBufferSize,numbOfBuffers);

	return true;
}

bool GPUBase::CreateBuffersOut( int maxBufferSize, int numbOfBuffers)
{
	GPU::getInstance().CreateBuffersOut(maxBufferSize,numbOfBuffers);
	
	return true;
}

cl_mem GPUBase::CreateBuffer(int size)
{
	return GPU::getInstance().CreateBuffer(size);
}

cl_kernel GPUBase::CreateKernel(const char* kernel, cl_program GPUProgram)
{
	return GPU::getInstance().CreateKernel(kernel, GPUProgram);
}

bool GPU::CreateBuffersIn(int maxBufferSize, int numbOfBuffers)
{
	if(numberOfBuffersIn > 0)
	{
		for(int i = 0 ; i < numberOfBuffersIn ; i++)
		{
			if(buffersListIn[i])clReleaseMemObject(buffersListIn[i]);
		}
	}

	numberOfBuffersIn = numbOfBuffers;
	buffersListIn = new cl_mem[numberOfBuffersIn];
	
	for (int i = 0; i < numberOfBuffersIn ; i++)
	{
		buffersListIn[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
	}
	return true;
}



bool GPU::CreateBuffersOut( int maxBufferSize, int numbOfBuffers)
{
	if(numberOfBuffersOut > 0)
	{
		for(int i = 0 ; i < numberOfBuffersOut ; i++)
		{
			if(buffersListOut[i])clReleaseMemObject(buffersListOut[i]);
		}
	}
	
	numberOfBuffersOut = numbOfBuffers;
	buffersListOut = new cl_mem[numberOfBuffersOut];
	
	for (int i = 0; i < numberOfBuffersOut ; i++)
	{
		buffersListOut[i] = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, maxBufferSize, NULL, &GPUError);
	}
	return true;
}

cl_mem GPU::CreateBuffer(int size)
{
	cl_mem buf = clCreateBuffer(GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	if (GPUError != CL_SUCCESS) {
			return NULL;
	}
	return buf;
}


cl_kernel GPU::CreateKernel(const char* kernel, cl_program GPUProgram)
{
	cl_kernel GPUKernel = clCreateKernel(GPUProgram, DESC_EXTREMA_OPENCL_KERNEL, &GPUError);
	if (GPUError != CL_SUCCESS) {
			return NULL;
	}
	return GPUKernel;
}


void GPUBase::CheckError( int code )
{
	switch(code)
	{
	case CL_SUCCESS:
		return;
		break;
	default:
		cout << "ERROR : " << code << endl;
	}
}


bool GPUBase::SendImageToBuffers(int number, ... )
{
	if(GPU::getInstance().buffersListIn == NULL)
		return false;


	va_list arg_ptr;
	va_start(arg_ptr, number);
	
	clock_t start, finish;
	double duration = 0;
	start = clock();
	
	for(int i = 0 ; i < number ; i++)
	{
		IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
		imageHeight = tmpImg->height;
		imageWidth = tmpImg->width;
		GPUError = clEnqueueWriteBuffer(GPU::getInstance().GPUCommandQueue, GPU::getInstance().buffersListIn[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
		CheckError(GPUError);
	}

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	SendTime += duration;

	va_end(arg_ptr);
}


bool GPUBase::ReceiveImageFromBuffers(int number, ... )
{
	if(GPU::getInstance().buffersListOut == NULL)
		return false;

	va_list arg_ptr;
	va_start(arg_ptr, number);

	clock_t start, finish;
	double duration = 0;
	start = clock();

	for(int i = 0 ; i < number ; i++)
	{
		IplImage* tmpImg = va_arg(arg_ptr, IplImage*);
		GPUError = clEnqueueReadBuffer(GPU::getInstance().GPUCommandQueue, GPU::getInstance().buffersListOut[i], CL_TRUE, 0, tmpImg->width*tmpImg->height*sizeof(float) , (void*)tmpImg->imageData, 0, NULL, NULL);
		CheckError(GPUError);
	}

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	RecvTime += duration;

	va_end(arg_ptr);
}


size_t GPUBase::RoundUpGroupDim(int group_size, int global_size)
{
	if(global_size < 80)
		global_size = 80;
	int r = global_size % group_size;
	if(r == 0)
	{
		return global_size;
	} else
	{
		return global_size + group_size - r;
	}
}

char* GPUBase::LoadProgramSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength)
{
	// locals
	FILE* pFileStream = NULL;
	size_t szSourceLength;


	pFileStream = fopen(cFilename, "rb");
	if(pFileStream == 0)
	{
		return NULL;
	}
	size_t szPreambleLength = strlen(cPreamble);

	// get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);

	// allocate a buffer for the source code string and read it in
	char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
	memcpy(cSourceString, cPreamble, szPreambleLength);
	if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
	{
		fclose(pFileStream);
		free(cSourceString);
		return 0;
	}

	// close the file and return the total length of the combined (preamble + source) string
	fclose(pFileStream);
	if(szFinalLength != 0)
	{
		*szFinalLength = szSourceLength + szPreambleLength;
	}
	cSourceString[szSourceLength + szPreambleLength] = '\0';

	return cSourceString;
}

GPUBase::~GPUBase()
{
	if(GPU::getInstance().GPUCommandQueue)clReleaseCommandQueue(GPU::getInstance().GPUCommandQueue);
	if(GPU::getInstance().GPUContext)clReleaseContext(GPU::getInstance().GPUContext);

	for(int i = 0 ; i<GPU::getInstance().numberOfBuffersOut ; i++)
	{
		if(GPU::getInstance().buffersListOut[i])clReleaseMemObject(GPU::getInstance().buffersListOut[i]);
	}

	for(int i = 0 ; i<GPU::getInstance().numberOfBuffersIn ; i++)
	{
		if(GPU::getInstance().buffersListIn[i])clReleaseMemObject(GPU::getInstance().buffersListIn[i]);
	}
}



PyramidProcess::PyramidProcess(char* source, char* KernelName) : GPUBase(source,KernelName)
{

}


PyramidProcess::~PyramidProcess(void)
{

}

bool PyramidProcess::CreateBufferForPyramid( float size )
{
	cmBufPyramid = clCreateBuffer(GPU::getInstance().GPUContext, CL_MEM_READ_WRITE, size, NULL, &GPUError);
	CheckError(GPUError);
	return true;
}


bool PyramidProcess::ReceiveImageFromPyramid( IplImage* img, int offset)
{
	clock_t start, finish;
	double duration = 0;
	start = clock();
	GPUError = clEnqueueReadBuffer(GPU::getInstance().GPUCommandQueue, cmBufPyramid, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	RecvTime += duration;

	return true;
}

bool PyramidProcess::SendImageToPyramid( IplImage* img, int offset)
{
	clock_t start, finish;
	double duration = 0;
	start = clock();
	GPUError = clEnqueueWriteBuffer(GPU::getInstance().GPUCommandQueue, cmBufPyramid, CL_TRUE, offset, img->imageSize, (void*)img->imageData, 0, NULL, NULL);
	CheckError(GPUError);
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	SendTime += duration;

	return true;
}



Subtract::Subtract(): PyramidProcess(SUBSTRACT_OPENCL_SOURCE,"KernelSubtractProcess")
{

}


Subtract::~Subtract(void)
{
}


bool Subtract::Process(cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev, int OffsetAct)
{

	OffsetAct = OffsetAct / 4;
	OffsetPrev = OffsetPrev / 4;

	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&gaussPyr);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&OffsetPrev);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_uint), (void*)&imageHeight);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPU::getInstance().GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}


int FeatureCmp( void* feat1, void* feat2, void* param )
 {
	 feature* f1 = (feature*) feat1;
	 feature* f2 = (feature*) feat2;

	 if( f1->scl < f2->scl )
		 return 1;
	 if( f1->scl > f2->scl )
		 return -1;
	 return 0;
 }


GaussFilter::~GaussFilter(void)
{
}

GaussFilter::GaussFilter(): PyramidProcess(BLUR_OPENCL_SOURCE,"KernelGaussProcess")
{

}

bool GaussFilter::Process(float sigma, int imageWidth, int imageHeight, int OffsetAct, int OffsetNext)
{

	OffsetAct = OffsetAct / 4;
	OffsetNext = OffsetNext / 4;

	int maskSize = 0;
	maskSize = cvRound(sigma * 3.0 * 2.0 + 1.0) | 1;


	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)imageHeight);
	
	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&cmBufPyramid);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_uint), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_uint), (void*)&OffsetNext);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_uint), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_uint), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_float), (void*)&sigma);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_uint), (void*)&maskSize);

	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPU::getInstance().GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) return false;
	return true;
}


int GaussFilter::GetGaussKernelSize(double sigma, double cut_off)
{
	unsigned int i;
	for (i=0;i<MAX_KERNEL_SIZE;i++)
		if (exp(-((double)(i*i))/(2.0*sigma*sigma))<cut_off)
			break;
	unsigned int size = 2*i-1;
	return size;
}






 SiftGPU::SiftGPU(int _intvls, float _contrastThreshold, int _curvaturesThreshold)
 {
	 intvls = _intvls;
	 contrastThreshold = _contrastThreshold;
	 curvaturesThreshold = _curvaturesThreshold;
	 total = 0;
	 SizeOfPyramid = 0;

	 sigmaList = (float*)calloc( intvls + 3, sizeof(float));

	 gaussFilterGPU = new GaussFilter();
	 subtractGPU = new Subtract();
	 detectExtremaGPU = new DetectExtrema();
 }



 int SiftGPU::DoSift( IplImage* img )
 {
	printf("\n ----------- DoSift inside --------------- \n");
	
	IplImage* init_img;
	CvSeq* features;

	
	init_img = CreateInitialImg( img, SIFT_IMG_DBL, SIFT_SIGMA );
	octvs = log( (float)MIN( init_img->width, init_img->height ) ) / log((float)2) - 2;
	sizeOfImages = new int[octvs];
	imageWidthInPyramid = new int[octvs];
	imageHeightInPyramid = new int[octvs];

	BuildGaussPyramid(init_img);
	storage = cvCreateMemStorage( 0 );
	
	clock_t start, finish;
	double duration = 0;
	start = clock();
		features = DetectAndGenerateDesc();
	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << endl;
	cout << "ScaleSpaceExtrema " << SIFTCPU << ": " << duration << endl;
	cout << endl;
	
	cvSeqSort( features, (CvCmpFunc)FeatureCmp, NULL );
	total = features->total;
	listOfPoints = (feature*)calloc(total, sizeof(feature));
	listOfPoints = (feature*)cvCvtSeqToArray( features, listOfPoints, CV_WHOLE_SEQ );
	for(int i = 0; i < total; i++ )
	{
		free( listOfPoints[i].feature_data );
		listOfPoints[i].feature_data = NULL;
	}

	cvReleaseMemStorage( &storage );
	cvReleaseImage( &init_img );



	return total;
 }



 
 bool SiftGPU::BuildGaussPyramid(IplImage* base)
 {
	float k;
	int intvlsSum = intvls + 3;
	float sig_total, sig_prev;

	cout << "Start BuildGaussPyramid height of img " << base->height << endl;

	imgArray = (IplImage**)calloc(octvs, sizeof(IplImage*));

	sigmaList[0] = SIFT_SIGMA;
	k = pow( 2.0, 1.0 / intvls );


	for(int i = 1; i < intvlsSum; i++ )
	{
		sig_prev = pow( k, i - 1 ) * SIFT_SIGMA;
		sig_total = sig_prev * k;
		sigmaList[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );
	}

	imgArray[0] = cvCloneImage(base);
	
	sizeOfImages[0] = imgArray[0]->imageSize;
	SizeOfPyramid += imgArray[0]->imageSize * intvlsSum;
	imageHeightInPyramid[0] = imgArray[0]->height;
	imageWidthInPyramid[0] = imgArray[0]->width;

	for(int o = 1; o < octvs; o++ )
	{
		imgArray[o] = Downsample( imgArray[o-1] );
		SizeOfPyramid += imgArray[o]->imageSize * intvlsSum;
		sizeOfImages[o] = imgArray[o]->imageSize;
		imageHeightInPyramid[o] = imgArray[o]->height;
		imageWidthInPyramid[o] = imgArray[o]->width;
	}


	gaussFilterGPU->CreateBufferForPyramid(SizeOfPyramid);
	subtractGPU->CreateBufferForPyramid(SizeOfPyramid);


	int offset = 0;

	offset = 0;
	int OffsetAct = 0;
	int OffsetPrev = 0;

	for(int o = 0; o < octvs; o++ )
	{
		for(int i = 0; i < intvlsSum; i++ )
		{

			if( o == 0  &&  i == 0 )
			{
				gaussFilterGPU->SendImageToPyramid(imgArray[o], OffsetAct);
			} else if(i == 0)
			{
				gaussFilterGPU->ReceiveImageFromPyramid(imgArray[o-1], OffsetPrev);
				imgArray[o] = Downsample( imgArray[o-1] );
				gaussFilterGPU->SendImageToPyramid(imgArray[o], OffsetAct);
			}

			if(i > 0 )
			{
				gaussFilterGPU->Process( sigmaList[i], imgArray[o]->width, imgArray[o]->height, OffsetPrev, OffsetAct);
				subtractGPU->Process(gaussFilterGPU->cmBufPyramid, imageWidthInPyramid[o], imageHeightInPyramid[o], OffsetPrev, OffsetAct);
			}
			OffsetPrev = OffsetAct;
			OffsetAct += sizeOfImages[o];
		}
	}



	//free( sigmaList );	
	return true;
}


 /*
 Downsamples an image to a quarter of its size (half in each dimension)
 using nearest-neighbor interpolation

 @param img an image

 @return Returns an image whose dimensions are half those of img
 */
 IplImage* SiftGPU::Downsample( IplImage* img )
 {
	 int width = img->width / 2;
	 int height = img->height / 2;

	 if( width < 50 || height < 50 )
	 {
		 width = width*2;
		 height = height*2;
	 }
	 IplImage* smaller = cvCreateImage( cvSize( width, height),
		 img->depth, img->nChannels );
	 cvResize( img, smaller, CV_INTER_NN );

	 return smaller;
 }



 CvSeq* SiftGPU::DetectAndGenerateDesc()
{
	float prelim_contrastThreshold = 0.5 * contrastThreshold / intvls;
	struct detection_data* ddata;
	int o, i, r, c;
	int num=0;				// Number of keypoins detected
	int numRemoved=0;		// The number of key points rejected because they failed a test
	int numberExtrema = 0;
	int number = 0;

	CvSeq* features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(feature), storage );
	total = features->total;
	int intvlsSum = intvls + 3;
	int OffsetAct = 0;
	int OffsetNext = 0;
	int OffsetPrev = 0;

	Keys keysArray[SIFT_MAX_NUMBER_KEYS];
	/*for (int j = 0 ; j < SIFT_MAX_NUMBER_KEYS ; j++)
	{
		keysArray[j].x = 0.0;
		keysArray[j].y = 0.0;
		keysArray[j].intvl = 0.0;
		keysArray[j].octv = 0.0;
		keysArray[j].subintvl = 0.0;
		keysArray[j].scx = 0.0;
		keysArray[j].scy = 0.0;
		keysArray[j].ori = 0.0;
	}*/


	for( o = 0; o < octvs; o++ )
	{
		for( i = 0; i < intvlsSum; i++ )
		{
			OffsetNext += sizeOfImages[o];
				
			if( i > 0 && i <= intvls )
			{
				num = 0;
				detectExtremaGPU->Process(subtractGPU->cmBufPyramid, gaussFilterGPU->cmBufPyramid, imageWidthInPyramid[o], imageHeightInPyramid[o], OffsetPrev, OffsetAct, OffsetNext, &num, prelim_contrastThreshold, i, o, keysArray);
				total = features->total;
				number = num;
				struct detection_data* ddata;

				for(int ik = 0; ik < number ; ik++)
				{ 
					cv::KeyPoint* key = new cv::KeyPoint(keysArray[ik].scx, keysArray[ik].scy, 1);
					key->octave = keysArray[ik].octv;
					key->angle = keysArray[ik].ori;
					


					listOfPoints = NewDesc();
					ddata = FeatDetectionData( listOfPoints );
					listOfPoints->img_pt.x = listOfPoints->x = keysArray[ik].scx;
					listOfPoints->img_pt.y = listOfPoints->y = keysArray[ik].scy;
					ddata->r = keysArray[ik].y;
					ddata->c = keysArray[ik].x;
					ddata->subintvl = keysArray[ik].subintvl;
					ddata->octv = keysArray[ik].octv;
					ddata->intvl = keysArray[ik].intvl;
					listOfPoints->scl = keysArray[ik].scl;
					ddata->scl_octv = keysArray[ik].scl_octv;
					listOfPoints->ori = (double)keysArray[ik].ori;
					listOfPoints->d = 128;
					for(int i = 0; i < 128 ; i++ )
					{
						listOfPoints->descr[i] = keysArray[ik].desc[i];
					}
					cvSeqPush( features, listOfPoints );
					free( listOfPoints );
				}
			}
			OffsetPrev = OffsetAct;
			OffsetAct += sizeOfImages[o];
		}
	}

	return features;
}



 feature* SiftGPU::NewDesc()
 {
	feature* listOfPoints;
	struct detection_data* ddata;

	listOfPoints = (feature*)malloc( sizeof( feature ) );
	memset( listOfPoints, 0, sizeof( feature ) );
	ddata = (detection_data*)malloc( sizeof( struct detection_data ) );
	memset( ddata, 0, sizeof( struct detection_data ) );
	listOfPoints->feature_data = ddata;
	listOfPoints->type = FEATURE_LOWE;

	return listOfPoints;
}



 IplImage* SiftGPU::CreateInitialImg( IplImage* img, int img_dbl, float sigma )
 {
	 IplImage* gray, * dbl;
	 float sig_diff;

	 gray = ConvertToGray32( img );
	 if( img_dbl )
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
		 dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ), 32, 1 );

		 cvResize( gray, dbl, CV_INTER_CUBIC );

		 cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );

		 cvReleaseImage( &gray );
		 return dbl;
	 }
	 else
	 {
		 sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );

		 cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
		 
		 return gray;
	 }
 }



 IplImage* SiftGPU::ConvertToGray32( IplImage* img )
 {
	 IplImage* gray8, * gray32;

	 gray32 = cvCreateImage( cvGetSize(img), 32, 1 );
	 if( img->nChannels == 1 )
		 gray8 = (IplImage*)cvClone( img );
	 else
	 {
		 gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
		 cvCvtColor( img, gray8, CV_BGR2GRAY );
	 }
	 cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

	 cvReleaseImage( &gray8 );
	 return gray32;
 }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           




DetectExtrema::~DetectExtrema(void)
{
}

DetectExtrema::DetectExtrema(): GPUBase( DETECT_EXTREMA_OPENCL_SOURCE, DETECT_EXTREMA_OPENCL_KERNEL)
{
	int counter = 0;
	int maxNumberKeys = SIFT_MAX_NUMBER_KEYS;

	cmDevBufNumber = clCreateBuffer(GPU::getInstance().GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);

	cmDevBufCount = clCreateBuffer(GPU::getInstance().GPUContext, CL_MEM_READ_WRITE, sizeof(int), NULL, &GPUError);
	CheckError(GPUError);
	GPUError = clEnqueueWriteBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&counter, 0, NULL, NULL);
	CheckError(GPUError);

	cmDevBufKeys = clCreateBuffer(GPU::getInstance().GPUContext, CL_MEM_READ_WRITE, maxNumberKeys*sizeof(Keys), NULL, &GPUError);
	CheckError(GPUError);

	GPUKernelDesc = clCreateKernel(GPUProgram, DESC_EXTREMA_OPENCL_KERNEL, &GPUError);
	CheckError(GPUError);
}


bool DetectExtrema::Process(cl_mem dogPyr, cl_mem gaussPyr, int imageWidth, int imageHeight, int OffsetPrev , int OffsetAct, int OffsetNext, int* numExtr, float prelim_contrastThreshold, int intvl, int octv, Keys* keys)
{
	int counter = 0;
	int numberExtr = 0;

	OffsetAct = OffsetAct / 4;
	OffsetPrev = OffsetPrev / 4;
	OffsetNext = OffsetNext / 4;
	
	GPUError = clEnqueueWriteBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)&numberExtr, 0, NULL, NULL);
	CheckError(GPUError);

	size_t GPULocalWorkSize[2];
	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)imageWidth);
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)imageHeight);

	int iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernel, 0, sizeof(cl_mem), (void*)&dogPyr);
	GPUError |= clSetKernelArg(GPUKernel, 1, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernel, 2, sizeof(cl_mem), (void*)&cmDevBufNumber);
	GPUError |= clSetKernelArg(GPUKernel, 3, sizeof(cl_int), (void*)&OffsetPrev);
	GPUError |= clSetKernelArg(GPUKernel, 4, sizeof(cl_int), (void*)&OffsetAct);
	GPUError |= clSetKernelArg(GPUKernel, 5, sizeof(cl_int), (void*)&OffsetNext);
	GPUError |= clSetKernelArg(GPUKernel, 6, sizeof(cl_int), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernel, 7, sizeof(cl_int), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernel, 8, sizeof(cl_float), (void*)&prelim_contrastThreshold);
	GPUError |= clSetKernelArg(GPUKernel, 9, sizeof(cl_int), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernel, 10, sizeof(cl_int), (void*)&octv);
	
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPU::getInstance().GPUCommandQueue, GPUKernel, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL))
	{
		cout << "Error clEnqueueNDRangeKernel" << endl;
		return false;
	}

	GPUError = clEnqueueReadBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufNumber, CL_TRUE, 0, sizeof(int), (void*)&numberExtr, 0, NULL, NULL);
	CheckError(GPUError);

	GPUError = clEnqueueWriteBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&counter, 0, NULL, NULL);
	CheckError(GPUError);

	float sqrtNuber = sqrt((float)numberExtr);

	GPULocalWorkSize[0] = iBlockDimX;
	GPULocalWorkSize[1] = iBlockDimY;
	GPUGlobalWorkSize[0] = RoundUpGroupDim((int)GPULocalWorkSize[0], (int)(sqrtNuber+1));
	GPUGlobalWorkSize[1] = RoundUpGroupDim((int)GPULocalWorkSize[1], (int)sqrtNuber);

	iLocalPixPitch = iBlockDimX + 2;
	GPUError = clSetKernelArg(GPUKernelDesc, 0, sizeof(cl_mem), (void*)&gaussPyr);
	GPUError |= clSetKernelArg(GPUKernelDesc, 1, sizeof(cl_uint), (void*)&OffsetPrev);
	GPUError |= clSetKernelArg(GPUKernelDesc, 2, sizeof(cl_mem), (void*)&cmDevBufCount);
	GPUError |= clSetKernelArg(GPUKernelDesc, 3, sizeof(cl_mem), (void*)&cmDevBufKeys);
	GPUError |= clSetKernelArg(GPUKernelDesc, 4, sizeof(cl_int), (void*)&imageWidth);
	GPUError |= clSetKernelArg(GPUKernelDesc, 5, sizeof(cl_int), (void*)&imageHeight);
	GPUError |= clSetKernelArg(GPUKernelDesc, 6, sizeof(cl_float), (void*)&prelim_contrastThreshold);
	GPUError |= clSetKernelArg(GPUKernelDesc, 7, sizeof(cl_int), (void*)&intvl);
	GPUError |= clSetKernelArg(GPUKernelDesc, 8, sizeof(cl_int), (void*)&octv);
	GPUError |= clSetKernelArg(GPUKernelDesc, 9, sizeof(cl_mem), (void*)&cmDevBufNumber);
	if(GPUError) return false;

	if(clEnqueueNDRangeKernel( GPU::getInstance().GPUCommandQueue, GPUKernelDesc, 2, NULL, GPUGlobalWorkSize, GPULocalWorkSize, 0, NULL, NULL)) 
	{
		cout << "Error clEnqueueNDRangeKernel" << endl;
		return false;
	}
	
	GPUError = clEnqueueReadBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufCount, CL_TRUE, 0, sizeof(int), (void*)&counter, 0, NULL, NULL);
	CheckError(GPUError);
	
	GPUError = clEnqueueReadBuffer(GPU::getInstance().GPUCommandQueue, cmDevBufKeys, CL_TRUE, 0, SIFT_MAX_NUMBER_KEYS*sizeof(Keys), (void*)keys, 0, NULL, NULL);
	CheckError(GPUError);
	
	*numExtr = numberExtr;

	return true;
}










} //: namespace ClSIFT
} //: namespace Processors

