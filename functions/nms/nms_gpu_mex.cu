/*
* Example of how to use the mxGPUArray API in a MEX file.  This example shows
* how to write a MEX function that takes a gpuArray input and returns a
* gpuArray output, e.g. B=mexFunction(A).
*
* Copyright 2012 The MathWorks, Inc.
*/

#include "mex.h"
#include <vector>
#include <iostream>

#define DIVUP(m,n)		((m)/(n)+((m)%(n)>0))
int const threadsPerBlock = (sizeof(unsigned long long) * 8);

/*
* Device code
*/
__device__ inline float devIoU(float const * const a, float const * const b)
{
	float left = max(a[0], b[0]), right = min(a[2], b[2]);
	float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
	float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
	float interS = width * height;
	float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
	float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
	return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thres, const float *dev_boxes, unsigned long long *dev_mask)
{
	const int row_start = blockIdx.y, col_start = blockIdx.x;
	const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock), col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

	//if (row_start > col_start) return;

	__shared__ float block_boxes[threadsPerBlock * 5];
	if (threadIdx.x < col_size)
	{
		block_boxes[threadIdx.x * 5 + 0] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
		block_boxes[threadIdx.x * 5 + 1] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
		block_boxes[threadIdx.x * 5 + 2] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
		block_boxes[threadIdx.x * 5 + 3] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
		block_boxes[threadIdx.x * 5 + 4] = dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
	}
	__syncthreads();

	if (threadIdx.x < row_size)
	{
		const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
		const float *cur_box = dev_boxes + cur_box_idx * 5;
		int i = 0;
		unsigned long long t = 0;
		int start = 0;
		if (row_start == col_start) start = threadIdx.x + 1;
		for (i = start; i < col_size; i++)
		{
			if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thres)
			{
				t |= 1ULL << i;
			}
		}
		const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
		dev_mask[cur_box_idx * col_blocks + col_start] = t;
	}
}

/*
* Host code
*/
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	
	/* Declare all variables.*/
	mxArray const *boxes, *ov_thres;
	float *boxes_host = NULL;
	float *boxes_dev = NULL;
	unsigned long long *mask_dev = NULL;

	/* Throw an error if the input is not a array. */
	if (nrhs != 2) {
		mexErrMsgTxt("nms_gpu_mex::need 2 inputs");
	}

	boxes = prhs[0];
	if (mxGetClassID(boxes) != mxSINGLE_CLASS) {
		mexErrMsgTxt("nms_gpu_mex::input boxes must be single");
	}

	ov_thres = prhs[1];
	if (mxGetClassID(ov_thres) != mxDOUBLE_CLASS) {
		mexErrMsgTxt("nms_gpu_mex::input boxes must be double");
	}

	float nms_overlap_thres = (float)mxGetScalar(ov_thres);

	int boxes_dim = mxGetM(boxes);
	int boxes_num = mxGetN(boxes);
	if (boxes_dim != 5)
	{
		mexErrMsgTxt("nms_gpu_mex::input boxes's row must be 5");
	}

	boxes_host = (float *)(mxGetPr(boxes));
	const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

	cudaMalloc(&boxes_dev, mxGetNumberOfElements(boxes) * sizeof(float));
	cudaMemcpy(boxes_dev, boxes_host, mxGetNumberOfElements(boxes) * sizeof(float), cudaMemcpyHostToDevice);

	/* Create a GPUArray to hold the result and get its underlying pointer. */
	cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));
	

	/*
	* Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
	* and it would be possible for the number of elements to be too large for
	* the grid. For this example we are not guarding against this possibility.
	*/

	dim3 blocks(DIVUP(boxes_num, threadsPerBlock), DIVUP(boxes_num, threadsPerBlock));
	dim3 threads(threadsPerBlock);
	nms_kernel << <blocks, threads >> >(boxes_num, nms_overlap_thres, boxes_dev, mask_dev);

	std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
	cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks, cudaMemcpyDeviceToHost);

	std::vector<unsigned long long> remv(col_blocks);
	memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

	std::vector<int> keep;
	keep.reserve(boxes_num);
	for (int i = 0; i < boxes_num; i++)
	{
		int nblock = i / threadsPerBlock;
		int inblock = i % threadsPerBlock;

		if (!(remv[nblock] & (1ULL << inblock)))
		{
			keep.push_back(i + 1);  // to matlab's index

			unsigned long long *p = &mask_host[0] + i * col_blocks;
			for (int j = nblock; j < col_blocks; j++)
			{
				remv[j] |= p[j];
			}
		}
	}

	/* Wrap the result up as a MATLAB cpuArray for return. */
	mwSize dims[4] = { (int)keep.size(), 1, 1, 1 };
	plhs[0] = mxCreateNumericArray(4, dims, mxINT32_CLASS, mxREAL);

	int *output = (int *)(mxGetPr(plhs[0]));
	memcpy(output, &keep[0], (int)keep.size() * sizeof(int));


	cudaFree(boxes_dev);
	cudaFree(mask_dev);
}
