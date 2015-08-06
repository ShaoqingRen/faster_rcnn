#include "mex.h"
#ifdef WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <algorithm>
#endif
#include <vector>
#include <map>
#include <omp.h>
using namespace std;

struct score {
	double s;
	int idx;
	bool operator() (score i, score j) { return (i.idx < j.idx);}
} score;

template <typename T>
void nms(const mxArray *input_boxes, int iScoreIdx, double overlap, const vector<double> &vArea, vector<int> &vPick, int &nPick)
{
	int nSample = (int)mxGetM(input_boxes);
	int nDim_boxes = (int)mxGetN(input_boxes);

    T *pBoxes = (T*)mxGetData(input_boxes);

	//vector<double> vArea(nSample);
	//for (int i = 0; i < nSample; ++i)
	//{
	//	vArea[i] = double(pBoxes[2*nSample + i] - pBoxes[0*nSample + i] + 1) 
	//	* (pBoxes[3*nSample + i] - pBoxes[1*nSample + i] + 1);
	//	if (vArea[i] < 0)
	//		mexErrMsgTxt("Boxes area must >= 0");
	//}

	std::multimap<T, int> scores;
	for (int i = 0; i < nSample; ++i)
		scores.insert(std::pair<T,int>(pBoxes[iScoreIdx*nSample + i], i));

	nPick = 0;

	do 
	{
		int last = scores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;

		for (typename std::multimap<T, int>::iterator it = scores.begin(); it != scores.end();)
		{
			int it_idx = it->second;
			T xx1 = std::max(pBoxes[0*nSample + last], pBoxes[0*nSample + it_idx]);
			T yy1 = std::max(pBoxes[1*nSample + last], pBoxes[1*nSample + it_idx]);
			T xx2 = std::min(pBoxes[2*nSample + last], pBoxes[2*nSample + it_idx]);
			T yy2 = std::min(pBoxes[3*nSample + last], pBoxes[3*nSample + it_idx]);

			double w = max(0.0, xx2-xx1+1), h = max(0.0, yy2-yy1+1);

			double ov = w*h / (vArea[last] + vArea[it_idx] - w*h);

			if (ov > overlap)
			{
				#ifdef WIN32
				it = scores.erase(it);
                #else
                typename std::multimap<T, int>::iterator save=it; ++save;
				scores.erase(it);
                it=save;
                #endif
			}
			else
			{
				it++;
			}
		}

	} while (scores.size() != 0);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
	if (nrhs != 2)
		mexErrMsgTxt("Wrong number of inputs"); 
	if (nlhs != 1)
		mexErrMsgTxt("One output");

	const mxArray *input_boxes = prhs[0];
	if (mxGetClassID(input_boxes) != mxDOUBLE_CLASS && mxGetClassID(input_boxes) != mxSINGLE_CLASS)
		mexErrMsgTxt("Input boxes must be Double or Single");

	const mxArray *input_overlap = prhs[1];
	if (mxGetClassID(input_overlap) != mxDOUBLE_CLASS )
		mexErrMsgTxt("Input overlap must be Double");

	double overlap = mxGetScalar(input_overlap);

	int nSample = (int)mxGetM(input_boxes);
	int nDim_boxes = (int)mxGetN(input_boxes);

	if (nSample * nDim_boxes == 0)
	{
		plhs[0] = mxCreateNumericMatrix(0, 0, mxDOUBLE_CLASS, mxREAL);
		return;
	}

	if (nDim_boxes < 5)
		mexErrMsgTxt("nms_mex boxes must has least 5 columns");

	vector<double> vArea(nSample);
	if(mxGetClassID(input_boxes) == mxDOUBLE_CLASS)
	{
		double *pBoxes = (double*)mxGetData(input_boxes);
		for (int i = 0; i < nSample; ++i)
		{
			vArea[i] = double(pBoxes[2*nSample + i] - pBoxes[0*nSample + i] + 1) 
				* (pBoxes[3*nSample + i] - pBoxes[1*nSample + i] + 1);
			if (vArea[i] < 0)
				mexErrMsgTxt("Boxes area must >= 0");
		}
	}
	else
	{
		if(mxGetClassID(input_boxes) == mxDOUBLE_CLASS)
		{
			float *pBoxes = (float*)mxGetData(input_boxes);
			for (int i = 0; i < nSample; ++i)
			{
				vArea[i] = double(pBoxes[2*nSample + i] - pBoxes[0*nSample + i] + 1) 
					* (pBoxes[3*nSample + i] - pBoxes[1*nSample + i] + 1);
				if (vArea[i] < 0)
					mexErrMsgTxt("Boxes area must >= 0");
			}
		}
	}

	vector<int> nPick(nDim_boxes - 4, 0);
	vector<vector<int> > vPicks(nDim_boxes - 4);
	plhs[0] = mxCreateCellMatrix_730(nDim_boxes - 4, 1);

#pragma omp parallel for ordered schedule(dynamic)
	for (int i = 0; i < vPicks.size(); ++i)
	{
		vPicks[i].resize(nSample);
	
		if(mxGetClassID(input_boxes) == mxDOUBLE_CLASS)
			nms<double>(input_boxes, i+4, overlap, vArea, vPicks[i], nPick[i]);
		else
			nms<float>(input_boxes, i+4, overlap, vArea, vPicks[i], nPick[i]);

		mxArray *mxPick = mxCreateNumericMatrix(nPick[i], 1, mxDOUBLE_CLASS, mxREAL);
		double *pRst = mxGetPr(mxPick);
		for (int j = 0; j < nPick[i]; ++j)
			pRst[j] = vPicks[i][j] + 1;

		mxSetCell(plhs[0], i, mxPick);
	}
	
}