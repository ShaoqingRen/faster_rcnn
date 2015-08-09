#include "mex.h"
#ifdef _MSC_VER
#include <windows.h>
#include <tchar.h>
#endif
#include <vector>
#include <map>
using namespace std;

struct score {
	double s;
	int idx;
	bool operator() (score i, score j) { return (i.idx < j.idx);}
} score;

template <typename T>
void nms(const mxArray *input_boxes, double overlap, vector<int> &vPick, int &nPick)
{
	int nSample = (int)mxGetM(input_boxes);
	int nDim_boxes = (int)mxGetN(input_boxes);

    T *pBoxes = (T*)mxGetData(input_boxes);

	vector<double> vArea(nSample);
	for (int i = 0; i < nSample; ++i)
	{
		vArea[i] = double(pBoxes[2*nSample + i] - pBoxes[0*nSample + i] + 1) 
		* (pBoxes[3*nSample + i] - pBoxes[1*nSample + i] + 1);
		if (vArea[i] < 0)
			mexErrMsgTxt("Boxes area must >= 0");
	}

	std::multimap<T, int> scores;
	for (int i = 0; i < nSample; ++i)
		scores.insert(std::pair<T,int>(pBoxes[4*nSample + i], i));

	nPick = 0;

	do 
	{
		int last = scores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;

		for (typename std::multimap<T, int>::iterator it = scores.begin(); it != scores.end();)
		{
			int it_idx = it->second;
			T xx1 = max(pBoxes[0*nSample + last], pBoxes[0*nSample + it_idx]);
			T yy1 = max(pBoxes[1*nSample + last], pBoxes[1*nSample + it_idx]);
			T xx2 = min(pBoxes[2*nSample + last], pBoxes[2*nSample + it_idx]);
			T yy2 = min(pBoxes[3*nSample + last], pBoxes[3*nSample + it_idx]);

			double w = max(T(0.0), xx2-xx1+1), h = max(T(0.0), yy2-yy1+1);

			double ov = w*h / (vArea[last] + vArea[it_idx] - w*h);

			if (ov > overlap)
			{
				it = scores.erase(it);
			}
			else
			{
				it++;
			}
		}

	} while (scores.size() != 0);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
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

	if (nDim_boxes != 5)
		mexErrMsgTxt("nms_mex boxes must has 5 columns");

	
	int nPick = 0;
	vector<int> vPick(nSample);
	if(mxGetClassID(input_boxes) == mxDOUBLE_CLASS)
		nms<double>(input_boxes, overlap, vPick, nPick);
	else
		nms<float>(input_boxes, overlap, vPick, nPick);

	plhs[0] = mxCreateNumericMatrix(nPick, 1, mxDOUBLE_CLASS, mxREAL);
	double *pRst = mxGetPr(plhs[0]);
	for (int i = 0; i < nPick; ++i)
		pRst[i] = vPick[i] + 1;
}
