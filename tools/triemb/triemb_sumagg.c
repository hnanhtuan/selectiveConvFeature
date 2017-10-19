#include "mex.h"

#include "triemb.h"



void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs != 3) 
    mexErrMsgTxt ("Invalid number of input arguments");
  
  if (nlhs != 1)
    mexErrMsgTxt ("1 output argument only");

  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("First parameter must be a single precision matrix"); 
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("First parameter must be a single precision matrix"); 
  if (mxGetClassID(prhs[2]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("First parameter must be a single precision matrix"); 


  int d = mxGetM (prhs[0]);  /* vector dimensionality */
  int n = mxGetN (prhs[0]);  /* number of input vectors */
  
  int d2 = mxGetM (prhs[1]); /* dimensionality of centroids (should be d) */
  int k = mxGetN (prhs[1]);  /* number of clusters */
  
  int D = mxGetM (prhs[2]);  /* dimensionality of mean vector (d*k) */
  int n3 = mxGetN (prhs[2]); /* number of mean vector -> should be 1 */
  
  if (d != d2 || D != (d*k) || n3 != 1)
    mexErrMsgTxt ("Invalid dimensionality of arguments\n");
  
  const float * v = (float*) mxGetPr (prhs[0]);  /* the set of vectors to be normalized */
  const float * c = (float*) mxGetPr (prhs[1]);  /* centroids */
  const float * m = (float*) mxGetPr (prhs[2]);  /* mean vector */

  
  plhs[0] = mxCreateNumericMatrix (D, 1, mxSINGLE_CLASS, mxREAL);
  float * y = (float*) mxGetPr (plhs[0]);
  
  compute_triemb_aggsum (v, c, m, y, n, d, k);
}
