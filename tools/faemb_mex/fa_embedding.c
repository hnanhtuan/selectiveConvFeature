#include "mex.h"

#include "faemb.h"


/*function [ PHIX ] = embeding( X, B, gama_all )*/
/*
 * X: kxm: data
 * B: kxn: dictionary
 * gama_all: nxm: coefficient matrix
 * PHIX: k*(k+1)/2*n x m: embedded vectors
 */

void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs != 3) 
    mexErrMsgTxt ("Invalid number of input arguments");
  
  if (nlhs < 1 && nlhs > 2)
    mexErrMsgTxt ("1 or 2 output arguments are required");

  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("First parameter must be a single precision matrix"); 
  if (mxGetClassID(prhs[1]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("Second parameter must be a single precision matrix"); 
  if (mxGetClassID(prhs[2]) != mxSINGLE_CLASS)
    mexErrMsgTxt ("Third parameter must be a single precision matrix"); 
  

  int k = mxGetM (prhs[0]);  /* dimenstion of one sample */
  int m = mxGetN (prhs[0]);  /* number of samples */
  
  int k2 = mxGetM (prhs[1]); /* dimensionality of centroids (should equal k) */
  int n = mxGetN (prhs[1]);  /* number of anchor points */
  
  int n2 = mxGetM (prhs[2]);  /* dimensionality of each coefficent vector (should equal n)*/
  int m2 = mxGetN (prhs[2]); /* number of samples */
  
   
  
  if (k != k2 || n != n2 || m != m2)
    mexErrMsgTxt ("Invalid dimensionality of arguments\n");
  
  const float * X = (float*) mxGetPr (prhs[0]);  /* the set of samples */
  const float * B = (float*) mxGetPr (prhs[1]);  /* centroids */
  const float * gamma_all = (float*) mxGetPr (prhs[2]);  /* coefficient matrix */
  
  int D = k*(k+1)/2 * n;
  plhs[0] = mxCreateNumericMatrix (D, m, mxSINGLE_CLASS, mxREAL); /*embedded vectors*/
  float * PHIX = (float*) mxGetPr (plhs[0]);
  
 
  compute_faemb(X, B, gamma_all, PHIX, k, m, n);
}
