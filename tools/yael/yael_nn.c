/* This file is a mex-matlab wrap for the nearest neighbor search function of yael */

#include <assert.h>
#include <math.h>
#include "mex.h"
#include <sys/time.h>


#include <yael/nn.h> 


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 2 || nrhs > 4) 
    mexErrMsgTxt ("Invalid number of input arguments");
  
  if (nlhs != 2)
    mexErrMsgTxt ("2 output arguments required");

  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  int nq = mxGetN (prhs[1]);

  if (mxGetM (prhs[1]) != d)
      mexErrMsgTxt("Dimension of base and query vectors are not consistent");
  
  
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS 
      || mxGetClassID(prhs[1]) != mxSINGLE_CLASS )
    mexErrMsgTxt ("need single precision array"); 


  float *b = (float*) mxGetPr (prhs[0]);  /* database vectors */
  float *v = (float*) mxGetPr (prhs[1]);  /* query vectors */
  int k = 1; 
  int distype = 2;

  if (nrhs >= 3)
    k = (int) mxGetScalar(prhs[2]);

  if (nrhs >= 4)
    distype = (int) mxGetScalar(prhs[3]);

  if (n < k) 
    mexErrMsgTxt("fewer vectors than number to be returned");    


  /* ouptut: centroids, assignment, distances */

  plhs[0] = mxCreateNumericMatrix (k, nq, mxINT32_CLASS, mxREAL);
  int *assign = (int*) mxGetPr (plhs[0]);
  
  plhs[1] = mxCreateNumericMatrix (k, nq, mxSINGLE_CLASS, mxREAL);
  float *dis = (float*) mxGetPr (plhs[1]);

  /* With Matlab, we have to avoid using threads for the L2 distance, 
     because this one makes a call to MKL, which is no thread-safe */
  knn_full (distype, nq, n, d, k, b, v, NULL, assign, dis);

  /* post-processing: convert to matlab indices, and enforce full sort */

  int i;

  for (i = 0 ; i < nq * k ; i++)
    assign[i]++;
}
