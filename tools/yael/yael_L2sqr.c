/* This file is a mex-matlab wrap for the nearest neighbor search function of yael */

#include <assert.h>
#include <math.h>
#include "mex.h"
#include <sys/time.h>


#include <yael/nn.h> 


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 2 || nrhs > 3) 
    mexErrMsgTxt ("Invalid number of input arguments");
  
  if (nlhs != 1)
    mexErrMsgTxt ("1 output arguments required");

  int d = mxGetM (prhs[0]);
  int na = mxGetN (prhs[0]);
  int nb = mxGetN (prhs[1]);

  if (mxGetM (prhs[1]) != d)
      mexErrMsgTxt("Dimension of base and query vectors are not consistent");
  
  
  if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS 
      || mxGetClassID(prhs[1]) != mxSINGLE_CLASS )
    mexErrMsgTxt ("need single precision array"); 


  float *a = (float*) mxGetPr (prhs[0]);  
  float *b = (float*) mxGetPr (prhs[1]); 

  /* ouptut: centroids, assignment, distances */
  plhs[0] = mxCreateNumericMatrix (na, nb, mxSINGLE_CLASS, mxREAL);
  float *dis = (float*) mxGetPr (plhs[0]);

  compute_cross_distances (d, na, nb, a, b, dis);
}
