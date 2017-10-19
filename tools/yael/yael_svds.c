#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/matrix.h>
#include <yael/machinedeps.h>

#include "mex.h"


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if(!((nrhs == 2 && (nlhs == 1 || nlhs == 2 || nlhs == 3)) || 
       (nrhs == 3 && nlhs == 2))) 
    mexErrMsgTxt("wrong number or inputs or outputs.");
  
  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  
  if(mxGetClassID(prhs[0]) != mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  int nev = (int) mxGetScalar (prhs[1]);


  if(nev > d || nev > n) 
    mexErrMsgTxt("too many singular values requested");
  
  float *u = NULL, *s = NULL, *v = NULL;  

  if(nlhs == 3) {    
    plhs[0] = mxCreateNumericMatrix (d, nev, mxSINGLE_CLASS, mxREAL);
    u = (float*)mxGetPr(plhs[0]);
    plhs[1] = mxCreateNumericMatrix (nev, 1, mxSINGLE_CLASS, mxREAL);
    s = (float*)mxGetPr(plhs[1]);
    plhs[2] = mxCreateNumericMatrix (n, nev, mxSINGLE_CLASS, mxREAL);
    v = (float*)mxGetPr(plhs[2]);
  } else if(nlhs == 2) {
    if(nrhs == 2) {
      plhs[0] = mxCreateNumericMatrix (d, nev, mxSINGLE_CLASS, mxREAL);
      u = (float*)mxGetPr(plhs[0]);
      plhs[1] = mxCreateNumericMatrix (nev, 1, mxSINGLE_CLASS, mxREAL);
      s = (float*)mxGetPr(plhs[1]);
    } else {
      plhs[0] = mxCreateNumericMatrix (nev, 1, mxSINGLE_CLASS, mxREAL);
      s = (float*)mxGetPr(plhs[0]);
      plhs[1] = mxCreateNumericMatrix (n, nev, mxSINGLE_CLASS, mxREAL);
      v = (float*)mxGetPr(plhs[1]);
    }
  } else if(nlhs == 1) {
    plhs[0] = mxCreateNumericMatrix (nev, 1, mxSINGLE_CLASS, mxREAL);
    s = (float*)mxGetPr(plhs[0]);    
  }

  int ret = fmat_svd_partial(d, n, nev, (float*)mxGetPr(prhs[0]), 
                             s, u, v); 
  
  if(ret <= 0) {
    mexErrMsgTxt("Did not find any singluar value.");    
  }

  if(ret != nev) {    
    fprintf(stderr, "WARN only %d / %d singular values converged\n", ret, nev); 
  }

}
