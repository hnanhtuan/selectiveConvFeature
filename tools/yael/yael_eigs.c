#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/eigs.h>
#include <yael/machinedeps.h>

#include "mex.h"


void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs != 2) 
    mexErrMsgTxt("need 2 inputs.");
  else if (nlhs != 2) 
    mexErrMsgTxt("need 2 outputs.");

  int d = mxGetM (prhs[0]);
  int n = mxGetN (prhs[0]);
  
  if(mxGetClassID(prhs[0])!=mxSINGLE_CLASS || n != d)
    mexErrMsgTxt("need square single precision array.");

  int nev = (int) mxGetScalar (prhs[1]);

  if(nev * 2 > d) {
    mexErrMsgTxt("If so many eigenvals are needed, it is better to use the full version...");    
  }


  /* ouptut: values, vectors */

  plhs[1] = mxCreateNumericMatrix (nev, 1, mxSINGLE_CLASS, mxREAL);
  plhs[0] = mxCreateNumericMatrix (d, nev, mxSINGLE_CLASS, mxREAL);
  
  int ret = eigs_sym_part (d, (float*)mxGetPr(prhs[0]), nev, 
                           (float*)mxGetPr(plhs[1]), 
                           (float*)mxGetPr(plhs[0])); 

  if(ret < 0) {
    char errormsg[256]; 
    sprintf(errormsg, "arpack returned error %d", ret); 
    mexErrMsgTxt(errormsg);   
  }

}
