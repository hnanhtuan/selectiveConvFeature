/* *** Not tested yet on an image set *** */
#include <stdio.h>
#include <string.h>


#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <yael/vector.h>
#include <yael/gmm.h>
#include <yael/machinedeps.h>

#include "mex.h"

#define PARAM_V        prhs[0]
#define PARAM_W        prhs[1]
#define PARAM_MU       prhs[2]
#define PARAM_SIGMA    prhs[3]




void mexFunction (int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray*prhs[])

{
  if (nrhs < 4) 
    mexErrMsgTxt("At least 4 arguments are required even nb of input arguments required.");
  else if (nlhs != 1) 
    mexErrMsgTxt("yael_fisher produces exactly 1 output argument.");

  int flags = GMM_FLAGS_MU;
  int verbose = 0;
  int fishernorm1 = 1;
  
  if(mxGetClassID(PARAM_V)!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(PARAM_W)!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(PARAM_MU)!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  if(mxGetClassID(PARAM_SIGMA)!=mxSINGLE_CLASS)
    mexErrMsgTxt("need single precision array.");

  float *v = (float*) mxGetPr (PARAM_V);
  float *w = (float*) mxGetPr (PARAM_W);
  float *mu = (float*) mxGetPr (PARAM_MU);
  float *sigma = (float*) mxGetPr (PARAM_SIGMA);

  {
    int i;
    for(i = 4 ; i < nrhs ; i += 1) {
      char varname[256];
      if (mxGetClassID(prhs[i]) != mxCHAR_CLASS) 
        mexErrMsgTxt ("variable name required");         

      if (mxGetString (prhs[i], varname, 256) != 0)
        mexErrMsgTxt ("Could not convert string data");

      if (!strcmp(varname, "sigma")) 
	flags |= GMM_FLAGS_SIGMA;
      
      else if (!strcmp(varname,"weights")) 
        flags |= GMM_FLAGS_W;

      else if (!strcmp(varname,"nomu")) 
        flags &= ~ GMM_FLAGS_MU;

      else if (!strcmp(varname,"verbose")) 
        verbose = 1;

      else if (!strcmp(varname,"nonorm")) 
        fishernorm1 = 0;

      else 
        mexErrMsgTxt("unknown variable name");  
    }
  }

  if (verbose) {
    fprintf (stdout, "v     -> %ld x %ld\n", mxGetM (PARAM_V), mxGetN (PARAM_V));
    fprintf (stdout, "w     -> %ld x %ld\n", mxGetM (PARAM_W), mxGetN (PARAM_W));
    fprintf (stdout, "mu    -> %ld x %ld\n", mxGetM (PARAM_MU), mxGetN (PARAM_MU));
    fprintf (stdout, "sigma -> %ld x %ld\n", mxGetM (PARAM_SIGMA), mxGetN (PARAM_SIGMA));
  }

  int d = mxGetM (PARAM_V);  /* vector dimensionality */
  int n = mxGetN (PARAM_V);  /* number of fisher vector to produce */
  int k = mxGetN (PARAM_W);  /* number of gaussian */

  if (verbose)
    fprintf (stdout, "d       = %d\nn       = %d\nk       = %d\n", d, n, k);

  if (mxGetM (PARAM_MU) != d || mxGetM (PARAM_SIGMA) != d || 
      mxGetN (PARAM_MU) !=k || mxGetN (PARAM_SIGMA) != k || 
      (mxGetM (PARAM_W) != 1 && mxGetN (PARAM_W) != 1) )
    mexErrMsgTxt("Invalid input dimensionalities.");

  

  /* ouptut: GMM, i.e., weights, mu and variances */
  gmm_t g = {d, k, w, mu, sigma};
  int dout = gmm_fisher_sizeof (&g, flags); 
  if (verbose)
    fprintf (stdout, "Size of the fisher vector = %d\n", dout);

  plhs[0] = mxCreateNumericMatrix (dout, 1, mxSINGLE_CLASS, mxREAL);
  float * vf = (float *) mxGetPr (plhs[0]);
  gmm_fisher (n, v, &g, flags, vf);

  if (fishernorm1) {
    int ret = fvec_normalize (vf, dout, 2.);
    if (ret == 1)
      fvec_set (vf, dout, 1);
  }
}
