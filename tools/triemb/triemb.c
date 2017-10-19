/* C function for triangulation embedding */

#include <stdlib.h>
#include <math.h>

#include "triemb.h"


void compute_residual_1 (const float * v, const float * c, const float * m,
                         float * y, double * alpha, long d, long k)
{
  long j, l;
  
  for (j = 0 ; j < k ; j++) {
    double nr = 0;
    for (l = 0 ; l < d ; l++) {
      y[l] = v[l] - c[l];
      nr += y[l] * y[l];
    }
      
    nr = 1 / sqrt (nr);
    *alpha = nr;
    for (l = 0 ; l < d ; l++)
      y[l] = y[l] * nr - m[l];
     
    y += d;
    c += d;
    m += d;
    alpha++;
  }
}


void compute_residual_all (const float * v, const float * c, const float * m,
                           float * y, double * alpha, long n, long d, long k)
{
  long i;
  long D = d * k;
#ifdef _OPENMP
#pragma omp parallel for private (i)
  for (i = 0 ; i < n ; i++)
    compute_residual_1 (v + i * d, c, m, y + i * D, alpha + i * k, d, k);
#else
  for (i = 0 ; i < n ; i++)
    compute_residual_1 (v + i * d, c, m, y + i * D, alpha + i * k, d, k);
#endif    
}



void compute_triemb_aggsum (const float * v, const float * c, const float * m,
                           float * y, long n, long d, long k)
{
  long i, j, l;
  
  /* A single malloc for all the temporary vectors */
  float yres[k * d];

  for (j = 0 ; j < k * d ; j++)
    y[k * d] = 0;
  
  /* #pragma omp parallel for private (j) */
  for (j = 0 ; j < k ; j++) {
    float * yj = y + d * j;         /* to output result */
    float * yresj = yres + d * j;   /* temporary area */
    const float * cj = c + d * j;   /* focus on centroid j */
    /* const float * mj = m + d * j; */  /* interesting part of the mean */
    
    for (i = 0 ; i < n ; i++) {
      const float * vi = v + i * d; /* interesting vector */
      
      double nr = 0;
      for (l = 0 ; l < d ; l++) {
        yresj[l] = vi[l] - cj[l];
        nr += yresj[l] * yresj[l];
      }
      
      nr = 1 / sqrt (nr);
      for (l = 0 ; l < d ; l++)
        yj[l] += yresj[l] * nr;
    }
  }
  
  for (l = 0 ; l < d * k ; l++)
    y[l] -= n * m[l];  
}

