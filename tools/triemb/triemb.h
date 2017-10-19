#ifndef __triemb_h
#define __triemb_h


/* Compute a single multi-residual vector associated with an input desc */
void compute_residual_1 (const float * v, const float * c, const float * m,
                         float * y, double * alpha, long d, long k);

void compute_residual_all (const float * v, const float * c, const float * m,
                           float * y, double * alpha, long n, long d, long k);

/* Compute and directly aggregate multi-residual vectors */
void compute_triemb_aggsum (const float * v, const float * c, const float * m,
                           float * y, long n, long d, long k);


#endif
