#ifndef __faemb_h
#define __faemb_h

void matrix_mul(const float *first, const float* second, float* out, long m, long n, long p, long q);


/* Compute a single multi-residual vector associated with an input desc */
void compute_residual_1 (const float * v, const float * c, const float * m,
                         float * y, double * alpha, long d, long k);

void compute_residual_all (const float * v, const float * c, const float * m,
                           float * y, double * alpha, long n, long d, long k);

/*Temb moi data point den nn nearest neighbor, chu o ko phai tat ca centroid nhu original Temb*/
void compute_residual_all_new (const float * v, const float * c, const float * m, const float *idx,
                           float * y, double * alpha, long n, long d, long k, long nn);

/*for compute phix of faemb*/
void compute_faemb(const float* X, const float* B, const float* gamma_all, float* PHIX, long k, long m, long n);

/*VLATemb moi data point den nn nearest neighbor*/
void compute_residual_all_new_VLAT (const float * v, const float * c, const float * m, const float *  idxnn,  
        float * y, double * alpha, long n, long d, long k, long nn);
                         
/* Compute and directly aggregate multi-residual vectors */
void compute_triemb_aggsum (const float * v, const float * c, const float * m,
                           float * y, long n, long d, long k);

void compute_triemb_1 (const float * vtrain, const float * c, const float * m, const float * Pemb_local,
                           float * vout, long n, long d, long k);
/*
void compute_triemb_2 (const float * vtrain, const float * c, const float * m, const float * Pemb_local,
                           float * vout, long n, long d, long k);
 */
#endif
