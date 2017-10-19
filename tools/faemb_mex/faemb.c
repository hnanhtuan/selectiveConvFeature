/* C function for triangulation embedding */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "faemb.h"

/*lay tung cot cua ma tran 1 nhan tung cot cua ma tran 2. Khi input thi ma tran 1 phai chuyen vi. Ma tran out kich thuoc q*n. Ra matlab chuyen vi lai thi duoc n*q */
void matrix_mul(const float *first, const float* second, float* out, long m, long n, long p, long q)
{
    long c, d, k;
    for (c = 0; c < n; c++) /*n cot cua ma tran 1, moi cot co m hang*/
    {
        float* outc = out + c*q;
        
        const float*  x = first + c*m;
        
        for(d = 0; d < q; d++) /*q cot cua ma tran thu 2, moi cot co m hang. m = p*/
        {
            const float*  y = second + d*p;
            float sum = 0;
            for(k = 0; k < m; k++)
            {
                sum += x[k]*y[k];
            }
            outc[d] = sum;
        }
    }
}
void L2_norm_subtract_mean(const float*m, float * y, long D)
{
    long i; 
    double nr = 0;
    for ( i = 0; i < D; i++)
        nr += y[i]*y[i];
    
    nr = 1/sqrt(nr);
    for (i = 0; i < D; i++)
        y[i] *= nr;
    for (i = 0; i<D; i++)
        y[i] -= m[i];
}

void compute_residual_1 (const float * v, const float * c, const float * m,
                         float * y, double * alpha, long d, long k)
{
  long j, l;
  
  for (j = 0 ; j < k ; j++) 
  {
        double nr = 0;
        for (l = 0 ; l < d ; l++) 
        {
          y[l] = v[l] - c[l];
          nr += y[l] * y[l];
        }

        nr = 1 / sqrt (nr);
        *alpha = nr;

        for (l = 0 ; l < d ; l++)
        {
          y[l] = y[l] * nr - m[l];  /* Minh da comment cho nay. Muc dich la sau khi L2 R roi moi tru mean */
           /*y[l] = y[l] * nr;*/
        }

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
  /*Toan codes*/
  /*
  for (i = 0; i< n; i++)
      L2_norm_subtract_mean (m, y+i*D, D);
   */
}

/*
 * v: 1 data point
 * c: centroids
 * m: vector mean R0
 * idx: idx of NN cua v
 * y: output
 * nn: number of considered NN
 */
void compute_residual_1_new (const float * v, const float * c, const float * m, const float* idxnn,
                         float * y, double * alpha, long d, long k, long nn)
{
  long j, l, t;
  
  for (t = 0; t < k*d; t ++)
      y[t] = 0;
  
  for (j = 0 ; j < k ; j++) 
  {
         *alpha = 0;
        for (t = 0; t<nn; t++) 
        {
                if (idxnn[t] - 1 == j) /*if cluster jth is nn of v --> embed*/
                {
                    double nr = 0;
                    for (l = 0 ; l < d ; l++) 
                    {
                      y[l] = v[l] - c[l];
                      nr += y[l] * y[l];
                    }
                    nr = 1 / sqrt (nr);
                    *alpha = nr;
                    for (l = 0 ; l < d ; l++)
                    {
                       y[l] = y[l] * nr;
                    }
                    break;
                }
        }
    
        for (l = 0; l<d; l++)
        {
            y[l] = y[l] - m[l];
        }
    
    y += d;
    c += d;
    m += d;
    alpha++;
  }
}

void compute_residual_all_new (const float * v, const float * c, const float * m, const float *  idxnn,  
        float * y, double * alpha, long n, long d, long k, long nn)
{
    
  long i;
  long D = d * k;
#ifdef _OPENMP
#pragma omp parallel for private (i)
  for (i = 0 ; i < n ; i++)
    compute_residual_1_new (v + i * d, c, m, idxnn + i*nn, y + i * D, alpha + i * k, d, k, nn);
#else
  for (i = 0 ; i < n ; i++)
    compute_residual_1_new (v + i * d, c, m, idxnn + i*nn, y + i * D, alpha + i * k, d, k, nn);
#endif    
     
}
 
/*
 * v: 1 data point
 * c: centroids
 * m: vector mean R0
 * idx: idx of NN cua v
 * y: output
 * nn: number of considered NN
 */
void compute_residual_1_new_VLAT (const float * v, const float * c, const float * m, const float* idxnn,
                         float * y, double * alpha, long d, long k, long nn)
{

    /*
  FILE *f = fopen("track.txt","wt");
  fprintf(f,"d = %ld, k = %ld, nn = %ld\n",d,k,nn);
  */
  long j, l,h, t;
  
  /*for (t = 0; t < k*d*d; t ++)*/
  for (t = 0; t < k*d*(d+1)/2; t ++) /*chi tinh 1/2 ma tran*/
      y[t] = 0;
  
  float r[d];
  
  for (j = 0 ; j < k ; j++) 
  {
        *alpha = 0;
        for (t = 0; t<nn; t++) 
        {
                if (idxnn[t] - 1 == j) /*if cluster jth is nn of v --> embeded*/
                {
                    /*fprintf(f,"data point thuoc cluster %ld \n",j);*/
                    double nr = 0;
                    for (l = 0 ; l < d ; l++) 
                    {
                      r[l] = v[l] - c[l];
                      nr += r[l] * r[l];
                      /*fprintf(f,"%f  \n",r[l]);*/
                    }
                    nr = 1 / sqrt (nr);
                    *alpha = nr;
                    for (l = 0 ; l < d ; l++)
                    {
                       r[l] = r[l] * nr;
                    }
                    /*tinh rr' */
                    long count = 0;
                    for (l = 0; l<d; l++)
                        /*for(h = 0; h<d; h++)*/ 
                        for (h = l; h<d; h++) /*chi tinh 1/2 ma tran*/
                        {
                            y[count++] = r[l]*r[h];
                        }
                    
                    break;
                }
        }
    
       /* for (l = 0; l<d*d; l++)*/
        for (l  = 0; l<d*(d+1)/2; l++) /*chi tinh 1/2 ma tran*/
        {
            y[l] = y[l] - m[l];
        }
    
   /* y += d*d;
     m += d*d;*/  
    y += d*(d+1)/2; /*chi tinh 1/2 ma tran*/
    m += d*(d+1)/2;
    
    c += d;
    alpha++;
  }
  
  /*fclose(f);*/
}

void compute_residual_all_new_VLAT (const float * v, const float * c, const float * m, const float *  idxnn,  
        float * y, double * alpha, long n, long d, long k, long nn)
{
    
  long i;
  long D = d * k;
  /*long D2 = d*d*k;*/
  long D2 = d*(d+1)*k/2; /*chi tinh 1/2 ma tran*/
#ifdef _OPENMP
#pragma omp parallel for private (i)
  for (i = 0 ; i < n ; i++)
    compute_residual_1_new_VLAT (v + i * d, c, m, idxnn + i*nn, y + i * D2, alpha + i * k, d, k, nn);
#else
  for (i = 0 ; i < n ; i++)
    compute_residual_1_new_VLAT (v + i * d, c, m, idxnn + i*nn, y + i * D2, alpha + i * k, d, k, nn);
#endif    
     
}
/*
 * x: one samples
 * B: centroids
 * gamma: coefficient of sample x
 * phix: embedded vector of x
 * k: dimension of sample
 * n: number of centroids
 */
void compute_faemb_one_sample (const float* x, const float* B, const float* gamma, float* phix, long k, long n)
{
    
    long j, l, t, h;
    long   D = k*(k+1)/2 * n;
    for (t = 0; t < D; t ++) /**/
    {
        phix[t] = 0;
    }
    
    
    float r[k];
    
    for (j = 0; j < n; j ++) /*for each centroid*/
    {
        double nr = 0;
        /*compute residual to each centroid*/
        for (l = 0 ; l < k ; l++) 
        {
           r[l] = x[l] - B[l];
           nr += r[l] * r[l];
        }
        nr = 1 / sqrt (nr);
        /*normalize residual by its norm*/
        for (l = 0 ; l < k ; l++)
        {
            r[l] = r[l] * nr;
        }
       /*compute rr' */
        long count = 0;
        for (l = 0; l < k; l++)
            for (h = l; h < k; h++) /*only compute upper triangle*/
            {
                phix[count++] = r[l] * r[h] * gamma[j];
            }
        
        phix += k*(k+1)/2; /*move to next centroid*/
        B += k; /*move to next centroid*/
    }
}

/*
 * X: sample matrix: kxm
 * B: centroids: kxn
 * gamma_all: coefficient matrix: nxm
 * k: dimension of each sample
 * m: number of samples
 * n: number of centroids
 */
void compute_faemb(const float* X, const float* B, const float* gamma_all, float* PHIX, long k, long m, long n)
{
    long i;
    long   D = k*(k+1)/2 * n;
    for (i = 0; i < m; i++) /*embed for each sample*/
    {
        compute_faemb_one_sample (X + i * k, B, gamma_all + i * n, PHIX + i * D, k, n);
    }
}

void compute_triemb_aggsum (const float * v, const float * c, const float * m,
                           float * y, long n, long d, long k)
{
  long i, j, l,t;
  
  /* A single malloc for all the temporary vectors */
  float yres[k * d];

  for (j = 0 ; j < k * d ; j++)
    y[j] = 0;
  
  /* #pragma omp parallel for private (j) */
  for (j = 0 ; j < k ; j++) /*duyet tung centroid*/
  {
    float * yj = y + d * j;         /* output tren centroid thu j. to output result */
    float * yresj = yres + d * j;   /* temporary area */
    const float * cj = c + d * j;   /* centroid thu j. focus on centroid j */
    /* const float * mj = m + d * j; */  /* interesting part of the mean */
    
    for (i = 0 ; i < n ; i++) 
    {
      const float * vi = v + i * d; /* datapoint thu i; interesting vector */
      
      double nr = 0;
      for (l = 0 ; l < d ; l++) 
      {
        yresj[l] = vi[l] - cj[l]; /*residual cua data point j va centroid j*/
        nr += yresj[l] * yresj[l];
      }
      
      nr = 1 / sqrt (nr);
      for (l = 0; l < d; l++)
        yresj[l] = yresj[l] * nr; /*L2 norm*/
      
      for (l = 0 ; l < d ; l++)
        yj[l] += yresj[l]; /*accumulate tung data point*/
    }
  } 
    for (l = 0 ; l < d * k ; l++)
        y[l] -= n * m[l];  
   
}

 /*chieu len tung PCA cua tung cell*/
void compute_triemb_1 (const float * vtrain, const float * c, const float * m, const float * Pemb_local,
                           float * vout, long n, long d, long k)
{
  long i, j, l, t, h;
  
  for (i = 0; i < n * d * k; i ++)
      vout[i] = 0;
  FILE *f = fopen("track.txt","wt");
  for (i = 0; i < n; i++)
  {
      if (i%10000 == 0)
        fprintf(f,"%ld ",i);
      
      float * vouti = vout + i * d * k; /*output cua data point thu i*/
      
      const float * vtraini = vtrain + i * d; /* datapoint thu i*/
      
      float yi[k*d];
      
      for (t = 0; t < k * d; t++)
          yi[t] = 0;
      
      compute_triemb_aggsum (vtraini, c, m, yi, 1, d, k);
      
      for (j = 0; j < k; j ++)
      {
        float *yi_j = yi + d * j; /*cac thanh phan thuoc centroid thu j*/
        float *vouti_j = vouti + d * j;
        /*Pemb_local_j nhan voi yi_j*/
        const float *Pemb_local_j = Pemb_local + d*d*j;
        for (l = 0; l < d; l ++) /*duyet tung hang cua Pemb_local_j*/
        { 
            const float *tmp = Pemb_local_j + l*d;
            float sum = 0;
            for (h = 0; h < d; h++) /*duyet cac cot cua Pemb_local_j*/
                sum += tmp[h]*yi_j[h];
            vouti_j[l] = sum;
        }      
      }
  }
  fclose(f);
}

/*
void compute_triemb_2 (const float * vtrain, const float * c, const float * m, const float * Pemb_local,
                           float * vout, long n, long d, long k)
{
  long i, j, l, t, h;
  
  for (i = 0; i < n * d * k; i ++)
      vout[i] = 0;

  for (i = 0; i < n; i++)
  {
     
      float * vouti = vout + i * d * k;
      
      const float * vtraini = vtrain + i * d;
      
      float yi[k*d];
      
      for (t = 0; t < k * d; t++)
          yi[t] = 0;
      
      compute_triemb_aggsum (vtraini, c, m, yi, 1, d, k);
      for (t=0; t < k*d; t++)
          vouti[t] = yi[t];
  }
}
 */
 