#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cblas.h"

int main ( )
{
   enum CBLAS_ORDER order;
   enum CBLAS_TRANSPOSE transa;

   float *a, *x, *y;
   float alpha, beta;
   int m, n, lda, incx, incy, i;

   order = CblasColMajor;
   transa = CblasNoTrans;

   m = 3; /* Size of Column ( the number of rows ) */
   n = 3; /* Size of Row ( the number of columns ) */
   lda = 3; /* Leading dimension of 5 * 4 matrix is 5 */
   incx = 1;
   incy = 1;
   alpha = 1;
   beta = 0;

   a = (float *)malloc(sizeof(float)*m*n);
   x = (float *)malloc(sizeof(float)*n);
   y = (float *)malloc(sizeof(float)*n);
   /* The elements of the first column */
   a[0] = 1;
   a[1] = 2;
   a[2] = 3;
   /* The elements of the second column */
   a[m] = 1;
   a[m+1] = -4;
   a[m+2] = -1;
   /* The elements of the third column */
   a[2 * m] = 1;
   a[2 * m + 1] = 1;
   a[2 * m + 2] = 2;
   /* The elemetns of x and y */
   x[0] = 1;
   x[1] = 2;
   x[2] = 0;
   y[0] = 0;
   y[1] = 0;
   y[2] = 0;

   /* Do three times
    *
    *    x <- ReLU(A * x)
    *
    */
   for (int step = 0; step < 3; ++step)
   {
     // matrix-vector multiplication
     cblas_sgemv( order, transa, m, n, alpha, a, lda, x, incx, beta,
                  y, incy );
     // ReLU activation function
     for (i = 0; i < n; ++i)
       y[i] = fmaxf(y[i], 0);
     // copy y to x
     memcpy(x, y, sizeof(float) * n);

     /* Print y */
     printf("Step %d\n", step + 1);
     for( i = 0; i < n; i++ )
	printf(" y%d = %f\n", i, y[i]);
   }

   free(a);
   free(x);
   free(y);
   return 0;
}
