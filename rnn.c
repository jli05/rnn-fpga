#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


/*  Multiply dense matrix a by sparse vector x

    m, n:   dense matrix a is of shape m-by-n
    a:      dense matrix
    cx:     coordinates in sparse vector x with non-zero values
    x:      non-zero values in sparse vector x
    nnz_x:  num non-zeros in length-n sparse vector x
    w:      length-m dense vector for result
*/
void mv(int m, int n, float *a, int *cx, float*x, int nnz_x,
        float *w)
{
  assert(nnz_x <= n);

  int h, i, j;
  memset(w, 0, sizeof(float) * m);
  for (h = 0; h < nnz_x; ++h)
  {
    // cx[h]-th column of a is to be multiplied
    j = cx[h];
    for (i = 0; i < m; ++i)
      w[i] += a[j * m + i] * x[h];
  }

}

/* Take ReLU and store result as sparse vector

    m:      length of input dense vector
    w:      input dense vector
    cy:     coordinates in sparse vector y
    y:      non-zero values in sparse vector y
    nnz_y:  num non-zeros in length-m sparse vector y
*/
void relu(int m, float *w, int *cy, float *y, int *nnz_y)
{
  int h, i;
  h = 0;
  for (i = 0; i < m; ++i)
    if (w[i] > 1e-12)
    {
      cy[h] = i;
      y[h] = w[i];
      ++h;
    }
  *nnz_y = h;
}

int main ( )
{
  float *a, *x, *y;
  int *cx, *cy, nnz_x, nnz_y;
  int m, n;

  m = 3; /* the number of rows */
  n = 3; /* the number of columns */

  a = (float *)malloc(sizeof(float)*m*n);
  x = (float *)malloc(sizeof(float)*n);
  y = (float *)malloc(sizeof(float)*m);
  cx = (int *)malloc(sizeof(int) * n);
  cy = (int *)malloc(sizeof(int) * m);

  // Column-major matrix storage as in Fortran
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
  nnz_x = 2;
  cx[0] = 0;
  cx[1] = 1;
  x[0] = 1;
  x[1] = 2;

  // working memory
  float *w;
  w = (float *)malloc(sizeof(float) * m);

  /* Do three times
   *
   *    x <- ReLU(A * x)
   *
   */
  for (int step = 0; step < 3; ++step)
  {
    // matrix-vector multiplication
    mv(m, n, a, cx, x, nnz_x, w);
    // take ReLU and store result as sparse vector
    relu(m, w, cy, y, &nnz_y);
    // copy y to x
    memcpy(cx, cy, sizeof(int) * nnz_y);
    memcpy(x, y, sizeof(float) * nnz_y);
    nnz_x = nnz_y;

    /* Print y */
    printf("Step %d\n", step + 1);
    for(int h = 0; h < nnz_y; ++h)
      printf(" y%d = %f\n", cy[h], y[h]);
  }

   free(a);
   free(x);
   free(y);
   free(cx);
   free(cy);
   free(w);
   return 0;
}
