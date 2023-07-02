/*
We based our optimised program on the code given originally in sales.c.
We tried a few things;
Firstly we threaded the inner loop but this slowed down the program as we had to write a crticial section containing;
	if (dist(cities, ThisPt, j) < CloseDist) {
	  CloseDist = dist(cities, ThisPt, j);
	  ClosePt = j;
	}
which only one thread could access at a time.
Secondly we tried to vectorise the function that calculates the distance from point to point with the library "xmmintrin.h"
as it offered a square vector function.
This did not work however as it was hard to manually vectories in an elegant way especially with the branch "if (!visited[j])"
It was also difficult to keep track of the cities' indexes as their distances were being mashed up in Vectors.

We came to our solution by altering the original program in a way that would allow each thread to work correctly without
having to enter a Critical Section.
Our solution lead us to first creating a ncities*ncities table which could simply done with threading with a "#pragma omp parallel for"
on the outer loop and then we discovered that a "#pragma omp simd" was optimal for the inner loop.
Once we created this table we then started creating the trip itinery by searching through the current points (ThisPt) distances and finding the closest
then update ThisPt to that point and loop.
We put a "#pragma omp SIMD" declaration above the inner loop in this this part and also declared the distance function as "#pragma omp declare simd"
as this lead to a small increase in optimisation.

When we ran the original code with 10,000 cities we averaged 2022007 microseconds
This code runs 29.5% quicker than the original, running on average at 1427421 microseconds
Load this file onto a stoker machine and run the following compiling comand using GCC
gcc -fopenmp-simd sales.c mytour.c -o a -lm
*/

#include "mytour.h"
#include <stdio.h>
#include "xmmintrin.h"
#include <omp.h>
#include <float.h>
#include "math.h"
#include <stdlib.h>

#pragma omp declare simd
float sqr_new(float x)
{
  return x*x;
}

#pragma omp declare simd
float dist_new(const point cities[], int i, int j) {
  return sqrt(sqr_new(cities[i].x-cities[j].x)+
              sqr_new(cities[i].y-cities[j].y));
}

void my_tour(const point cities[], int tour[], int ncities)
{
  int i,j,k,l;
  char *visited = alloca(ncities);
  int ThisPt, ClosePt=0;
  float CloseDist;
  int endtour=0;
  const int n = ncities;
  float **closestPts = (float **)malloc(ncities * sizeof(float *));
    for (i=0; i<ncities; i++)
         closestPts[i] = (float *)malloc(ncities * sizeof(float));

  #pragma omp simd
  for (i=0; i<ncities; i++)
    visited[i]=0;

  ThisPt = ncities-1;
  visited[ncities-1] = 1;
  tour[endtour++] = ncities-1;
  #pragma omp parallel for
  for (i=0; i<ncities; i++) {
    #pragma omp simd
    for (j=0; j<ncities; j++) {
         closestPts[i][j] = dist_new(cities,i,j);
        }
    }

  for(k=0; k<ncities; k++){
      CloseDist = DBL_MAX;
      #pragma omp simd
      for(l = 0; l<ncities; l++){
        if(!visited[l]){
          if (closestPts[ThisPt][l] < CloseDist && ThisPt != l){
              CloseDist = closestPts[ThisPt][l];
              ClosePt = l;
            }//if
          }//if
       }//loop
    tour[endtour++] = ClosePt;
    visited[ClosePt] = 1;
    ThisPt = ClosePt;
  }//loop
}
/*
James Lunt 18323467,
Cian Crowley 18328635
*/