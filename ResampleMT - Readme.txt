Internaly multi-threaded resampling functions.

The functions inside this plugin are :
PointResizeMT
BilinearResizeMT
BicubicResizeMT
LanczosResizeMT
Lanczos4ResizeMT
BlackmanResizeMT
Spline16ResizeMT
Spline36ResizeMT
Spline64ResizeMT
GaussResizeMT
SincResizeMT

Parameters are exactly the same than the orignal resampling functions, and in the same order, so they are totaly
backward compatible.

A new parameter is added at the end of all the parameters :
   threads -

      Controls how many threads will be used for processing. If set to 0, threads will
      be set equal to the number of detected processors.

      Default:  0  (int)

So, syntax is :
ResampleFunction([original parameters],int threads)
