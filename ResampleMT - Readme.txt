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

Several new parameters are added at the end of all the parameters :

   threads -

      Controls how many threads will be used for processing. If set to 0, threads will
      be set equal to the number of detected logical or physical cores,according logicalCores parameter.

      Default:  0  (int)

   logicalCores -
      If threads is set to 0, it will specify if the number of threads will be the number
      of logical CPU (true) or the number of physical cores (false). If your processor doesn't
      have hyper-threading or threads<>0, this parameter has no effect.

      Default: true (bool)

   MaxPhysCore -
      If true, the threads repartition will use the maximum of physical cores possible. If your
      processor doesn't have hyper-threading or the SetAffinity parameter is set to false,
      this parameter has no effect.

      Default: true (bool)

   SetAffinity -
      If this parameter is set to true, the pool of threads will set each thread to a specific core,
      according MaxPhysCore parameter. If set to false, it's leaved to the OS.

      Default: false (bool)


The logicalCores, MaxPhysCore and SetAffinity are parameters to specify how the pool of thread will be created,
allowing if necessary each people to tune according his configuration.

So, syntax is :
ResampleFunction([original parameters],int threads, bool logicalCores, bool MaxPhysCore, bool SetAffinity)

