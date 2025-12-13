This is a serious edging project.

## Building and running

### Manual compilation and running

1. Copy the folder `__data_template` to the project's root directory. Rename it to `data`.
2. Put image(s) to process is `data/input/`
    - IMPORTANT: all images that begin with `.` will be ignored
    - If you are using the `__data_template` folder, the 50mp images is ignored. Rename the file to remove the prefix
      `.` if you wish to use this. It is slow.
    - No exotic image formats please. Jpeg and PNG are recommended.
3. Compile and run the project

Sequential
```bash
# compile
g++ main.cpp sequential.cpp bitmap.cpp utils.cpp -std=c++17 -O2 -o whyareyourunning

# compile debug
g++ main.cpp sequential.cpp bitmap.cpp utils.cpp -std=c++17 -g -Wall -o whyareyourunning

# run
./whyareyourunning

# run with Extra arguments (optional)
# whyareyourunning <variant> <blur kernel size> <blur sigma size>
# blur kernel size defines how large the Gaussian kernel will be, sigma size defines the blur strength
# Variants: 1 --> sequential. 2 --> openmp/cuda. 3 --> openmpi
./whyareyourunning 1 11 0.2    # run sequential, kernel size of 11, sigma size of 0.2
```

Or all together

```bash
g++ main.cpp sequential.cpp bitmap.cpp utils.cpp  -std=c++17 -O2 -o whyareyourunning && ./whyareyourunning
```

CUDA
# run the program
# experiment data will be stored in cuda/analysis
```bash
nvcc cuda/cudaSobelAPI.cu cuda/bitmap.cpp -o temp/cuso_api && ./temp/cuso_api
```

# run the integrity test
```bash
nvcc cuda/cudaSobelTest.cu cuda/bitmap.cpp -o temp/cuso_test && ./temp/cuso_test
```

OpenMPI

```bash
# compile. We will assume you have a working compiler that compiles openmp fine, and system that runs openmp fine
mpicxx -std=c++17 -O2 -o whyareyourunningMPI openmpiMain.cpp sequential.cpp bitmap.cpp utils.cpp

# compile debug
#mpicxx -g -Wall -std=c++17 -o whyareyourunningMPI openmpiMain.cpp sequential.cpp bitmap.cpp utils.cpp


# run with 5 workers (2 main, 4 gaussian)
mpirun -n 6 ./whyareyourunningMPI 3 11 0.2 3

# run with 8 workers (3 main, 12 gaussian)
mpirun -n 16 ./whyareyourunningMPI 3 11 0.2 4
```

Setup openmpi on fedora
```bash
#sudo dnf install openmpi openmpi-devel

# load the modules
source /etc/profile.d/modules.sh
module load mpi/openmpi-x86_64

# check/test
ompi_info
```

4. The output images are stored in `data/output/`
    - You can view `data/output/index.html` to check the results after executing main in a graphical way.

### Alternately

(If you are grading this project, plz ignore, this is just for us to make our life easier during development)

You may use cmake, clion, etc. if you wish to. For clion, the data directory is `edging\cmake-build-debug\data`

## Important notes when running

All images are loaded into memory *before* and processing. For large datasets, you should be aware of your memory usage.

For example, 355 1920x1080 images use ~2.8 GB of memory overall (including overhead).

When running, up to an additional `[num threads] * [pixel count] * 3 * 7` bytes of memory will be used. 
- For a single 1920x1080 image and single thread exception, this equates to roughly 43.54 MB.


## Debug stuff
You can enable/disable extra print/file saving in utils.h


EDGING_DEBUG enables extra debugging/verbose prints

SAVE_PROCESS_FRAMES is intended for algorithm debugging. Do not enable otherwise. This will roughly 7x your memory usage after execution as all images are saved.
