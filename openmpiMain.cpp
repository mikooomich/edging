#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <future>

#include "utils.h"
#include "bitmap.h"
#include "sequential.h"
namespace fs = std::filesystem;


#include <stdio.h>
#include <string.h>  /* For strlen             */
#include <mpi.h>     /* For MPI functions, etc */


// Enable extra debug print
// Uncomment to enable, comment to disable.
#define DEBUG


/**
 * Main program entry point
 * @param argc M
 * @param argv
 * @return
 */
int main(int argc, char *argv[]) {
    std::string INDIR = "./data/input";
    std::string OUTDIR = "./data/output";

    // args from program args
    int blur_kernel_size = -1;
    float blur_sigma = -1.0f;

    // for openmpi, this is entirely ignored
    /**
     *1 = sequential
     *2 = openmp/cude
     *3 = openmpi
     */
    int variant = 3;

    if (argc > 1) {
        variant = std::atoi(argv[1]);
    }
    if (argc > 2) {
        blur_kernel_size = std::atof(argv[2]);
    }
    if (argc > 3) {
        blur_sigma = std::atof(argv[3]);
    }


    if (blur_kernel_size == -1) {
        std::cout << "WARNING: No blur_kernel_size specified. Using default value 11" << std::endl;
        blur_kernel_size = 11;
        std::cout << "Usage: whyareyourunning <blur kernel size> <blur sigma size>" << std::endl;
        std::cout << "Example: whyareyourunning 11 0.2" << std::endl;
    }


    if (blur_sigma == -1.0f) {
        std::cout << "WARNING: No blur_sigma specified. Using default value 0.2" << std::endl;
        blur_sigma = 0.2f;
        std::cout << "Usage: whyareyourunning <blur kernel size> <blur sigma size>" << std::endl;
        std::cout << "Example: whyareyourunning 11 0.2" << std::endl;
    }

    int comm_sz; /* Number of processes    */
    int my_rank; /* My process rank        */

    /* Start up MPI */
    MPI_Init(NULL, NULL);

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    std::cout << "I am : " << my_rank << " -- " << comm_sz << std::endl;


    /**
     * Worker
     */
    if (my_rank != 0) {
        while (true) {
            // recieve data
            int picNumber = 0;
            int width = 0;
            int height = 0;

            printf("[%d of %d] Waiting from main...\n", my_rank, comm_sz - 1);
            // receive floatmap metadata
            MPI_Recv(&picNumber, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // this is the signal where master gives to say this worker is no longer needed
            if (picNumber == -1) {
                break;
            }

            MPI_Recv(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&height, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // receive image data, then reconstruct the floatmap
            std::vector<float> flat(width * height);
            MPI_Recv(flat.data(), width * height, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[%d of %d] Starting processing of image %i. w=%i h=%i\n", my_rank, comm_sz - 1, picNumber, width,
                   height);

            FloatMap resultFloatmap(width, height);
            deserialize(flat, resultFloatmap.data);
            // TODO: uhhh time stats
            BitmapResult bitmapResult = BitmapResult("hello how is it going", 0L, resultFloatmap);


            // process the image
            DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, true);
            FloatMap gaussianKernel = s.gaussianKernel;
            FloatMap sobelKernelVert = s.sobelKernelVert;
            FloatMap sobelKernelHoriz = s.sobelKernelVert;

            // gaussian blur part takes ~80% of the processing time... so we will further parallelize that part
            // and just do the remaining processing on this worker sequentially

            processImage(&bitmapResult, gaussianKernel, sobelKernelVert, sobelKernelHoriz);
            printf("Done processing (%d of %d). image = %i. Send back to master now\n", my_rank, comm_sz, picNumber);

            // send result back to master with which image index was just processed
            MPI_Send(&picNumber, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            FloatMap f = bitmapResult.image;
            std::vector<float> result(f.width * f.height);
            serialize(f.data, flat);
            MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        /**
        * master
        */

        printf("[MASTER]: process is starting [%d of %d]\n", my_rank, comm_sz);
        DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, false);
        std::vector<BitmapResult> files = s.files;
        printf("[MASTER]: Total images: %llu images\n", files.size());

        int maxFiles = files.size();

        // send initial batch to workers
        int initialBatchLimit = std::min(comm_sz - 1, maxFiles); // min of worker size and total file count
        for (int i = 0; i < initialBatchLimit; i++) {
            // each worker gets every "rank th" image
            // TODO: other scheduling patterns?
            int target = i + 1;
            printf("[MASTER]: Sending to worker %d. Image: %d\n", target, i);
            // can only send 1d arrays, so convert 2d into 1d
            FloatMap f = files[i].image;
            std::vector<float> flat(f.width * f.height);
            serialize(f.data, flat);


            // send metadata: image index, width, height
            MPI_Send(&i, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            MPI_Send(&f.width, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            MPI_Send(&f.height, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
            // send image data
            MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, target, 0, MPI_COMM_WORLD);
        }

        // give workers more stuff
        int nextImageIndex = initialBatchLimit - 1; // num workers
        int completed = 0;
        while (completed < files.size()) {
            MPI_Status status;
            int picNumber = 0;

            printf("[MASTER]: Receive iteration (%d of %llu)\n", completed, files.size());
            // get which picture the worker is sending over, needed to get which index in files array to save it to
            MPI_Recv(&picNumber, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            printf("[MASTER]: Download from %d for image: %d\n", status.MPI_SOURCE, picNumber);

            // same receive process as how worker receive master's image
            // But we can avoid sending floatmap height/width --> assume worker will send the correct image size over
            BitmapResult *destination = &files[picNumber];
            std::vector<float> flat(destination->image.width * destination->image.height);
            MPI_Recv(flat.data(), destination->image.width * destination->image.height, MPI_FLOAT, status.MPI_SOURCE, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);


            deserialize(flat, destination->image.data);
            destination->totalRuntime = 69; // TODO: uhhh time stats

            completed++;
            nextImageIndex++;

            printf("[MASTER]: Download FINISHED from %d for image: %d\n", status.MPI_SOURCE, picNumber);

            // send new image for the worker to work on
            if (nextImageIndex < files.size()) {
                // each worker gets every "rank th" image
                // TODO: other scheduling patterns?
                int target = status.MPI_SOURCE;

                printf("[MASTER]: Sending to worker %d. Image: %d\n", target, nextImageIndex);
                // can only send 1d arrays, so convert 2d into 1d
                FloatMap f = files[nextImageIndex].image;
                std::vector<float> flat(f.width * f.height);
                serialize(f.data, flat);


                // send metadata: image index, width, height
                MPI_Send(&nextImageIndex, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
                MPI_Send(&f.width, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
                MPI_Send(&f.height, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
                // send image data
                MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, target, 0, MPI_COMM_WORLD);
            } else {
                // done, tell worker to die
                int dieSignal = -1;
                MPI_Send(&dieSignal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }
        }
        std::cout << "[MASTER]: is done" << std::endl;

        // save results (this step will not be analyzed)
        saveResult(files, OUTDIR);
    }


    /* Shut down MPI */
    MPI_Finalize();

    return 0;
}
