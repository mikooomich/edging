#include <cassert>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <future>
#include <queue>

#include "utils.h"
#include "bitmap.h"
#include "sequential.h"
namespace fs = std::filesystem;

#include <sstream>
#include <stdio.h>
#include <string.h>  /* For strlen             */
#include <mpi.h>     /* For MPI functions, etc */


/*
 * Thread layout design for openmpi part
 *
 * Given a total number of threads (comm_sz) the threads will be as allocated as such:
 * [master, general workers (m worker)......, gaussian workers (g worker)......]
 *
 * master = rank 0
 * m worker = 1 ---> gaussStart - 1
 * g worker = gaussStart ---> comm-sz - 1
 *
 * gaussStart is a variable is defined with runtime args. This must be > 1, < comm-sz
 *
 */

/**
 * Running time is recorded with openmpi
 *
 * frame time: ALl kernel applications are done sequentially. The runtime for each step (aka frame) is saved separately under debugFrames (see BitmapResult)
 *
 *
 *
 * totalRuntime: The total runtime  is the time elapsed to process a single image from the time the
 * master starts sending it to a gaussian worker, to when the master finishes downloading from the main worker.
 * Includes any overhead due to message passing and/or overhead due to waiting for worker threads to become available.
 *
 *overheadTime: The overhead time is totalRuntime - the sum of the frame times
 *
 */

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

    // open mpi specific args

    /**
     * Start of gaussian worker threads
     *
     * This must be > 1, < comm-sz
     */
    int gaussStart = -1;

    // for openmpi compiled programs, this is entirely ignored
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
    if (argc > 4) {
        gaussStart = std::atof(argv[4]);
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

    int comm_sz;
    int my_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // partition must be within bounds
    assert(gaussStart < comm_sz && gaussStart > 1);

    std::cout << "I am : " << my_rank << " -- " << comm_sz << std::endl;


    /**
    * main Worker
    */
    if (my_rank > 0 && my_rank < gaussStart) {
        while (true) {
            // signal to master this thread is ready for more work
            // int hai = -1;
            // MPI_Send(&hai, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);


            // recieve data
            int picNumber = 0;
            long gaussTime;
            int width = 0;
            int height = 0;

            printf("[%d of %d, worker] Waiting from main...\n", my_rank, comm_sz - 1);
            // receive floatmap metadata
            MPI_Recv(&picNumber, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // this is the signal where master gives to say this worker is no longer needed
            if (picNumber == -1) {
                break;
            }

            MPI_Recv(&gaussTime, 1, MPI_LONG, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // time for gaussian blur. this will be passed to master eventually
            MPI_Recv(&width, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&height, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // receive image data, then reconstruct the floatmap
            std::vector<float> flat(width * height);
            MPI_Recv(flat.data(), width * height, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[%d of %d] Starting processing of image %i. w=%i h=%i\n", my_rank, comm_sz - 1, picNumber, width,
                   height);

            FloatMap resultFloatmap(width, height);
            deserialize(flat, resultFloatmap.data);
            // TODO: uhhh time stats
            BitmapResult bitmapResult = BitmapResult("hello how is it going", 0L, resultFloatmap);


            // process the image
            DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, true);
            // FloatMap gaussianKernel = s.gaussianKernel;
            FloatMap sobelKernelVert = s.sobelKernelVert;
            FloatMap sobelKernelHoriz = s.sobelKernelVert;

            // gaussian blur part takes ~80% of the processing time... so we will further parallelize that part
            // and just do the remaining processing on this worker sequentially
            processImageNoGauss(&bitmapResult, sobelKernelVert, sobelKernelHoriz);
            printf("Done processing (%d of %d). image = %i. Send back to master now\n", my_rank, comm_sz, picNumber);

            // turn the time into a string, use comma delimit
            std::string timeString = std::to_string(gaussTime) + ",";
            for (auto frame: bitmapResult.debugFrames) {
                timeString += std::to_string(frame.totalRuntime) + ",";
            }
            int len = timeString.size();

            // send result back to master with which image index was just processed
            MPI_Send(&picNumber, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // send time string
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(timeString.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

            // send image data
            FloatMap f = bitmapResult.image;
            std::vector<float> result(f.width * f.height);
            serialize(f.data, flat);
            MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    } else if (my_rank >= gaussStart) {
        /**
        * gaussian worker
        */
        while (true) {
            // recieve data
            int picNumber = 0;
            int width = 0;
            int height = 0;

            printf("[%d of %d, gauss] Waiting from main...\n", my_rank, comm_sz - 1);
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
            printf("[%d of %d] Starting processing of gauss image %i. w=%i h=%i\n", my_rank, comm_sz - 1, picNumber,
                   width,
                   height);

            FloatMap resultFloatmap(width, height);
            deserialize(flat, resultFloatmap.data);
            // TODO: uhhh time stats
            BitmapResult bitmapResult = BitmapResult("hello how is it going", 0L, resultFloatmap);


            // process the image
            DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, true);
            FloatMap gaussianKernel = s.gaussianKernel;
            // FloatMap sobelKernelVert = s.sobelKernelVert;
            // FloatMap sobelKernelHoriz = s.sobelKernelVert;

            // gaussian blur part takes ~80% of the processing time... so we will further parallelize that part
            // and just do the remaining processing on this worker sequentially

            processGaussianBlur(&bitmapResult, gaussianKernel);
            // processImage(&bitmapResult, gaussianKernel, sobelKernelVert, sobelKernelHoriz);
            printf("Done processing (%d of %d). gauss image = %i. Send back to master now\n", my_rank, comm_sz,
                   picNumber);


            // prepare data to sedn to worker
            FloatMap f = bitmapResult.image;
            std::vector<float> result(f.width * f.height);
            serialize(f.data, flat);


            // ask master for which worker is available, then send to that worker directly
            int nextWorkerId;
            MPI_Send(&picNumber, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&nextWorkerId, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


            // send result to the next worker in the pipeline
            MPI_Send(&picNumber, 1, MPI_INT, nextWorkerId, 0, MPI_COMM_WORLD);
            long processingTime = bitmapResult.debugFrames[0].totalRuntime;
            // master just needs to know it's not -1 when asking for a worker... save 1 message pass by sending over the time lol
            MPI_Send(&processingTime, 1, MPI_LONG, nextWorkerId, 0, MPI_COMM_WORLD);
            MPI_Send(&f.width, 1, MPI_INT, nextWorkerId, 0, MPI_COMM_WORLD);
            MPI_Send(&f.height, 1, MPI_INT, nextWorkerId, 0, MPI_COMM_WORLD);
            MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, nextWorkerId, 0, MPI_COMM_WORLD);

            // signal to master this thread is ready for more work
            int hai = -1;
            MPI_Send(&hai, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    } else {
        /**
        * master
        */

        std::queue<int> gaussianWorkers; // gauss workers waiting on worker
        std::queue<int> mainWorkers; // main workers that are available

        printf("[MASTER]: process is starting [%d of %d]\n", my_rank, comm_sz);
        DataSet s = prepareDataset(blur_kernel_size, blur_sigma, INDIR, false);
        std::vector<BitmapResult> files = s.files;
        printf("[MASTER]: Total images: %llu images\n", files.size());
        long startTime = getSysTime();

        int maxFiles = files.size();

        // send initial batch to gauss workers
        int initialBatchLimit = std::min((comm_sz - 1) - (gaussStart - 1), maxFiles);
        // min of worker size and total file count
        for (int i = 0; i < initialBatchLimit; i++) {
            int target = gaussStart + i;
#ifdef EDGING_DEBUG
            printf("[MASTER]: Sending to gauss worker %d. Image: %d\n", target, i);
#endif

            // temporarily store start time, when the image is finished saving, this will be updated to represent the real total time
            files[i].totalRuntime = getSysTime();

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

        // register all main workers
        for (int i = 1; i < gaussStart; i++) {
            mainWorkers.push(i);
        }

        /**
         * At this point, the master has sent all the gauss workers stuff to process. After that, the general process
         * (without going into all the nuanced details such as keeping track of worker avalibility) is:
         *
         * if gauss worker:
         *      Gauss worker finishes processing, then it messages master to ask main worker it can use.
         *          Gauss worker sends its result directly to main worker
         *      When the gauss worker is done sending to worker, it will notify master that it is done, then
         *          master sends a new image
         *
         *  if main worker:
         *      Worker processes image, then sends to master and Master Downloads the finished image, worker waits until a gauss thread sense it a new image
         *
         *
         * This while loop handles the master part of the above logic
         *
         */
        int nextImageIndex = initialBatchLimit; // starts the index after the initial send
        int completed = 0;
        while (completed < files.size()) {
            MPI_Status status;

            /**
             *Gaussian worker
             *      -1              -----> Ready for new image
             *      anything else   -----> Wants a worker rank number from master
*/
            int picNumber = 0;

#ifdef EDGING_DEBUG
            printf("[MASTER]: Receive iteration (%d of %llu)\n", completed, files.size());
#endif
            // get which picture the gaussian/main worker is sending over
            // this is also used to signal the status of the worker
            MPI_Recv(&picNumber, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

            int workerRank = status.MPI_SOURCE;
#ifdef EDGING_DEBUG
            printf("[MASTER]: RECEIVING FROM (%d of %llu)\n", workerRank, files.size());
#endif

            // assign the gauss worker a
            if (status.MPI_SOURCE >= gaussStart && picNumber != -1) {
                // send the gauss worker a main worker. if there is no available worker, save it for next iteration or when a main worker is available
                if (!mainWorkers.empty()) {
                    int nextWorkerID = mainWorkers.front();
                    mainWorkers.pop();
                    MPI_Send(&nextWorkerID, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
                } else {
                    gaussianWorkers.push(status.MPI_SOURCE);
                }
                continue;
            }

            // send new image for the gauss worker to work on
            if (status.MPI_SOURCE >= gaussStart) {
                if (nextImageIndex < files.size()) {
#ifdef EDGING_DEBUG
                    printf("[MASTER]: Sending to gauss worker %d. Image: %d\n", workerRank, nextImageIndex);
#endif

                    // temporarily store start time, when the image is finished saving, this will be updated to represent the real total time
                    files[nextImageIndex].totalRuntime = getSysTime();

                    // can only send 1d arrays, so convert 2d into 1d
                    FloatMap f = files[nextImageIndex].image;
                    std::vector<float> flat(f.width * f.height);
                    serialize(f.data, flat);


                    // send metadata: image index, width, height
                    MPI_Send(&nextImageIndex, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
                    MPI_Send(&f.width, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
                    MPI_Send(&f.height, 1, MPI_INT, workerRank, 0, MPI_COMM_WORLD);
                    // send image data
                    MPI_Send(flat.data(), f.width * f.height, MPI_FLOAT, workerRank, 0, MPI_COMM_WORLD);

                    nextImageIndex++;
                } else {
                    // done, tell worker to die
                    int dieSignal = -1;
                    MPI_Send(&dieSignal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                }
            } else {
                // download from the main worker
#ifdef EDGING_DEBUG
                printf("[MASTER]: Download from %d for image: %d\n", status.MPI_SOURCE, picNumber);
#endif

                // same receive process as how worker receive master's image
                // But we can avoid sending floatmap height/width --> assume worker will send the correct image size over
                BitmapResult *destination = &files[picNumber];

                // receive time string, then update the main data with it
                int len;
                MPI_Recv(&len, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string timeString;
                timeString.resize(len);
                MPI_Recv(timeString.data(), len, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // i miss my .split().foreach {}
                std::stringstream ss(timeString);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    destination->debugFrames.push_back(BitmapResult("", std::stol(item), FloatMap(0, 0)));
                }


                // receive image data
                std::vector<float> flat(destination->image.width * destination->image.height);
                MPI_Recv(flat.data(), destination->image.width * destination->image.height, MPI_FLOAT,
                         status.MPI_SOURCE, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);


                deserialize(flat, destination->image.data);

                // save running times. total and overhead
                destination->totalRuntime = getSysTime() - destination->totalRuntime;

                long frameRuntime = 0L;
                for (auto frame: destination->debugFrames) {
                    frameRuntime += frame.totalRuntime;
                }

                destination->overheadTime = destination->totalRuntime - frameRuntime;

                mainWorkers.push(status.MPI_SOURCE); // add back to available workers
                completed++;


#ifdef EDGING_DEBUG
                printf("[MASTER]: Download FINISHED from %d for image: %d\n", status.MPI_SOURCE, picNumber);
#endif

                // clear out any backlogs of gauss workers that are waiting on main workers
                if (!mainWorkers.empty()) {
                    // dispatch
                    while (!mainWorkers.empty() && !gaussianWorkers.empty()) {
                        int nextWorkerID = mainWorkers.front();
                        int nextGaussWorkerID = gaussianWorkers.front();
                        mainWorkers.pop();
                        gaussianWorkers.pop();

                        MPI_Send(&nextWorkerID, 1, MPI_INT, nextGaussWorkerID, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
        std::cout << "[MASTER]: is done" << std::endl;

        for (int i = 1; i < gaussStart; i++) {
            // done, tell main worker to die
            int dieSignal = -1;
            MPI_Send(&dieSignal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // save results (this step will not be analyzed)
        saveResult(files, OUTDIR, startTime,
                    /*
                     0 is always main.
                     # gauss workers = total - gaussStart
                     # main workers = total - gauss workers - 1

                   0 | 1 2 3 4 5 // 3 gaussStart, 3 gaussian workers, 2 main workers
                   0 | 1 2 3 4 5 // 2 gaussStart, 4 gaussian workers, 1 main workers
                   */
                   "Worker count: g: " + std::to_string(comm_sz - gaussStart) + ", m: " +std::to_string(comm_sz - (comm_sz - gaussStart) - 1) + "\n" +
                   "variant, blur kernel size, blur sigma\n" + std::to_string(variant) + "," +
                   std::to_string(blur_kernel_size) + "," + std::to_string(blur_sigma) + "\n");
    }


    MPI_Finalize();

    return 0;
}
