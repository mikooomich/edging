
The following is test commands used to run

General rule:
- Master is always first process
- Main worker count is n - <gauss start>
- From 0 -> gaussStart -1  is main workers
- From gaussStart -> n-1 is gauss workers


1 main
- 1 main worker, 6 gauss worker
    - mpirun -n 8 ./whyareyourunningMPI 3 11 0.2 2

- 1 main worker, 5 gauss worker
    - mpirun -n 7 ./whyareyourunningMPI 3 11 0.2 2

- 1 main worker, 4 gauss worker
    - mpirun -n 6 ./whyareyourunningMPI 3 11 0.2 2

- 1 main worker, 3 gauss worker
    - mpirun -n 5 ./whyareyourunningMPI 3 11 0.2 2

- 1 main worker, 2 gauss worker
    - mpirun -n 4 ./whyareyourunningMPI 3 11 0.2 2

- 1 main worker, 1 gauss worker
    - mpirun -n 3 ./whyareyourunningMPI 3 11 0.2 2


2 main
- 2 main worker, 6 gauss worker
    - mpirun -n 9 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 3

- 2 main worker, 5 gauss worker
    - mpirun -n 8 ./whyareyourunningMPI 3 11 0.2 3

- 2 main worker, 4 gauss worker
    - mpirun -n 7 ./whyareyourunningMPI 3 11 0.2 3

- 2 main worker, 3 gauss worker
    - mpirun -n 6 ./whyareyourunningMPI 3 11 0.2 3

- 2 main worker, 2 gauss worker
    - mpirun -n 5 ./whyareyourunningMPI 3 11 0.2 3



3 main
- 3 main worker, 10 gauss worker
    - mpirun -n 14 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 4

- 3 main worker, 9 gauss worker
    - mpirun -n 13 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 4

- 3 main worker, 8 gauss worker
    - mpirun -n 12 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 4

- 3 main worker, 7 gauss worker
    - mpirun -n 11 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 4

- 3 main worker, 6 gauss worker
    - mpirun -n 10 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 4


4 main
- 4 main worker, 15 gauss worker
    - mpirun -n 20 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 13 gauss worker
    - mpirun -n 18 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 11 gauss worker
    - mpirun -n 16 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 10 gauss worker
    - mpirun -n 15 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 9 gauss worker
    - mpirun -n 14 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 7 gauss worker
    - mpirun -n 12 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

- 4 main worker, 6 gauss worker
    - mpirun -n 11 --hostfile hostfile --oversubscribe ./whyareyourunningMPI 3 11 0.2 5

