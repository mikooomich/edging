This is a serious edging project.

## Building and running

### Manual compilation and running

1. Copy the folder `__data_template` to the project's root directory. Rename it to `data`.
2. Put image(s) to process is `data/input/`
3. Compile and run the project

```bash
# compile
g++ main.cpp bitmap.cpp -std=c++17 -o whyareyourunning
# run
./whyareyourunning
```

Or all together

```bash
g++ main.cpp bitmap.cpp -std=c++17 -o whyareyourunning && ./whyareyourunning
```

4. The output images are stored in `data/output/`
    - You can view `data/output/index.html` to check the results after executing main in a graphical way.

### Alternately

(If you are grading this project, plz ignore, this is just for us to make our life easier during development)

You may use cmake, clion, etc. if you wish to. For clion, the data directory is `edging\cmake-build-debug\data`
