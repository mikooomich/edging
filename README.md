This is a serious edging project.

## Building and running

### Manual compilation and running

1. Copy the folder `__data_template` to the project's root directory. Rename it to `data`.
2. Put image(s) to process is `data/input/`
   - IMPORTANT: all images that begin with `.` will be ignored
   - If you are using the `__data_template` folder, the 50mp images is ignored. Rename the file to remove the prefix `.` if you wish to use this. It is slow.
   - No exotic image formats please. Jpeg and PNG are recommended.
3. Compile and run the project

```bash
# compile
g++ main.cpp bitmap.cpp utils.cpp -std=c++17 -o whyareyourunning
# run
./whyareyourunning

# run with Extra arguments (optional)
# whyareyourunning <blur kernel size> <blur sigma size>
# blur kernel size defines how large the Gaussian kernel will be, sigma size defines the blur strength
./whyareyourunning 11 0.2    # kernel size of 11, sigma size of 0.2
```

Or all together

```bash
g++ main.cpp bitmap.cpp utils.cpp -std=c++17 -o whyareyourunning && ./whyareyourunning
```

4. The output images are stored in `data/output/`
    - You can view `data/output/index.html` to check the results after executing main in a graphical way.

### Alternately

(If you are grading this project, plz ignore, this is just for us to make our life easier during development)

You may use cmake, clion, etc. if you wish to. For clion, the data directory is `edging\cmake-build-debug\data`
