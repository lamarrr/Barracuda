nvcc barracuda/fusion/fusion.cu -c -o fusion_device.o -I. --expt-extended-lambda
clang++ barracuda/fusion/fusion_bm.cc -c -o fusion_host.o -I.
clang++ fusion_host.o fusion_device.o -o fusion.o -lcudart -lbenchmark -pthread
./fusion.o
