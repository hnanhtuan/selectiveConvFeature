mex siftgeo_read.c ;
mex -g -largeArrayDims -DFINTEGER=long CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_kmax.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_kmin.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_hamming.c ../yael/hamming.c ../yael/machinedeps.c  ;
mex -g -largeArrayDims -DFINTEGER=long CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_modulate.c ../yael/embedding.c  ;

mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_kmeans.c ../yael/kmeans.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_nn.c ../yael/vector.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_L2sqr.c ../yael/binheap.c ../yael/nn.c ../yael/vector.c  ../yael/machinedeps.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 " LDFLAGS="\$LDFLAGS " yael_ivf.c ../yael/ivf.c ../yael/hamming.c ;

mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_cross_distances.c ../yael/binheap.c ../yael/nn.c ../yael/vector.c  ../yael/machinedeps.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_eigs.c ../yael/eigs.c ../yael/vector.c ../yael/matrix.c ../yael/machinedeps.c ../yael/sorting.c ../yael/binheap.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_svds.c ../yael/eigs.c ../yael/vector.c ../yael/matrix.c ../yael/machinedeps.c ../yael/sorting.c ../yael/binheap.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_gmm.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_fisher.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c ;
mex -g -largeArrayDims -DFINTEGER=long -lmwblas -lmwlapack CFLAGS="\$CFLAGS -msse4 -I.. -Wall -O3 -DHAVE_ARPACK " LDFLAGS="\$LDFLAGS  -lmwarpack" yael_fisher_elem.c ../yael/gmm.c ../yael/kmeans.c ../yael/vector.c ../yael/matrix.c ../yael/eigs.c ../yael/machinedeps.c ../yael/binheap.c ../yael/nn.c ../yael/sorting.c ;
