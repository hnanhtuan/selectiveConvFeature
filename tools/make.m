% Compile the Mex files require by the package

mex -outdir triemb triemb/triemb_res.c triemb/triemb.c
mex -outdir triemb triemb/triemb_sumagg.c triemb/triemb.c

mex -outdir faemb_mex faemb_mex/fa_embedding.c faemb_mex/faemb.c



