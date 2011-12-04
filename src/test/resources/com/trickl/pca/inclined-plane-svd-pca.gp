set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Raw 3D Data Plot"
splot "inclined-plane-svd-pca.dat" every ::1 
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
