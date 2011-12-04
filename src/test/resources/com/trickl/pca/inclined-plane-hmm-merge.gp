set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Inclined Plane Hall Marshall Martin (Merged) PCA Plot"
set pal rgb 15,7,5
set key off
u = 0
inc(u) = +u

splot "inclined-plane-hmm-merge.dat" using 1:2:3:4 with points lt pal
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
