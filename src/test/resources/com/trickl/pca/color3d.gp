set  autoscale                        # scale axes automatically
set datafile separator ","
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Color 3D Data Plot"
set pm3d
set pal rgb 15,7,5
set key off
u = 0
inc(u) = +u

splot "inclined-plane-hmm.dat" using 1:2:3:inc(u) with points lt pal
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
