set  autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Film Data PCA"
plot "film-data-hmm-incremental.dat" using 1:2:3 with labels font "arial,8" tc lt 12 point pt 19 ps 1
pause -1 "\nPush 'q' and 'return' to exit Gnuplot ...\n"
