
set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
set datafile separator ","
set grid

# Creates output directory if handled by shell, but here we assume report/images exists
# (We created it in python script previously, but let's ensure it exists via command)

# F1 a
set output 'report/images/f1_a.png'
set title "F1 (Raw) vs a"
set xlabel "a"
set ylabel "Jaccard"
plot 'report/data/f1_a.csv' using 1:2 notitle with lines lw 2

# F2 a
set output 'report/images/f2_a.png'
set title "F2 (Mode) vs a"
plot 'report/data/f2_a.csv' using 1:2 notitle with lines lw 2

# F3 a
set output 'report/images/f3_a.png'
set title "F3 (MedK) vs a"
plot 'report/data/f3_a.csv' using 1:2 notitle with lines lw 2

# F4 a
set output 'report/images/f4_a.png'
set title "F4 (MedP) vs a"
plot 'report/data/f4_a.csv' using 1:2 notitle with lines lw 2

# F1 t
set output 'report/images/f1_t.png'
set title "F1 (Raw) vs t"
set xlabel "t"
plot 'report/data/f1_t.csv' using 1:2 notitle with lines lw 2

# F2 t
set output 'report/images/f2_t.png'
set title "F2 (Mode) vs t"
plot 'report/data/f2_t.csv' using 1:2 notitle with lines lw 2

# F3 t
set output 'report/images/f3_t.png'
set title "F3 (MedK) vs t"
plot 'report/data/f3_t.csv' using 1:2 notitle with lines lw 2

# F4 t
set output 'report/images/f4_t.png'
set title "F4 (MedP) vs t"
plot 'report/data/f4_t.csv' using 1:2 notitle with lines lw 2
