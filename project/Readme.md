Monte Carlo simulation of underlying assets for class project (ECE 598).

To use,  
(1) make (set either main.cu or main_opt.cu in Makefile compile line)
(2) ./derivPrice  
Note that kernel.cu is included in main.cu, so compiler line numbers for errors can be screwy...

The serial version is just serial.c and make -f makefileSerial
