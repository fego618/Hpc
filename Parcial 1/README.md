# Parcial 1 - High Performance Computing

Se realizó la implementación en ~~~ CUDA c ~~~ para la multiplicación de matrices  de diferentes dimensiones, las dimensiones con las que trabajamos fueron para la matriz  A(m*n) y B(n*y), se realizó un algoritmo que realizaba la multiplicación de matrices en el host, es decir de forma secuencial y se implementó otro que realiza la multiplicación de matrices aprovechando el paralelismo que nos proporcionan las GPU usando memoria compartida haciendo uso del concepto de TILES, posteriormente se tomaron tiempos sobre diferentes dimensiones de las matrices y de los TILES sacando promedios para obtener una medida más exacta y así comprobar el rendimiento de ambos algoritmos junto con su factor de aceleración. A continuacion las tablas con su respectiva informacion



