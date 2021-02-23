#ifndef A_H
#define A_H

#include <stdio.h>
#include <stdlib.h>
// #include <iostream>
#include <time.h>
#include <string>
#include <map>
#include <mpi.h>

#define MSC 'm' // microsecond
#define SEC 's' // second
#define CLK 'c' // clock

class Timer{
private:
    double io, cpu, comm;
    clock_t io_s, io_e;
    int is_enable, is_enable_mpi, rank, c_size, show_rank;
    char unit;
    std::map<const char*, double> rec;
    std::map<const char*, clock_t> work;
public:
    Timer();
    ~Timer();
    void start_rec(const char *);
    double pause_rec(const char *);
    void clear_rec(const char *);
    double get_rec(const char *);
    void show_rec(const char *);
    bool is_in(const char *);
    double convert(clock_t , clock_t);

    void report(const char*, const char **, int);

    // Functions for MPI
    double reduce(double);
    void reduce_rec(const char *s);
    void reduce_all();

    // Set up class prperties
    Timer* set_unit(char);
    Timer* set_mpi_rank(int);
    Timer* set_mpi_comm_size(int);
    Timer* set_mpi_show_rank(int);
    Timer* enable_mpi();
    Timer* enable();
    Timer* disable();
};

#endif