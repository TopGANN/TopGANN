#pragma once
#include "../flags.h"


#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include "../PTK.h"


using namespace PTK;
namespace hnswlib {
    typedef size_t labeltype;
    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);




    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        //pure virtual(=0) means we must override these functions in derived classes
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(const void *datapoint, labeltype label)=0;
        virtual ~AlgorithmInterface(){
        }
    };

}

#include "space_l2.h"
#include "space_ip.h"
#include "hnswalg.h"
