#ifndef WTS_UTILS_H
#define WTS_UTILS_H

#include <iostream>
#include "hnswlib/hnswlib.h"
#include <getopt.h>

using namespace hnswlib;
using namespace std;

#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <getopt.h>
#include <time.h>

#include "sys/stat.h"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <getopt.h>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include "dirent.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

using namespace std;
using namespace hnswlib;
using namespace hnswlib;
using namespace std;

typedef float ts_type;
void read_data(char * dataset,
               ts_type ** pdata,
               unsigned int ts_length,
               unsigned int data_size,
               unsigned int offset)
{
    FILE *dfp = fopen(dataset, "rb");
    fseek(dfp, 0, SEEK_SET);
    fseek(dfp, (unsigned long) offset * ts_length * sizeof(ts_type), SEEK_SET);
    fread(*pdata, sizeof(ts_type), data_size * ts_length, dfp);
    fclose(dfp);
}


struct CompareByFirst{
    constexpr bool operator()(std::pair<float, tableint> const &a,
                              std::pair<float, tableint> const &b) const noexcept {
        return a.first < b.first;
    }
};
void printhelpmsg() {
    std::cout << "Usage: AGRENN [OPTIONS]\n\n"
              << "A Graph aRena for NN \n\n"
              << "Options:\n"
              << "  -d, --dataset <FILE>        Input dataset file (default: /no_data_path)\n"
              << "  -q, --queries <FILE>        Query file (default: /no_query_path)\n"
              << "  --dn, --dataset-size <INT>  Number of data points in the dataset (default: 1000)\n"
              << "  --qn, --queries-size <INT>  Number of query points (default: 5)\n"
              << "  --dim <INT>                 Dimensionality of data points (default: 256)\n"
              << "  -p, --index-path <PATH>     Path to save/load index (default: default_index_path/)\n"
              << "  -m, --mode <INT>            Mode of operation (0: indexing, 1: search) (default: 0)\n"
              << "  -k, --K <INT>               Number of nearest neighbors to find (default: 10)\n"
              << "  -l, --L <INT>               Beamwidth (number of candidate neighbors to consider) (default: 300)\n"
              << "  --SS <INT>                  Search strategy (0: SN, 1: KS, 2: DST) (default: 0)\n"
              << "  --AL <INT>                  Space (default: l2 = 0; MIPS = 1)\n"
              << "  --VAC <INT>                 Use visited nodes as candidate NN (default: 0)\n"
              << "  --MVAC <INT>                Max visited candidate nodes (default: 0)\n"
              << "  --TPC <INT>                 Type of candidate pruning (0: farthest, 1: RND, 2: RRND, 3: MOND, 4: TRND) (default: 1)\n"
              << "  --VPC <FLOAT>               Value for candidate pruning (default: 1.0)\n"
              << "  --GNN <INT>                 Get NN through expansion (default: 0)\n"
              << "  --TPN <INT>                 Type of neighbor pruning (0: farthest, 1: RND, 2: RRND, 3: MOND, 4: TRND) (default: 1)\n"
              << "  --VPN <FLOAT>               Value for neighbor pruning (default: 1.0)\n"
              << "  --CCDFS <INT>               Check connectivity with DFS (0: no, >0: use multi DFS) (default: 0)\n"
              << "  -?, --help                  Display this help message\n\n"
              << "Example:\n"
              << "  AGRENN --dataset /home/iazizi/DATA/data_size1GB_deep1b_len96_znorm.bin --queries /home/iazizi/DATA/data_size1GB_deep1b_len96_znorm.bin --dn 1000 --qn 5 --dim 96 --index-path ../index_path/ --mode 0 --K 10 --L 300 --SS 0 --AL 0 --VAC 0 --TPC 1 --VPC 1.0 --GNN 0 --TPN 1 --VPN 1.0 --CCDFS 0\n";
}
void printKNN(float * results, int k, querying_stats stats,unsigned  int * ids, int space=0)
    {

       cout << "----------"<<k<<"-NN RESULTS----------- | visited nodes : \n";
        for(int i = 0 ; i < k ; i++){

                float dist =(space==0)?sqrt(results[i]) : results[i];
            
            if(i ==0)
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu | L1Miss : %lu | L2Miss : %lu | L3Miss : %lu \n",i+1,dist,
                        ids[i],stats.time_leaves_search,stats.distance_computations_bsl+stats.distance_computations_hrl
                        ,stats.distance_computations_hrl,stats.distance_computations_bsl,
                        stats.saxdist_computations_hsl+stats.saxdist_computations_bsl,stats.saxdist_computations_hsl,
                        stats.saxdist_computations_bsl, stats.num_hops_hrl, stats.num_hops_bsl,
                        stats.l1_cache_misses, stats.l2_cache_misses, stats.l3_cache_misses
                        );
            else
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu | L1Miss : %lu | L2Miss : %lu | L3Miss : %lu \n",i+1,dist,
                        ids[i],0.0,0,0,0,0,0,0,0,0,0,0,0);
        }
   

}





#endif 
