//
// Created by ilias.azizi on 02/04/24.
//


#include "utils.h"

//
// Created by ilias.azizi on 02/04/24.
//

#include <iostream>
#include "hnswlib/hnswlib.h"
#include <getopt.h>

using namespace hnswlib;
using namespace std;

typedef float ts_type;


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

#include <unordered_set>
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <random>
#include <algorithm>
#include <unordered_set>
using namespace std;
using namespace std;
using namespace hnswlib;




void printKNN(float * results, int k, querying_stats stats,unsigned  int * ids= nullptr){
    if(ids == nullptr){
        cout << "----------"<<k<<"-NN RESULTS----------- | visited nodes : \n";
        for(int i = 0 ; i < k ; i++){
            if(i ==0)
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu \n",i+1,sqrt(results[i]),
                        0,stats.time_leaves_search,stats.distance_computations_bsl+stats.distance_computations_hrl
                        ,stats.distance_computations_hrl,stats.distance_computations_bsl,
                        stats.saxdist_computations_hsl+stats.saxdist_computations_bsl,stats.saxdist_computations_hsl,
                        stats.saxdist_computations_bsl, stats.num_hops_hrl, stats.num_hops_bsl);
            else
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu \n",i+1,sqrt(results[i]),
                        0,0,0,0,0,0,0,0,0,0);


        }
    }else{
        cout << "----------"<<k<<"-NN RESULTS----------- | visited nodes : \n";
        for(int i = 0 ; i < k ; i++){
            if(i ==0)
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu \n",i+1,sqrt(results[i]),
                        ids[i],stats.time_leaves_search,stats.distance_computations_bsl+stats.distance_computations_hrl
                        ,stats.distance_computations_hrl,stats.distance_computations_bsl,
                        stats.saxdist_computations_hsl+stats.saxdist_computations_bsl,stats.saxdist_computations_hsl,
                        stats.saxdist_computations_bsl, stats.num_hops_hrl, stats.num_hops_bsl);
            else
                printf( " K N°%i  => Distance : %f | Node ID : %lu | Time  : %f |  "
                        "Total DC : %lu | HDC : %lu | BDC : %lu | "
                        "Total LBDC : %lu | HLBDC : %lu | BLBDC : %lu | HLhops : %lu | BLhops : %lu \n",i+1,sqrt(results[i]),
                        ids[i],0,0,0,0,0,0,0,0,0);


        }
    }

}
void peak_memory_footprint() {

    unsigned iPid = (unsigned)getpid();

    std::cout<<"PID: "<<iPid<<std::endl;

    std::string status_file = "/proc/" + std::to_string(iPid) + "/status";
    std::ifstream info(status_file);
    if (!info.is_open()) {
        std::cout << "memory information open error!" << std::endl;
    }
    std::string tmp;
    while(getline(info, tmp)) {
        if (tmp.find("Name:") != std::string::npos || tmp.find("VmPeak:") != std::string::npos || tmp.find("VmHWM:") != std::string::npos)
            std::cout << tmp << std::endl;
    }
    info.close();

}

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