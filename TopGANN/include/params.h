#ifndef PARAM_READER_H
#define PARAM_READER_H

#include <getopt.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <sstream>
#include <vector>

struct Params {
    char* dataset = (char*)"/no_data_path";
    char* queries = (char*)"/no_query_path";
    char* gtpath = (char*)"/home/TOPGANN/ANN_gts/";
    char* index_path = (char*)"./no_index_path";
   

    unsigned int dataset_size = 1000;
    unsigned int queries_size = 5;
    unsigned int dim = 256;
    int mode = 0;
    int K = 10;
    int L = 300;

    int type_pC = 1;
    float value_pC = 1;

    int type_pN = 1;
    float value_pN = 1.2;

    int step_bw = 6;
    int augment_bw = L;
    int step_out = 3;
    int step_alpha = 10;

    int seed_graph = 1996;

    bool savecompressed = true;

    int simpleoutput = 0;
    
    int num_threads = std::thread::hardware_concurrency();
    bool allign_dim = 1;

    int with_hl = 0;
    vector<int> L_mults;
};


inline std::vector<int> parse_int_list(const std::string& str) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        if (!item.empty()) {
            result.push_back(std::atoi(item.c_str()));
        }
    }
    return result;
}
inline void parse_args(int argc, char** argv, Params& p) {
    static struct option long_options[] = {
        {"dataset", required_argument, 0, 0},
        {"queries", required_argument, 0, 0},
        {"gtpath", required_argument, 0, 0},
        {"index-path", required_argument, 0, 0},
        {"dn", required_argument, 0, 0},
        {"qn", required_argument, 0, 0},
        {"dim", required_argument, 0, 0},
        {"mode", required_argument, 0, 0},
        {"K", required_argument, 0, 0},
        {"L", required_argument, 0, 0},
        {"TPC", required_argument, 0, 0},
        {"VPC", required_argument, 0, 0},
        {"TPN", required_argument, 0, 0},
        {"VPN", required_argument, 0, 0},
        {"SBW", required_argument, 0, 0},
        {"ABW", required_argument, 0, 0},
        {"SOUT", required_argument, 0, 0},
        {"SALPHA", required_argument, 0, 0},
        {"seed-graph", required_argument, 0, 0},
        {"savecompressed", required_argument, 0, 0},
        {"simpout", required_argument, 0, 0},
        {"num-threads", required_argument, 0, 0},
        {"allign-dim", required_argument, 0, 0},
        {"L-mults", required_argument, 0, 0},
       
        {"hl", required_argument, 0, 0},
        {"help", no_argument, 0, '?'},
        {0, 0, 0, 0}
    };

    while (true) {
        int option_index = 0;
        int c = getopt_long(argc, argv, "", long_options, &option_index);
        if (c == -1) break;
        if (c == '?') exit(-1);

        std::string name = long_options[option_index].name;

        if (name == "dataset") p.dataset = optarg;
        else if (name == "queries") p.queries = optarg;
        else if (name == "gtpath") p.gtpath = optarg;
        else if (name == "index-path") p.index_path = optarg;
        else if (name == "dn") p.dataset_size = atoi(optarg);
        else if (name == "qn") p.queries_size = atoi(optarg);
        else if (name == "dim") p.dim = atoi(optarg);
        else if (name == "mode") p.mode = atoi(optarg);
        else if (name == "K") p.K = atoi(optarg);
        else if (name == "L") p.L = atoi(optarg);
        else if (name == "TPC") p.type_pC = atoi(optarg);
        else if (name == "VPC") p.value_pC = atof(optarg);
        else if (name == "TPN") p.type_pN = atoi(optarg);
        else if (name == "VPN") p.value_pN = atof(optarg);
        else if (name == "SBW") p.step_bw = atoi(optarg);
        else if (name == "ABW") p.augment_bw = atoi(optarg);
        else if (name == "SOUT") p.step_out = atoi(optarg);
        else if (name == "SALPHA") p.step_alpha = atoi(optarg);
        else if (name == "seed-graph") p.seed_graph = atoi(optarg);
        else if (name == "savecompressed") p.savecompressed = atoi(optarg);
        else if (name == "simpout") p.simpleoutput = atoi(optarg);
        else if (name == "num-threads") p.num_threads = atoi(optarg);
        else if (name == "allign-dim") p.allign_dim = atoi(optarg);
        else if (name == "L-mults") p.L_mults = parse_int_list(std::string(optarg));
        else if (name == "hl") p.with_hl = atoi(optarg);
        
        else {
            std::cerr << "Unknown option: " << name << std::endl;
            exit(-1);
        }
    }
}

#endif
