#include <iostream>
#include <fstream>
#include <cstdlib>
#include <queue>
#include <chrono>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>
#include <unordered_set>
#include "dim_align.h"
#include "include/hnswlib/hnswlib.h"
#include "/home/TOPGANN/ANN_gts/GT-calc/recall.h"
#include "utils.h"
#include "params.h"

using namespace std;
using namespace hnswlib;

typedef float ts_type;

int main(int argc, char** argv) {
    Params params;
    parse_args(argc, argv, params);


    char* index_full_filename = (char*)malloc(strlen(params.index_path) + 20);
    strcpy(index_full_filename, params.index_path);
    strcat(index_full_filename, "index.bin");


    int spacetype = 0;

    if (params.mode == 0) {
        

        auto t_build = new PTK::Timer();
        float * data = nullptr;

        if(params.allign_dim){
            std::size_t dim_pad = align_and_load_f32(params.dataset, &data, params.dim, params.dataset_size, 0);
            cout << "Dim All: " <<params.dim << "=>"<<dim_pad<<endl;
            params.dim = dim_pad;
        }else{
            data = (ts_type*)malloc(params.dataset_size * params.dim * sizeof(ts_type));
            read_data(params.dataset, &data, params.dim, params.dataset_size, 0);
        }

        t_build->printElapsedTime("Data Loading");

        SpaceInterface<float>* space =
            (spacetype == 0)
            ? static_cast<SpaceInterface<float>*>(new L2Space(params.dim))
            : static_cast<SpaceInterface<float>*>(new InnerProductSpace(params.dim));




        HierarchicalNSW<ts_type> appr_alg(space, params.dataset_size, params.K, params.L, params.step_out, params.seed_graph, params.with_hl);
        appr_alg.setDim(params.dim);
        appr_alg.setValuePC(params.value_pC);

        appr_alg.setTypePN(params.type_pN);
        appr_alg.setValuePN(params.value_pN);

        appr_alg.setStepBW(params.step_bw);
        appr_alg.setAugmentBW(params.augment_bw);
        appr_alg.setStepOut(params.step_out);
        appr_alg.setStepAlpha(params.step_alpha);
        appr_alg.setPruningAlphaStart(params.type_pC); //start relaxing from pr = 3
        
        appr_alg.data_ = data;

    std::cout << "TypePN: "      << params.type_pN
              << " | ValuePN: "  << params.value_pN
              << " | StepBW: "   << params.step_bw
              << " | AugmentBW: "<< params.augment_bw
              << " | StepOut: "  << params.step_out
              << " | StepAlpha: "<< params.step_alpha
              << " | StartPruning: "   << params.type_pC
              << std::endl;

        
        std::cout << "=== Insertion Construction ===" << std::endl;

        const unsigned chunks = params.dataset_size / 10;   // safe given data_size >= 10k
        #pragma omp parallel for schedule(dynamic)

        for (size_t i = 0; i < params.dataset_size; ++i) {
            if (i % chunks == 0) {
                std::cout << (i / chunks) << "0%..." << std::flush;  // no newline
            }
            appr_alg.addPoint((void*)(data + params.dim * i), i);
        }
        

        std::cout << "100%" << std::endl;
        
        t_build->printElapsedTime("Index Building");
        t_build->restart();


        #ifdef DC_IDX
                cout << "The number of distance calculations " << hnswlib::dc_counter.load(std::memory_order_relaxed) << endl;
        #endif
    
        

        appr_alg.saveIndexCompact(index_full_filename);
        t_build->printElapsedTime((std::string("Index Saving @ ") + index_full_filename).c_str());

        

        if(params.allign_dim)
            ann_aligned_free(data);
        else
            free(data);

    } else if (params.mode >= 1) {
        if (chdir(params.index_path) != 0)
            throw std::runtime_error("The index folder doesn't exist!");


        float * data = nullptr;
        float * queries = nullptr;

        if(params.allign_dim){

            std::size_t dim_pad = align_and_load_f32(params.dataset, &data, params.dim, params.dataset_size, 0);
            std::size_t dim_pad2 = align_and_load_f32(params.queries, &queries, params.dim, params.queries_size, 0);

            
            cout << "Dim All: " <<params.dim << "=>"<<dim_pad <<","<<dim_pad2<<endl;
            assert( dim_pad == dim_pad2);
            params.dim = dim_pad;

        }else{

            data = (ts_type*)malloc(params.dataset_size * params.dim * sizeof(ts_type));
            read_data(params.dataset, &data, params.dim, params.dataset_size, 0);
            
            queries = (ts_type*)malloc(params.queries_size * params.dim * sizeof(ts_type));
            read_data(params.queries, &queries, params.dim, params.queries_size, 0);
        
        }




        SpaceInterface<float>* space =
            (spacetype == 0)
            ? static_cast<SpaceInterface<float>*>(new L2Space(params.dim))
            : static_cast<SpaceInterface<float>*>(new InnerProductSpace(params.dim));

        HierarchicalNSW<ts_type> appr_alg(index_full_filename, space);
        appr_alg.data_ = data; //loadData(params.dataset);
        appr_alg.setWithHL(params.with_hl);
    
        cout << "Num. Queries: " << params.queries_size << endl;
        cout << "Num. Neighbors: " << params.K << endl;
        

        auto gt_paths = get_gt_paths(params.dataset, params.queries, appr_alg.cur_element_count, params.gtpath);
        bool gtfound = !gt_paths.first.empty() && !gt_paths.second.empty();
        if (gtfound && !params.simpleoutput) {
            cout << "ID file: " << gt_paths.first << "\n";
            cout << "Distances file: " << gt_paths.second << "\n";
        }



            for(auto Li : params.L_mults) {
                    cout << "================= L = " << Li << " =================" << endl;
                    appr_alg.setEf(Li);
                    

                    vector<vector<float>> ann_dists(params.queries_size, vector<float>(params.K));
                    vector<vector<unsigned>> ann_ids(params.queries_size, vector<unsigned>(params.K));
                    vector<size_t> dist_comps(params.queries_size);
                    vector<querying_stats> queries_stats(params.queries_size);

                    auto t_search = new PTK::Timer();
                    
                    #pragma omp parallel for
                    for (int i = 0; i < params.queries_size; ++i) {
                        querying_stats s;
                        s.reset();
                        auto tmp = appr_alg.BeamSearch(queries + i * params.dim, params.K, s);
                        
                        ann_dists[i].reserve(params.K);
                        ann_ids[i].reserve(params.K);
                        
                        for (int j = 0; j < params.K; ++j) {
                            ann_dists[i][j] = tmp[j].distance;
                            ann_ids[i][j] = tmp[j].id;
                        }
                        dist_comps[i] = s.distance_computations_hrl + s.distance_computations_bsl;
                    }
                    
                    double total_time = t_search->getElapsedTime();
                    
                    size_t total_dcs = 0;
                    for (auto v : dist_comps) total_dcs += v;

        
                    double avg_time = total_time / static_cast<double>(params.queries_size);
                    double avg_dcs  = static_cast<double>(total_dcs) / static_cast<double>(params.queries_size);

               
                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "Total time        = " << total_time << "\n";
                    std::cout << "Avg time/query    = " << avg_time   << "\n";
                    std::cout << "Total DCs         = " << total_dcs  << "\n";
                    std::cout << "Avg DCs/query     = " << avg_dcs    << "\n";

                    if (gtfound) {
                        analyze_recall(ann_dists, gt_paths.first, 1);
                        } else {
                        for (int i = 0; i < params.queries_size; ++i) 
                            printKNN(ann_dists[i].data(), params.K, queries_stats[i], ann_ids[i].data());
                        }

                    delete t_search;
            }
            

        if(params.allign_dim){
            ann_aligned_free(data);
            ann_aligned_free(queries);
        }else{
            free(data);
            free(queries);
        }
    }

    peak_memory_footprint();

    return 0;
}
