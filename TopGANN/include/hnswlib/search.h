#pragma once

#include "hnswalg.h"


namespace hnswlib {

        template<typename dist_t>
        pair<unsigned, float>  HierarchicalNSW<dist_t>::HLSearch(const void * query_data, querying_stats & stats){
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data,getDataByInternalId2(currObj), dist_func_param_);

            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    stats.distance_computations_hrl +=size;
                    stats.num_hops_hrl++;
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data,  getDataByInternalId2(cand),
                            dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            return make_pair(currObj,curdist);
        };

        template<typename dist_t>
        std::vector<Neighbor> 
         HierarchicalNSW<dist_t>::BeamSearch(const void *query_data,
            size_t K,
            querying_stats & stats) {

            assert(cur_element_count > 0);

            std::vector<Neighbor> best_L_nodes;
            best_L_nodes.reserve(ef_+1);
            tsl::robin_set<unsigned> inserted_into_pool;
            inserted_into_pool.reserve(ef_ * 10);
            unsigned l =0;
            unsigned k = 0;


            auto start = new PTK::Timer();



            if(this->with_hl){
            auto ep = this->HLSearch(query_data,stats);
            unsigned currObj = ep.first;
            float curdist = ep.second;
            Neighbor nn = Neighbor(currObj,curdist, true);
            inserted_into_pool.insert(currObj);
            best_L_nodes[l++] = nn;
            }else{
                while(l < K){
                    tableint random_id = rand() % cur_element_count;
                    if(inserted_into_pool.find(random_id) != inserted_into_pool.end()) continue;
                    inserted_into_pool.insert(random_id);
                    float dist = fstdistfunc_(query_data, data_ + dim_*random_id, dist_func_param_);
                    Neighbor nn = Neighbor(random_id, dist, true);
                    best_L_nodes[l++] = nn;
                }
                std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
            }
            while (k <  l) {
                unsigned nk =  l;
                if ( best_L_nodes[k].flag) {
                    best_L_nodes[k].flag = false;
                    auto n =  best_L_nodes[k].id;

                    if(stats.collect_path_data){
                    stats.search_path_ids.push_back(n);
                    stats.search_path_distances.push_back( best_L_nodes[k].distance);
                    }

                    stats.num_hops_bsl++;

                    for (unsigned m = 0; m < final_graph_[n].size(); m++) {
                        unsigned id = final_graph_[n][m];
                        if(inserted_into_pool.find(id) == inserted_into_pool.end()) {
                            inserted_into_pool.insert(id);



                            if ((m + 1) < final_graph_[n].size()) {
                                auto nextn = final_graph_[n][m + 1];
                                prefetch_vector(
                                        (const char *) ( data_ + nextn*dim_),
                                        sizeof(float) * dim_);
                            }


                            stats.distance_computations_bsl++;

                            float dist =  fstdistfunc_(query_data, data_ + dim_*id, dist_func_param_);
                            if (dist >=  best_L_nodes[ l - 1].distance && ( l == ef_))
                                continue;

                            Neighbor nn(id, dist, true);
                            unsigned r = InsertIntoPool( best_L_nodes.data(),  l, nn);

                            if ( l < ef_)
                                ++ l;
                            if (r < nk)
                                nk = r;
                        }
                    }

                    if (nk <= k)
                        k = nk;
                    else
                        ++k;
                }
                else
                    k++;
            }
            stats.time_leaves_search = start->getElapsedTime();
            return best_L_nodes;
        };

}