#pragma once

#include "hnswalg.h"

namespace hnswlib{


template<typename dist_t>
    tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement(const void *data_point, tableint cur_c, size_t init_out,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                           int level) {

            

            if(level!=0)
                getNeighborsByHeuristic2(top_candidates, init_out);
            else 
                getNeighborsByRNGALPHA(top_candidates, init_out, 1.0);
            
            adaptive_out_[cur_c] = 2;

            if (top_candidates.size() > init_out)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(init_out);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = get_linklist0(cur_c);
                else
                    ll_cur = get_linklist(cur_c, level);

                if (*ll_cur) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx])
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }


            

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                size_t Mcurmax = level ? M_ : maxM0_;
                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
                
            
                if (level == 0) {
                    size_t cap = static_cast<size_t>(adaptive_out_[selectedNeighbors[idx]]) * M_;
                    if (cap > maxM0_) cap = maxM0_;
                    Mcurmax = cap;
                }

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > maxM0_)
                    throw std::runtime_error(
                        "Bad value of sz_link_list_other. sz_link_list_other = " 
                        + std::to_string(sz_link_list_other)
                        + ", Mcurmax = " + std::to_string(Mcurmax)
                    );

                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);


                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else {
                        
                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(
                                                            selectedNeighbors[idx]),
                                                    dist_func_param_);

                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                                candidates;

                            for (size_t j = 0; j < sz_link_list_other; j++) {
                                candidates.emplace(
                                        fstdistfunc_(getDataByInternalId(data[j]),
                                                     getDataByInternalId(selectedNeighbors[idx]),
                                                     dist_func_param_), data[j]);
                            }
                        candidates.emplace(d_max,cur_c);
                        if(level!=0)
                            getNeighborsByHeuristic2(candidates, Mcurmax);
                        else if(type_pN ==0)
                            getNeighborsByHeuristic2(candidates, Mcurmax);
                        else if(type_pN==1)
                            getNeighborsByHeuristic2(candidates, Mcurmax);
                        else if(type_pN==2)
                            getNeighborsByRNGALPHA(candidates, Mcurmax, value_pN);
                        else if (type_pN == 5) {
                            float relax_factor= 1.0f;  
                            if(start_step_pruning < adaptive_out_[selectedNeighbors[idx]]){
                                const int progress = std::min(adaptive_out_[selectedNeighbors[idx]] - start_step_pruning, step_alpha);
                                relax_factor = 1.0f + (value_pN - 1.0f) * 
                                            (static_cast<float>(progress) / static_cast<float>(step_alpha));
                            }
                            getNeighborsByRNGALPHA(candidates, Mcurmax, relax_factor);
                        }

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }
                        setListCount(ll_other, indx);
                        if (level == 0)
                            adaptive_out_[selectedNeighbors[idx]]++;
                              
                    }
                
            }
            return next_closest_entry_point;
        }


template<typename dist_t>
tableint HierarchicalNSW<dist_t>::mutuallyConnectNewElement_CSR(
    const void *data_point,
    tableint cur_c,
    size_t init_out,
    std::priority_queue<std::pair<dist_t, tableint>,
        std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst> &top_candidates) {


    getNeighborsByRNGALPHA(top_candidates, init_out, 1.0f);

    adaptive_out_[cur_c] = 2;

    if (top_candidates.size() > init_out)
        throw std::runtime_error("Should not be more than init_out candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(init_out);
    while (!top_candidates.empty()) {
        selectedNeighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();


    final_graph_[cur_c].assign(selectedNeighbors.begin(), selectedNeighbors.end());

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        const tableint des = selectedNeighbors[idx];

        std::unique_lock<std::mutex> lock(link_list_locks_[des]);

        size_t Mcurmax = maxM0_;
        {
            size_t cap = static_cast<size_t>(adaptive_out_[des]) * M_;
            if (cap > maxM0_) cap = maxM0_;
            Mcurmax = cap;
        }

        auto &nbrs = final_graph_[des];
        size_t sz_link_list_other = nbrs.size();

        if (sz_link_list_other > maxM0_)
            throw std::runtime_error(
                "Bad value of sz_link_list_other. sz_link_list_other = " +
                std::to_string(sz_link_list_other) +
                ", Mcurmax = " + std::to_string(Mcurmax));

        if (des == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");

        if (sz_link_list_other < Mcurmax) {
            nbrs.push_back(cur_c);
        } else {
 
            dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c),
                                       getDataByInternalId(des),
                                       dist_func_param_);

            std::priority_queue<std::pair<dist_t, tableint>,
                std::vector<std::pair<dist_t, tableint>>,
                CompareByFirst> candidates;

            for (size_t j = 0; j < sz_link_list_other; j++) {
                tableint v = nbrs[j];
                candidates.emplace(
                    fstdistfunc_(getDataByInternalId(v),
                                 getDataByInternalId(des),
                                 dist_func_param_),
                    v);
            }
            candidates.emplace(d_max, cur_c);

            if (type_pN == 0)
                getNeighborsByHeuristic2(candidates, Mcurmax);
            else if (type_pN == 1)
                getNeighborsByHeuristic2(candidates, Mcurmax);
            else if (type_pN == 2)
                getNeighborsByRNGALPHA(candidates, Mcurmax, value_pN);
            else if (type_pN == 5) {
                float relax_factor = 1.0f;
                if (start_step_pruning < adaptive_out_[des]) {
                    const int progress = std::min((int)adaptive_out_[des] - start_step_pruning, step_alpha);
                    relax_factor = 1.0f + (value_pN - 1.0f) *
                        (static_cast<float>(progress) / static_cast<float>(step_alpha));
                }
                getNeighborsByRNGALPHA(candidates, Mcurmax, relax_factor);
            } else {
                getNeighborsByHeuristic2(candidates, Mcurmax);
            }

            nbrs.clear();


            while (!candidates.empty()) {
                nbrs.push_back(candidates.top().second);
                candidates.pop();
            }

            adaptive_out_[des]++;
        }
    }

    return next_closest_entry_point;
}




template<typename dist_t>
tableint HierarchicalNSW<dist_t>::addPoint(const void *data_point, labeltype label, int level) {

    tableint cur_c = (tableint)label;
    unsigned snapshot_cur_element_count = (unsigned)label;

    {
        std::unique_lock<std::mutex> templock_curr(cur_element_count_guard_);
        if (cur_element_count >= max_elements_)
            throw std::runtime_error("The number of elements exceeds the specified limit");
        cur_element_count++;
        snapshot_cur_element_count = (unsigned)cur_element_count;
        label_lookup_[label] = label;
    }
    
    // update lock then element lock
    std::unique_lock<std::mutex> lock_el_update(
        link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);

    int curlevel = (with_hl) ? getRandomLevel(mult_) : 0;
    if (level > 0) curlevel = level;
    element_levels_[cur_c] = curlevel;

    std::unique_lock<std::mutex> templock(global);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) templock.unlock();

    tableint currObj = enterpoint_node_;
    tableint enterpoint_copy = enterpoint_node_;

    #ifndef CSR_BUILD
        std::memset(data_level0_memory_ + (size_t)cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        std::memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        std::memcpy(getDataByInternalId(cur_c), data_point, data_size_);
    #else
        final_graph_[cur_c].clear();
    #endif
    adaptive_out_[cur_c] = 1; 


    if (curlevel) {
        linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * (size_t)curlevel + 1);
        if (linkLists_[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
        std::memset(linkLists_[cur_c], 0, size_links_per_element_ * (size_t)curlevel + 1);
    }

    if ((signed)currObj != -1) {

   
        if (curlevel < maxlevelcopy) {
            dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
            for (int l = maxlevelcopy; l > curlevel; l--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                    linklistsizeint *data = get_linklist(currObj, l);
                    int size = (int)getListCount(data);
                    tableint *datal = (tableint *)(data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

    
        for (int l = std::min(curlevel, maxlevelcopy); l >= 0; l--) {
            std::priority_queue<
                std::pair<dist_t, tableint>,
                std::vector<std::pair<dist_t, tableint>>,
                CompareByFirst> top_candidates;

            size_t init_out = M_;

            if (l == 0) {
                int L = (int)ef_construction_;
                if (step_bw > 1) {
                    L = getL(snapshot_cur_element_count);
                    if (L == augment_bw) augment_bw = 0;
                }
           
                if (with_hl)
                    top_candidates = searchBaseLayerOpt(currObj, data_point, L);
                else
                    top_candidates = searchBaseLayerOpt(snapshot_cur_element_count, data_point, L);
      
            } else {
                top_candidates = searchBaseLayer(currObj, data_point, l);
            }
           
            
            if (l == 0) {
            #ifdef CSR_BUILD

                currObj = mutuallyConnectNewElement_CSR(data_point, cur_c, init_out, top_candidates);
            #else
                currObj = mutuallyConnectNewElement(data_point, cur_c, init_out, top_candidates, l);
            #endif

            } else {
                currObj = mutuallyConnectNewElement(data_point, cur_c, init_out, top_candidates, l);
            }
        }

    } else {
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }

    return cur_c;
}


}