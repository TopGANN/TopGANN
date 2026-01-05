#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <chrono>
#include <omp.h>
#include <string>

#include "../tsl/robin_set.h"

#include "neighbors.h"
#include "prefetching.h"

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
    using namespace PTK;
    using namespace std;
    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s) {

        }


        HierarchicalNSW(SpaceInterface<dist_t> *s,
                        size_t max_elements, size_t M = 16,
                        size_t ef_construction = 200, int so = 2,int random_seed = 100, bool with_hl = true) :
                link_list_locks_(max_elements),
                link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {
            max_elements_ = max_elements;

            has_deletions_=false;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            step_out = so;
            maxM_ = M_;
            maxM0_ = M_ * step_out;

            //if(so_ > 2)
            //to allow second maxout > init maxout
            adaptive_out_ = vector<unsigned short>(max_elements_, 0); // 1 by default to count the candidate pruning

            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);
            
            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            cur_element_count = 0;
            
            #ifdef CSR_BUILD
            data_level0_memory_ = nullptr; 
            final_graph_.resize(max_elements_);
            #else
            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");
            final_graph_.clear();
            #endif


            visited_list_pool_ = new VisitedListPool(1, max_elements);

            

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            this->with_hl = with_hl;
            if(with_hl){
            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
            }else{
                linkLists_ = nullptr;
                size_links_per_element_ = 0;
                mult_ = 0;
                revSize_ = 0;
            }
        }


        struct CompareByFirst{
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };
        struct CompareByFirst2{
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first > b.first;
            }
        };

        ~HierarchicalNSW() {

            if(data_level0_memory_ != nullptr)
                free(data_level0_memory_);
            else{
                std::vector<std::vector<tableint>>().swap(final_graph_);
            }
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }


        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        bool with_hl;
        double mult_, revSize_;
        int maxlevel_;



        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;


        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;

        bool has_deletions_;



        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;



        float * data_;
        size_t dim_;
        std::vector<std::vector<unsigned>>  final_graph_;
        std::mutex global;
        size_t ef_;

        
        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        void setDim(unsigned int dim)
        {
            dim_ = dim;
        };





/////////////////////////////////////// SETTING II PARAMS///////////////////////////////
        int type_pC; // 0:farthest ; 1:RND ; 2:RRND; 3:MOND 
        float value_pC;  // Pruning the Candidate NN sets
        int type_pN; //  0:farthest ; 1:RND ; 2:RRND; 3:MOND
        float value_pN; // value for ND approach, i.e RND=>1; RRND=>1.2; MOND=>60; TRND=>4

        int step_bw;// adaptive L
        int augment_bw;// augment beam
        int step_out;// adaptive out
        int step_alpha;// adaptive alpha
        int start_step_pruning;// step to start pruning
        std::vector<unsigned short> adaptive_out_;
        int max_adaptive_out_;


        void setEf(size_t ef) {
            ef_ = ef;
        }

        void setTypePC(int type_pC) {
            this->type_pC = type_pC;
        }
        void setValuePC(float value_pC) {
            double kPi = 3.14159265358979323846264;
            if(type_pC==3)
                this->value_pC  = std::cos(value_pC / 180 * kPi);
            else
                this->value_pC = value_pC;
        }

        void setTypePN(int type_pN) {
            this->type_pN = type_pN;
        }

        void setValuePN(float value_pN) {
            double kPi = 3.14159265358979323846264;
            if(type_pN==3)
                this->value_pN  = std::cos(value_pN / 180 * kPi);
            else
                this->value_pN = value_pN;
        }

        void setStepBW(int step) {
            step_bw = step;
        }
        void setAugmentBW(int augment) {
            augment_bw = augment;
        }
        void setStepOut(int step) {
            step_out = step;
        }
        void setStepAlpha(int step) {
            step_alpha = step;
        }
        void setPruningAlphaStart(int step_start) {
            //if start_step == 1 => we will relax pruning at pr = 3
            //if start_step == 0 => we will relax pruning at pr = 2
            start_step_pruning = step_start;
        }


        void setWithHL(bool with_hl_) {
            with_hl = with_hl_;
        }
   
//////////////////////////////////////////END SETTING II PARAMS///////////////////////////////


/////////////////////////////////////////FLAGS////////////////////////////////////////
        bool count_num_prune = 0;
        vector<size_t> num_prune;
        bool save_search_path = 0;
        vector<unsigned> search_path;
        bool save_search_distances = 0;
        vector<float> search_distances;
        bool count_long_range = 0;

    void set_count_num_prune(bool p) {
        count_num_prune = p;
        if (count_num_prune) {
            num_prune.resize(max_elements_, 0);
        }
    }

#include <string>
#include <cstdio>   // std::fwrite
  void index_stats() {
    vector<unsigned> indeg(max_elements_, 0);
    vector<vector<float>> dist_nns(max_elements_);
    vector<double> rc_nns(max_elements_, 0);

    // Parallel loop for dist_nns and rc_nns
    #pragma omp parallel for
    for (int i = 0; i < max_elements_; i++) {
        const char* data = getDataByInternalId(i);
        std::vector<float> dists;

        int* link_data = (int*) get_linklist0(i);
        size_t len = getListCount((linklistsizeint*) link_data);
        tableint* neighbors = (tableint*) (link_data + 1);

        float sum = 0.0f, max_dist = 0.0f;

        for (size_t j = 0; j < len; j++) {
            tableint neighbor = neighbors[j];
            const char* neighbor_data = getDataByInternalId(neighbor);
            float dist = fstdistfunc_(data, neighbor_data, dist_func_param_);
            dists.push_back(dist);
            sum += dist;
            if (dist > max_dist) max_dist = dist;
        }

        dist_nns[i] = std::move(dists);
        rc_nns[i] = (max_dist > 0 && len > 0) ? sum / (len * max_dist) : 0.0;
    }

    // Serial loop for indegree
    for (int i = 0; i < max_elements_; i++) {
        int* link_data = (int*) get_linklist0(i);
        size_t len = getListCount((linklistsizeint*) link_data);
        tableint* neighbors = (tableint*) (link_data + 1);
        for (size_t j = 0; j < len; j++) {
            indeg[neighbors[j]]++;
        }
    }

    std::ios::sync_with_stdio(false);
    std::cout.tie(nullptr);
    std::string buf;
    buf.reserve(static_cast<size_t>(max_elements_) * 64);

    buf.append(">>STATS>>\n");

    for (int i = 0; i < max_elements_; i++) {
        int* link_data = (int*) get_linklist0(i);
        size_t len = getListCount((linklistsizeint*) link_data);
        int outdeg = static_cast<int>(len);

        // node id
        buf.append(std::to_string(i));
        buf.push_back(',');

        // hierarchical (1 or 0)
        buf.append(element_levels_[i] > 0 ? "1," : "0,");

        // indegree
        buf.append(std::to_string(indeg[i]));
        buf.push_back(',');

        // num pruned
        buf.append(std::to_string(adaptive_out_[i]));
        buf.push_back(',');

        // relative contrast
        buf.append(std::to_string(rc_nns[i]));
        buf.push_back(',');

        // outdegree
        buf.append(std::to_string(outdeg));
        buf.push_back('\n');
    }

    buf.append("EndSTATS\n");

    std::fwrite(buf.data(), 1, buf.size(), stdout);

    }



////////////////////////////////////////////////////////////////////////////////////////

        inline labeltype getExternalLabel(tableint internal_id) const {
            #ifdef CSR_BUILD
            return internal_id;
            #else
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
            #endif
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            #ifdef CSR_BUILD
            return internal_id;
            #else
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
            #endif
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            #ifdef CSR_BUILD
            return reinterpret_cast<char*>(data_ + internal_id * dim_);
            #else
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
            #endif
            }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }

        int getL(size_t size){
            
            int chunkIndex = static_cast<int>(floor(static_cast<double>(size) * step_bw / max_elements_));
            if (chunkIndex >= step_bw) {
                chunkIndex = step_bw - 1;//
            }
            int Li = static_cast<int>((static_cast<double>(chunkIndex + 1) / step_bw) * ef_construction_);
            //if(size%10000==0)
            //            std::cerr << "Inserting element: " << size
            //                  <<"Step="<<step_bw  << " with Li="<<Li<<", L="<<max(Li, (int) M_)<<std::endl;
            return max(Li, (int) M_);
        }


         std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayerOpt(tableint ep_id, const void *data_point, int L) 
    {

    int layer = 0;
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
        std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst> top_candidates;

    std::priority_queue<std::pair<dist_t, tableint>,
        std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst> visited;

    std::priority_queue<std::pair<dist_t, tableint>,
        std::vector<std::pair<dist_t, tableint>>,
        CompareByFirst> candidateSet;

    dist_t lowerBound, dist;

    if (with_hl) {
        dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        if (augment_bw) visited.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
        visited_array[ep_id] = visited_array_tag;
    } else {
        std::uniform_int_distribution<tableint> distribution(0, static_cast<tableint>(ep_id - 1));
        for (int i = 0; i < L; i++) {
            tableint random_id = distribution(level_generator_);
            if (visited_array[random_id] == visited_array_tag) continue;
            visited_array[random_id] = visited_array_tag;

            dist_t dist1 = fstdistfunc_(data_point, getDataByInternalId(random_id), dist_func_param_);

            candidateSet.emplace(-dist1, random_id);
            top_candidates.emplace(dist1, random_id);
            if (augment_bw) visited.emplace(dist1, random_id);
        }
        lowerBound = top_candidates.top().first;
    }

    while (!candidateSet.empty()) {
        std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

        if ((-curr_el_pair.first) > lowerBound) break;

        candidateSet.pop();

        tableint curNodeNum = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

        int *data = nullptr;
        tableint *datal = nullptr;
        size_t size = 0;

#ifdef CSR_BUILD
        size = final_graph_[curNodeNum].size();
        datal = (size ? (tableint*)final_graph_[curNodeNum].data() : nullptr);
#else
        data = (int*)get_linklist0(curNodeNum);
        size = getListCount((linklistsizeint*)data);
        datal = (tableint*)(data + 1);
#endif

#ifdef USE_SSE
        // FIX: only prefetch if datal is valid and size is enough
        if (size > 0) {
            _mm_prefetch((char*)(visited_array + datal[0]), _MM_HINT_T0);
            _mm_prefetch((char*)(visited_array + datal[0] + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(datal[0]), _MM_HINT_T0);
            if (size > 1) {
                _mm_prefetch(getDataByInternalId(datal[1]), _MM_HINT_T0);
            }
        }
#endif

        for (size_t j = 0; j < size; j++) {
            tableint candidate_id = datal[j];

#ifdef USE_SSE
            // FIX: bounds check
            if (j + 1 < size) {
                _mm_prefetch((char*)(visited_array + datal[j + 1]), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(datal[j + 1]), _MM_HINT_T0);
            }
#endif

            if (visited_array[candidate_id] == visited_array_tag) continue;
            visited_array[candidate_id] = visited_array_tag;

            char *currObj1 = getDataByInternalId(candidate_id);
            dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);

            if (augment_bw > 0) {
                if (visited.size() < (size_t)augment_bw || dist1 < visited.top().first) {
                    visited.emplace(dist1, candidate_id);
                    if (visited.size() > (size_t)augment_bw) visited.pop();
                }
            }

            if (top_candidates.size() < (size_t)L || lowerBound > dist1) {
                candidateSet.emplace(-dist1, candidate_id);

#ifdef USE_SSE
                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                top_candidates.emplace(dist1, candidate_id);
                if (top_candidates.size() > (size_t)L) top_candidates.pop();
                if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
            }
        }
    }

    visited_list_pool_->releaseVisitedList(vl);

    if (augment_bw > 0) return visited;
    return top_candidates;
}


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;

                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
 
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0) {
                    data = (int*)get_linklist0(curNodeNum);
                } else {
                    data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
                #ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(datal) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
                #endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }





        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level) const {
            return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        L2Space * l2space_;

        
        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }


      
////////////////////

        HierarchicalNSW(const std::string &location, SpaceInterface<dist_t> * space) {
            loadIndexCompact(location, space);
        }

        SpaceInterface<float> * space_;

        void saveIndexCompact(const std::string &location);
        void loadIndexCompact(const std::string &location, SpaceInterface<dist_t> *s);

        void loadData(const char *filename);

        inline float * getDataByInternalId2(unsigned id) const {
            return data_ + id * dim_;
        };
        
        pair<unsigned, float> HLSearch(const void * query_data, querying_stats & stats) ;

        std::vector<Neighbor> BeamSearch(const void *query_data,
            size_t K,
            querying_stats & stats);



/////
        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        tableint addPoint(const void *data_point, labeltype label, int level);

        

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,size_t init_out,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                           int level);
        tableint mutuallyConnectNewElement_CSR(const void *data_point, tableint cur_c,size_t init_out,
                                           std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates);


        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M);
        void getNeighborsByRNGALPHA(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M, float alpha);

        void getNeighborsByANGLE(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst> &top_candidates,
                const size_t M, float angle);

        void getNeighborsByminrank( std::priority_queue<std::pair<dist_t, tableint>,
                std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                                    const size_t M, int minsize);
        void getNeighbor(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
                const size_t M) ;

          };



}
#include "io.h"
#include "search.h"
#include "diversifications.h"
#include "build.h"