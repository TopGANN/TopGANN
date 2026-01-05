#pragma once

#include "hnswalg.h"

namespace hnswlib {

    template<typename dist_t>
    void HierarchicalNSW<dist_t>::saveIndexCompact(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);
            writeBinaryPOD(output, dim_);

            double noutdeg=0;
            unsigned min = 1000;
            unsigned max = 0;
            
            #ifdef CSR_BUILD
            
            for (size_t i = 0; i < cur_element_count; ++i) {
                const auto& nbrs = final_graph_[i];
                const size_t size = nbrs.size();

                if (size > maxM0_) throw std::runtime_error("Outdegree exceeds maxM0_");
                if (size > std::numeric_limits<uint16_t>::max()) throw std::runtime_error("Outdegree exceeds uint16_t");

                uint16_t size16 = static_cast<uint16_t>(size);
                output.write(reinterpret_cast<const char*>(&size16), sizeof(uint16_t));

                output.write(reinterpret_cast<const char*>(nbrs.data()), sizeof(tableint) * size);

                max = std::max(max, (unsigned)size);
                min = std::min(min, (unsigned)size);
                noutdeg += size;
            }

            #else

            for (size_t i = 0; i < cur_element_count; ++i)
            {
                int* data = (int*)get_linklist0(i);

                unsigned short int size = getListCount((linklistsizeint*)data);

                output.write((char*)(&size), sizeof(unsigned short int));
                output.write((char*)(data+1), sizeof(unsigned)*size);

                max = max < size? size:max;
                min = min < size? min:size;
                noutdeg+=size;

            }
            #endif
            //we will ensure that labels are written in the same order as the elements order during insertion
            //output.write((char*)&labels[0], sizeof(unsigned)*cur_element_count);

            noutdeg=noutdeg/cur_element_count;

            std::cout << "Graph AVG outdegree : "<<noutdeg << ", min : "<<min << ", max : "<<max<<std::endl;
            std::cout << "Graph AVG outdegree : "<<noutdeg << ", min : "<<min << ", max : "<<max<<std::endl;
            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }
            output.close();
        }
    
    template<typename dist_t>
    void HierarchicalNSW<dist_t>::loadIndexCompact(const std::string &location, SpaceInterface<dist_t> *s) {

            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file:"+location);

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);

            readBinaryPOD(input, dim_);

            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();


            final_graph_.clear();
            final_graph_.resize(cur_element_count);

            for (size_t i = 0; i < cur_element_count; ++i) {
                uint16_t outdegree16;
                readBinaryPOD(input, outdegree16);

                const size_t outdegree = static_cast<size_t>(outdegree16);
                if (outdegree > maxM0_) {
                    throw std::runtime_error(
                        "Outdegree exceeds maxM0_: " + std::to_string(outdegree) +
                        " > " + std::to_string(maxM0_));
                }

                auto &nbrs = final_graph_[i];
                nbrs.resize(outdegree);

                input.read(reinterpret_cast<char*>(nbrs.data()), sizeof(tableint) * outdegree);
                if (!input) throw std::runtime_error("Failed to read neighbor list");
            }

            data_ = nullptr;

            data_level0_memory_ =  nullptr;



            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

            visited_list_pool_ = new VisitedListPool(1, cur_element_count);

            linkLists_ = (char **) malloc(sizeof(void *) * cur_element_count);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(cur_element_count);
            revSize_ = 1.0 / mult_;

            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[i]=i; // label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0) {
                    element_levels_[i] = 0;

                    linkLists_[i] = nullptr;
                } else {
                    element_levels_[i] = linkListSize / size_links_per_element_;
                    linkLists_[i] = (char *) malloc(linkListSize);
                    if (linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(linkLists_[i], linkListSize);
                }
            }

            has_deletions_=false;
          
            input.close();

            return;
        }

    template<typename dist_t>
    void HierarchicalNSW<dist_t>::loadData(const char *filename){
            data_ = nullptr;
            data_ = static_cast<float *>(malloc(sizeof(float) * max_elements_ * dim_));

            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                throw std::runtime_error("Cannot open DATA file");
            }
            in.read((char *) data_, max_elements_ * dim_ * sizeof(float));
            in.close();
        };
        

}