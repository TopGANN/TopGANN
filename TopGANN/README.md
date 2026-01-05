# TopGANN

TopGANN is an experimental framework for adaptive graph based approximate nearest neighbor search.  
It is built on top of the HNSWLIB codebase and extends it with step wise adaptive mechanisms for graph construction and pruning.

---

## Core concepts

TopGANN adds adaptive controls during incremental graph construction.

SBW     Step beam width. Controls exploration cost.  
ABW     Augmented beam width. Keeps best pruned candidates.  
SOUT    Step out degree. Controls degree growth.  
SALPHA  Step alpha. Controls pruning relaxation.  
HL      Hierarchical layers flag.

HL = 1 enables hierarchical layers similar to HNSW.  
HL = 0 builds only the base layer graph.

---

## Requirements

Linux  
GCC 8 or newer  
CMake 3.10 or newer  
Large memory machine recommended for large datasets

---

## Build

    cd projects/AGRENN
    mkdir -p build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j

Binary produced:

    projects/AGRENN/Release/TopGANN

---

## Dataset format

Binary float vectors.

- .bin format
- float32 values
- row major order
- no header

Layout:

    data.bin    = N * D float32 values
    queries.bin = Q * D float32 values

---

## Running experiments

Indexing:

    ./TopGANN \
      --dataset data.bin \
      --dn 1000000 \
      --dim 96 \
      --index-path INDEX_DIR \
      --K 36 \
      --L 100 \
      --TPN 5 \
      --VPN 1.0 \
      --SBW 1 \
      --ABW 0 \
      --SOUT 2 \
      --SALPHA 2 \
      --hl 0 \
      --mode 0

Search:

    ./TopGANN \
      --dataset data.bin \
      --queries queries.bin \
      --qn 100 \
      --index-path INDEX_DIR \
      --dim 96 \
      --K 10 \
      --L-mults 10,30,60,90 \
      --hl 0 \
      --mode 1

---

## Parameter sweeps

The bash script sweeps over:

- SOUT in 2 3 4
- SALPHA in 1 10
- VPN in 1.0 1.2 2.0
- SBW in 1 4 6
- ABW in 0 or L_BUILD
- HL in 0 or 1

Each configuration builds an index and runs search.

---

## Output structure

    INDEX_TESTS/
      AIGANN_dataset_params/

    Tests/
      KSvsHL/
        AIGANN_dataset_params.log

Logs contain indexing time, memory usage, recall, and query latency.

---

## Reproducibility

Experiments are deterministic for fixed insertion order and seeds.  
All parameters are logged.

---

## License

Research use only.

---

## Citation

    @misc{topgann,
      title  = {TopGANN: Adaptive Graph Construction for Vector Search},
      author = {Azizi, Ilias and Echihabi, Karima and Christophides, Vassilis and Palpanas, Themis},
      year   = {2025},
      note   = {Built on top of HNSWLIB}
    }
