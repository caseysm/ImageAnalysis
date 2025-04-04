```mermaid
flowchart TD
    %% Main flow
    A[Input: ND2 Files] --> B1[Segmentation]
    A --> C1[Genotyping]
    A --> D1[Phenotyping]
    
    %% Base Pipeline
    BP["core/pipeline.py: Pipeline Base Class"] -->|extends| SB["core/segmentation/base.py: SegmentationBasePipeline"]
    BP -->|extends| GB["core/genotyping/base.py: GenotypingBasePipeline"]
    BP -->|extends| PB["core/phenotyping/base.py: PhenotypingBasePipeline"]
    
    %% Segmentation Branch
    subgraph segmentation[Segmentation]
        B1 --> SB
        SB -->|extends| S10["core/segmentation/segmentation_10x.py: Segmentation10XPipeline"]
        SB -->|extends| S40["core/segmentation/segmentation_40x.py: Segmentation40XPipeline"]
        S10 --> CS["core/segmentation/cell_segmentation.py: CellSegmentation"]
        S40 --> CS
        CS --> SN["segment_nuclei()"]
        CS --> SC["segment_cells()"]
        SN --> CL["clean_and_label()"]
        SC --> CL
        CL --> B6[Output: Cell & Nuclear Masks (.npy)]
    end
    
    %% Genotyping Branch
    subgraph genotyping[Genotyping]
        C1 --> GB
        GB -->|extends| SGP["core/genotyping/pipeline.py: StandardGenotypingPipeline"]
        SGP --> PC["core/genotyping/peak_calling.py: PeakCaller"]
        PC --> BA["core/genotyping/barcode_assignment.py: BarcodeAssigner"]
        BA --> C5[Output: Cell Genotypes (.csv)]
    end
    
    %% Connect segmentation output to genotyping
    B6 --> GB
    
    %% Phenotyping Branch
    subgraph phenotyping[Phenotyping]
        D1 --> PB
        PB -->|extends| SPP["core/phenotyping/pipeline.py: StandardPhenotypingPipeline"]
        SPP --> PM["core/phenotyping/metrics.py: calculate_metrics()"]
        SPP --> ET["process_tile()"]
        ET --> D5[Output: Cell Phenotypes (.csv)]
    end
    
    %% Connect outputs to phenotyping
    B6 --> PB
    C5 --> PB
    
    %% Mapping
    subgraph mapping[Mapping]
        MM["core/mapping/model.py: MappingModel"] -->|uses| MT["core/mapping/matching.py: PointMatcher"]
        MT --> MP["core/mapping/pipeline.py: MappingPipeline"]
        MP --> MO[Output: Transformation Matrices]
    end
    
    %% Connect mapping to phenotyping
    MO --> SPP
    
    %% Album Creation
    subgraph visualization[Visualization]
        VA["core/visualization/albums.py: AlbumCreator"] --> VC["crop_cells()"]
        VC --> VN["normalize_images()"]
        VN --> VCp["create_composite()"]
        VCp --> E5[Output: Cell Albums (.npy)]
    end
    
    %% Connect to album creation
    B6 --> VA
    D5 --> VA
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B6 fill:#bbf,stroke:#333,stroke-width:2px
    style C5 fill:#bbf,stroke:#333,stroke-width:2px
    style D5 fill:#bbf,stroke:#333,stroke-width:2px
    style E5 fill:#bbf,stroke:#333,stroke-width:2px
    style MO fill:#bbf,stroke:#333,stroke-width:2px
    style BP fill:#ffd,stroke:#333,stroke-width:3px
```