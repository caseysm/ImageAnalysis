```mermaid
flowchart TD
    %% Main flow
    A[Input: ND2 Files] --> B1[Segmentation]
    A --> C1[Genotyping]
    A --> D1[Phenotyping]
    
    %% Segmentation Branch
    subgraph segmentation[Segmentation]
        B1 --> B2["Segment_10X.py/Segment_40X.py"]
        B2 --> B3["In_Situ_Functions.py: segment_nuclei()"]
        B3 --> B4["In_Situ_Functions.py: segment_cells()"]
        B4 --> B5["In_Situ_Functions.py: clean_and_label()"]
        B5 --> B6[Output: Cell & Nuclear Masks (.npy)]
    end
    
    %% Genotyping Branch
    subgraph genotyping[Genotyping]
        C1 --> C2["Genotyping_Pipeline.py"]
        C2 --> C3["In_Situ_Feldman.py: call_peaks()"]
        C3 --> C4["Assign cell barcodes"]
        C4 --> C5[Output: Cell Genotypes (.csv)]
    end
    
    %% Connect segmentation output to genotyping
    B6 --> C2
    
    %% Phenotyping Branch
    subgraph phenotyping[Phenotyping]
        D1 --> D2["Phenotype_Cells.py"]
        D2 --> D3["Mapping_Functions.py: transform_coordinates()"]
        D3 --> D4["Calculate phenotypic measurements"]
        D4 --> D5[Output: Cell Phenotypes (.csv)]
    end
    
    %% Connect outputs to phenotyping
    B6 --> D2
    C5 --> D2
    
    %% Mapping
    subgraph mapping[Mapping]
        M1["Mapping_1_Cal_M_Matrix.ipynb"] --> M2[Calculate transformation matrices]
        M2 --> M3["Mapping_2_Find_Fiducials.ipynb"]
        M3 --> M4["Mapping_3_Optimize_Mapping_DOF.ipynb"]
        M4 --> M5[Output: M_10X.npy, M_40X.npy]
    end
    
    %% Connect mapping to phenotyping
    M5 --> D3
    
    %% Album Creation
    subgraph albums[Album Creation]
        E1["Make_Albums.ipynb"] --> E2["Album_Functions.py: create_cell_crops()"]
        E2 --> E3["Album_Functions.py: normalize_images()"]
        E3 --> E4["Album_Functions.py: create_composite()"]
        E4 --> E5[Output: Cell Albums (.npy)]
    end
    
    %% Connect to album creation
    B6 --> E1
    D5 --> E1
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B6 fill:#bbf,stroke:#333,stroke-width:2px
    style C5 fill:#bbf,stroke:#333,stroke-width:2px
    style D5 fill:#bbf,stroke:#333,stroke-width:2px
    style E5 fill:#bbf,stroke:#333,stroke-width:2px
    style M5 fill:#bbf,stroke:#333,stroke-width:2px
```