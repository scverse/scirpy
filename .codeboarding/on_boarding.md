```mermaid

graph LR

    I_O_Data_Management["I/O & Data Management"]

    Preprocessing["Preprocessing"]

    Immune_Receptor_Distance_Calculation["Immune Receptor Distance Calculation"]

    Repertoire_Analysis_Tools["Repertoire Analysis Tools"]

    Visualization["Visualization"]

    Core_Utilities["Core Utilities"]

    I_O_Data_Management -- "Provides Data To" --> Preprocessing

    I_O_Data_Management -- "Provides Data To" --> Immune_Receptor_Distance_Calculation

    I_O_Data_Management -- "Provides Data To" --> Repertoire_Analysis_Tools

    I_O_Data_Management -- "Provides Data To" --> Visualization

    Preprocessing -- "Receives Data From" --> I_O_Data_Management

    Immune_Receptor_Distance_Calculation -- "Receives Data From" --> I_O_Data_Management

    Repertoire_Analysis_Tools -- "Receives Data From" --> I_O_Data_Management

    I_O_Data_Management -- "Utilizes" --> Core_Utilities

    Preprocessing -- "Receives Data From" --> I_O_Data_Management

    Preprocessing -- "Provides Data To" --> I_O_Data_Management

    Preprocessing -- "Provides Data To" --> Immune_Receptor_Distance_Calculation

    Preprocessing -- "Provides Data To" --> Repertoire_Analysis_Tools

    Preprocessing -- "Utilizes" --> Core_Utilities

    Immune_Receptor_Distance_Calculation -- "Receives Data From" --> I_O_Data_Management

    Immune_Receptor_Distance_Calculation -- "Receives Data From" --> Preprocessing

    Immune_Receptor_Distance_Calculation -- "Provides Data To" --> I_O_Data_Management

    Immune_Receptor_Distance_Calculation -- "Provides Data To" --> Repertoire_Analysis_Tools

    Immune_Receptor_Distance_Calculation -- "Utilizes" --> Core_Utilities

    Repertoire_Analysis_Tools -- "Receives Data From" --> I_O_Data_Management

    Repertoire_Analysis_Tools -- "Receives Data From" --> Preprocessing

    Repertoire_Analysis_Tools -- "Receives Data From" --> Immune_Receptor_Distance_Calculation

    Repertoire_Analysis_Tools -- "Provides Data To" --> I_O_Data_Management

    Repertoire_Analysis_Tools -- "Provides Data To" --> Visualization

    Repertoire_Analysis_Tools -- "Utilizes" --> Core_Utilities

    Visualization -- "Receives Data From" --> I_O_Data_Management

    Visualization -- "Receives Data From" --> Repertoire_Analysis_Tools

    Visualization -- "Utilizes" --> Core_Utilities

    Core_Utilities -- "Supports" --> I_O_Data_Management

    Core_Utilities -- "Supports" --> Preprocessing

    Core_Utilities -- "Supports" --> Immune_Receptor_Distance_Calculation

    Core_Utilities -- "Supports" --> Repertoire_Analysis_Tools

    Core_Utilities -- "Supports" --> Visualization

    click I_O_Data_Management href "https://github.com/scverse/scirpy/blob/main/.codeboarding//I_O_Data_Management.md" "Details"

    click Preprocessing href "https://github.com/scverse/scirpy/blob/main/.codeboarding//Preprocessing.md" "Details"

    click Immune_Receptor_Distance_Calculation href "https://github.com/scverse/scirpy/blob/main/.codeboarding//Immune_Receptor_Distance_Calculation.md" "Details"

    click Repertoire_Analysis_Tools href "https://github.com/scverse/scirpy/blob/main/.codeboarding//Repertoire_Analysis_Tools.md" "Details"

    click Visualization href "https://github.com/scverse/scirpy/blob/main/.codeboarding//Visualization.md" "Details"

    click Core_Utilities href "https://github.com/scverse/scirpy/blob/main/.codeboarding//Core_Utilities.md" "Details"

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



The `scirpy` library, designed as a Specialized Scientific Computing Library/Bioinformatics Toolkit, exhibits a modular and data-centric architecture, with the AnnData object serving as the central data container. The system is structured around six core components, each with distinct responsibilities and clear interaction patterns, facilitating a robust and extensible framework for immune repertoire analysis.



### I/O & Data Management [[Expand]](./I_O_Data_Management.md)

This foundational component is responsible for the entire lifecycle of immune receptor data. It handles the ingestion of raw data from diverse formats (e.g., 10x VDJ, AIRR) into the central AnnData object, manages schema compatibility, and provides a consistent, validated API for accessing and manipulating data within AnnData. It acts as the primary interface for data persistence and retrieval.





**Related Classes/Methods**:



- `scirpy.datasets` (0:1)

- `scirpy.io` (0:1)

- `scirpy.util.DataHandler` (0:1)

- `scirpy.get` (0:1)





### Preprocessing [[Expand]](./Preprocessing.md)

This component prepares raw or loaded immune receptor data for downstream analysis. Its key functions include indexing chains within the AnnData object to ensure proper structure and merging multiple AnnData objects for combined analysis, ensuring data readiness and consistency.





**Related Classes/Methods**:



- `scirpy.pp` (0:1)





### Immune Receptor Distance Calculation [[Expand]](./Immune_Receptor_Distance_Calculation.md)

This specialized component implements various algorithms for calculating sequence-based distances between immune receptor sequences (e.g., CDR3s). It computes distance matrices and identifies clonotype neighbors based on these metrics, often leveraging performance optimizations like parallelization.





**Related Classes/Methods**:



- `scirpy.ir_dist` (0:1)





### Repertoire Analysis Tools [[Expand]](./Repertoire_Analysis_Tools.md)

This is the core analytical engine of the library, offering a comprehensive suite of tools for in-depth analysis of immune repertoires. Functions include defining clonotypes, analyzing clonal expansion, assessing diversity, quantifying repertoire overlap, and performing quality control on chains.





**Related Classes/Methods**:



- `scirpy.tl` (0:1)





### Visualization [[Expand]](./Visualization.md)

This component is dedicated to generating various plots that visualize repertoire characteristics and the results of analyses. It includes base plotting utilities and styling functions to ensure consistent and high-quality visualizations, aiding in exploratory data analysis and presentation.





**Related Classes/Methods**:



- `scirpy.pl` (0:1)





### Core Utilities [[Expand]](./Core_Utilities.md)

This foundational component provides common helper functions used across the entire library. This includes utilities for documentation injection, data type checks, sequence translation, parallelization, various mathematical helpers, and functionalities for graph creation, manipulation, and layout algorithms, particularly for clonotype networks. It acts as a support layer for all other components.





**Related Classes/Methods**:



- `scirpy.util` (0:1)









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)