# Data Folder
- collected_data: This folder contains the raw data collected directly from Mechnical Turker, all the text are lower-cased, containing foreign characters under some tables. There are two files, the r1 file is collected in the first easy round, which contains sentences involving less reasoning. The r2 file contains the sentences involving more complex reasoning. The two files in total contains roughly 110K statements. 
  ```
  Table-id: {
  [
  Statement 1,
  Statement 2,
  ...
  ],
  [
  Label 1,
  Label 2,
  ...
  ],
  Table Caption
  }
  ```
- tokenized_data: This folder contains the data after tokenization, the data is obtained using the script preprocess_data.py, you can simply reproduce it by:
  ```
  python preprocess_data.py
  ```
  this script is mainly used to perform feature-based entity linking, the entities in the statements are linked to the cell values in the table, the obtained file is tokenized_data/full_cleaned.json, the data format looks like:
  ```
  Table-id: {
  [
  Statement 1: xxxxx #xxx;idx1,idx2# xxx.
  Statement 2: xx xxx #xxx;idx1,idx2# xxx. 
  ...
  ],
  [
  Label 1,
  Label 2,
  ...
  ],
  Table Caption
  }
  ```
  here the enclosed snippet #xxx;idx1,idx2# denotes that the work "xxx" is linked to idx1-th row and idx2-th column, if idx1=-1, it means that it links to the caption. The entity linking step is essential for performing program search algorithm to connect these entities with known functions for semantic representation.
  
