# Interface
Interface for AMT


1. The READY/ folder stores all the data, the needed files are preprocess_?.json
2. Please go to code directory, the only task is to execute run.py
3. If there are two files in the name of preprocess_?.json, then you should run the following commands on different machines.
```
python run.py --part 0 --synthesize
python run.py --part 1 --synthesize
```
4. The results are stored in data/all_programs.
5. If you want to debug, just switch to the sequential execution paradigm.
```
python run.py --part 0 --synthesize --sequential

```
