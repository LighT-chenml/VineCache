cd ./FIFO_Cython
python3 setup_FIFO_Cython.py build_ext --inplace
cd ..

cd ./LRU_Cython
python3 setup_LRU_Cython.py build_ext --inplace
cd ..

cd ./EvLFU_Cython
python3 setup_EvLFU_Cython.py build_ext --inplace
cd ..

cd ./VineCache_Cython
python3 setup_VineCache_Cython.py build_ext --inplace
cd ..

cd ./VineCache_MFU_Cython
python3 setup_VineCache_MFU_Cython.py build_ext --inplace
cd ..
