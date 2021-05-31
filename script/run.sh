eval "$(/home/cyrille/miniconda3/bin/conda shell.bash hook)"
source activate pmd
python3 pmd_tiff_example.py c163_s19_00002.tiff input output
