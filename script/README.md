# Running trefide on a TIFF file

Work in progress. Current status: trying to understand how to reconstruct the signal from the decomposed matrices.

1. Put a tiff file in `input/` (I used `c163_s19_00002.tiff`).
2. Run the Docker container with `sudo docker run -v ~/git/trefide/script:/root/trefide/script -it -p 34000:34000 paninski/trefide:1.2`
3. Open the Jupyter notebook server served by the Docker container.
4. Execute the `trefide_test.ipynb` notebook.
