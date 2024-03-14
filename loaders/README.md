Note: I did not upload the sample images here,
but this works assuming the sample images are 
in Images/Sample Tiffs and are of the format NAIP_[fid].tif

If you would like to use jpeg images, simply change the part of
the code that says ".tif" for ."jpeg" and everything should work fine.

Also note that at the moment I am letting Pytorch infer data types in my DataSet class. Specifying data types in the future (assuming they are known at compile time) may reduce inference overhead, which could lead to a significant improvement given the dataset's size. We could also increase the number of workers in the DataLoader function to parallelize the data loading process.
