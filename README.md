Hello, thanks for visiting:smiley::heartpulse:.  
Mirage aims to learn SPRITE multi-way interaction features based on Hi-C data to generate new multi-way interaction matrices.  
Mirage is divided into three parts: sampling, training, splicing.  
The resolution is 10 kb.  
Since the application set performs sliding window sampling of all single chromosomes, two sampling and splicing schemes are used in the training set and the application set.  
**Sampling**  
Load 10 kb matrices of Hi-C and SPRITE in *df_hic*, *df_sprite*.  
In the application set, *chrlen* provides information on the sample number of 23 chromosomes of GM12878 by Rao et al. If you have your own dataset, you can change this parameter.  
Here, chromosome number *X* is set to *23*.  
**training**  
Get *epoch_300_conv1d_ARMA_10kb_out_128.pkl* file.  
**Splicing**  
The *cluster* parameter is set to the maximum suffix *k* of the *x_sample_10kb.pt* file for each chromosome sampled from the test set.  
When Splicing is done, a new large matrix of individual chromosomes can be obtained.  

Finally, use the *.bedgraph* file to calculate mirage loops.  
Thank you for your attention:snail::mushroom:.
