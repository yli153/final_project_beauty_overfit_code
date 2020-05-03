# final-project-beauty-overfit

##Group Name: Beauty Overfit
##Group Member: lyang39, xjiang5, ycui10, yli153



Arguments:
-al, --algorithm, 'original_nst', or 'portrait'. Default algorithm is original neural style transfer

-m, --mode, 'original', 'color_trans', or 'artist_style'. Default mode is original: the original NST doesn't perverse color infomration of the content image. 

-c, --color, 'original', 'cholesky', or 'image_analogies'. Default mode is original. However, if you choose 'color_trans' or 'artist_style' as your mode, you have to use either 'cholesky', or 'image_analogies' as color. Failed to do so will throw assertion error.

-a, --artist, 'artist1', 'artist2', or 'artist3'. Default mode is artist1. 

Examples of Running with Arguments:

python main.py

python main.py -al portrait

python main.py -al portrait -m artist_style -c cholesky -a artist2

