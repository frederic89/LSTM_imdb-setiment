需要修改的路径有imdb_preprocess.py中的dataset_path，以及视情况下修改tokenizer.perl中的my $mydir

tokenizer.perl 来自 Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
nonbreaking_prefixes来自Moses: https://github.com/moses-smt/mosesdecoder
均针对欧洲语言
