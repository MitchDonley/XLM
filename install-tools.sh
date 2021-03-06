# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

lg=$1  # input language

# data path
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts

# tools path
mkdir -p $TOOLS_PATH

#
# Download and install tools
#

cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

# Download fastBPE
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi

# Compile fastBPE
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  cd ..
fi

# Download Sennrich's tools
if [ ! -d "$WMT16_SCRIPTS" ]; then
  echo "Cloning WMT16 preprocessing scripts..."
  git clone https://github.com/rsennrich/wmt16-scripts.git
fi

# Download WikiExtractor
if [ ! -d $TOOLS_PATH/wikiextractor ]; then
    echo "Cloning WikiExtractor from GitHub repository..."
    git clone https://github.com/attardi/wikiextractor.git
fi

if [ ! -d $TOOLS_PATH/stanford-segmenter-* ]; then
    curl -o stanford-segmenter-2018-10-16.zip https://nlp.stanford.edu/software/stanford-segmenter-2018-10-16.zip
    unzip stanford-segmenter-2018-10-16.zip
    rm stanford-segmenter-2018-10-16.zip
    echo "Install stanford-segmenter"
fi

if [ ! -d $TOOLS_PATH/kytea-* ]; then
    curl -o kytea-0.4.7.tar.gz http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
    tar -xzf kytea-0.4.7.tar.gz
    cd kytea-0.4.7
    ./configure
    make
    sudo make install
    cd ..
    rm kytea-0.4.7.tar.gz
fi
  

# # Chinese segmenter
# if ! ls $TOOLS_PATH/stanford-segmenter-* 1> /dev/null 2>&1; then
#   echo "Stanford segmenter not found at $TOOLS_PATH/stanford-segmenter-*"
#   echo "Please install Stanford segmenter in $TOOLS_PATH"
#   exit 1
# fi
# 
# # Thai tokenizer
# if ! python -c 'import pkgutil; exit(not pkgutil.find_loader("pythainlp"))'; then
#   echo "pythainlp package not found in python"
#   echo "Please install pythainlp (pip install pythainlp)"
#   exit 1
# fi
# 
