#!/bin/bash

pair=$2;
langdir=$3;
out=$4;

echo "$1"
echo "$2"
echo "$3"

split -d -l 1000000 $splits $1 part;
mkdir -p /content/$pair/splits;
mkdir -p /content/$pair/translated_splits;
mv $PWD/part* /content/$pair/splits/.;
cp -r /content/$pair/splits "/content/gdrive/My Drive/NLP-project/translated/$pair/splits_$pair/"
for f in /content/$pair/splits/*; do
	echo "=======================Tranlsating $f========================";
	./translate-single.sh < $f > /content/$pair/translated_splits/${f##*/};
	cp -r /content/$pair/translated_splits/${f##*/} "/content/gdrive/My Drive/NLP-project/translated/$pair/translated_splits_$langdir/"
	echo "=======================Tranlsated $f========================";
done;
cat /content/$pair/translated_splits/* > /content/$pair/$out;