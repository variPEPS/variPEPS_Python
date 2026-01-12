#!/usr/bin/make -f

PDFS := $(wildcard *.pdf)
SVGS := $(PDFS:%.pdf=%.svg)

all: ${SVGS}

%.svg: %.pdf
	inkscape --pdf-poppler "$<" -o "$@"

clean: texauxclean
	rm -f ${SVGS}
