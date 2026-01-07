#!/usr/bin/make -f

SRCS := $(wildcard *.tex)
PDFS := $(SRCS:%.tex=%.pdf)
SVGS := $(SRCS:%.tex=%.svg)

AUXS := $(SRCS:%.tex=%.aux)
TEXLOGS := $(SRCS:%.tex=%.log)

all: ${PDFS} ${SVGS} logo.png

%.pdf: %.tex
	pdflatex "$<"

%.svg: %.pdf
	inkscape --pdf-poppler "$<" -o "$@"

%.png: %.pdf
	inkscape --pdf-poppler "$<" -b FFFFFF --export-dpi=150 --export-png="$@"

texauxclean:
	rm -f ${AUXS}
	rm -f ${TEXLOGS}

clean: texauxclean
	rm -f ${PDFS}
	rm -f ${SVGS}
