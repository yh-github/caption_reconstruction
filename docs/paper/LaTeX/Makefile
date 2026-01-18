# Makefile for ICCV 2025 paper

all: workshop_short.pdf

workshop_short.pdf: workshop_short.tex main.bib
	pdflatex workshop_short.tex
	bibtex workshop_short
	pdflatex workshop_short.tex
	pdflatex workshop_short.tex

clean:
	rm -f *.aux *.log *.out *.pdf *.bbl *.blg *.brf
