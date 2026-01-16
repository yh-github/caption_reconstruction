all: workshop_short.pdf

workshop_short.pdf: workshop_short.tex
	lualatex -interaction=nonstopmode workshop_short.tex

clean:
	rm -f *.aux *.log *.out *.pdf
