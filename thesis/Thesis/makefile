all:
	pdflatex thesis.tex

bib:
	pdflatex thesis.tex
	bibtex thesis
	pdflatex thesis.tex
	pdflatex thesis.tex

clean:
	rm -f *.aux *.log

distclean:clean
	rm -f *.lof *.out *.toc *.blg *.bbl *.lot
