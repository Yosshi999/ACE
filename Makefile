_slic.so := ace/slic/_slic.*.so
.PHONY: all proto slic


all: proto slic

proto:
	$(MAKE) -C ace

slic: $(_slic.so)

$(_slic.so): setup.py ace/slic/_slic.pyx
	python setup.py build_ext --inplace
