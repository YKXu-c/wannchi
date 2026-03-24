include ../make.sys

objects = input.o compute_chi.o green.o

LIBS = ../modules/mod.a

all: wannchi wanneff_js

wannchi: wannchi.f90 $(LIBS) $(objects)
	$(F90) $(F90FLAGS) -fpp -o wannchi.x wannchi.f90 $(objects) $(LIBS) -I$(MKLINCLUDE) -I../modules -I. $(LAPACK95LIBS) $(MKLLIBS)

wanneff_js: wanneff_JS.o green.o $(LIBS)
	$(F90) $(F90FLAGS) -fpp -o wanneff_js.x wanneff_JS.o green.o $(LIBS) -I$(MKLINCLUDE) -I../modules -I. $(LAPACK95LIBS) $(MKLLIBS)

wanneff_JS.o: wanneff_JS.f90
	$(F90) $(F90FLAGS) -fpp -c $(INCLUDES) wanneff_JS.f90

$(LIBS):
	cd ../modules
	make mod.a
	cd ..

clean:
	rm -rf *.o *.mod *.x *.i90
