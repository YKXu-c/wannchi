include ../make.sys

objects = constants.o linalgwrap.o symmetry.o para.o wanndata.o simp.o lattice.o intRPA.o pade.o \
          bayesian.o classical_mc.o ahc_calc.o

mod.a: $(objects)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

# New module dependencies
bayesian.o:     constants.o linalgwrap.o
classical_mc.o: constants.o
ahc_calc.o:     constants.o wanndata.o linalgwrap.o

clean:
	rm -rf *.o *.mod *.a *.i90
