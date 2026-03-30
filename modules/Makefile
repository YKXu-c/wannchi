include ../make.sys

objects = constants.o linalgwrap.o symmetry.o para.o wanndata.o simp.o lattice.o intRPA.o pade.o \
          gp_bo.o cma_es.o classical_mc.o transp_calc.o wannlog.o

mod.a: $(objects)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@

# New module dependencies
gp_bo.o:        constants.o linalgwrap.o
cma_es.o:       constants.o
classical_mc.o: constants.o
transp_calc.o:  constants.o wanndata.o linalgwrap.o
wannlog.o:      constants.o

clean:
	rm -rf *.o *.mod *.a *.i90
