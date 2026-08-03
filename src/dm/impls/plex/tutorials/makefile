# This makefile follows PETSc conventions for tutorial examples.
# To build: make ex21
# To run tests: make test
#
# Requires PETSC_DIR and PETSC_ARCH to be set.

CFLAGS   =
FFLAGS   =
CPPFLAGS =
FPPFLAGS =
LOCDIR   = src/dm/impls/plex/tutorials/
EXAMPLESC = ex21.c

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

ex21: ex21.o
	-${CLINKER} -o ex21 ex21.o ${PETSC_LIB}
	${RM} ex21.o
