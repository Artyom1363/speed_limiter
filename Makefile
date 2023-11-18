CFLAGS=-c -Wall -O2
LIBS = -lm -lpthread

all: liblcd2004.a

liblcd2004.a: lcd2004.o
	ar -rc liblcd2004.a lcd2004.o ;\
	sudo cp liblcd2004.a /usr/local/lib ;\
	sudo cp lcd2004.h /usr/local/include

lcd2004.o: lcd2004.c
	$(CC) $(CFLAGS) lcd2004.c

clean:
	rm *.o liblcd2004.a
