CC = cc
CFLAGS = -O3
LDFLAGS = -lm -lpthread

.PHONY: all clean

all: expectimax_c.so ntuple_c.so ntuple_train expectimax_calibrate

# Expectimax engine (shared library for server)
expectimax_c.so: expectimax_c.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

# N-Tuple player (shared library for server)
ntuple_c.so: ntuple_c.c
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

# N-Tuple trainer (standalone binary)
ntuple_train: ntuple_train.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Expectimax heuristic calibrator (standalone binary)
expectimax_calibrate: expectimax_calibrate.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f *.so *.o ntuple_train expectimax_calibrate
