

CLAGS = -Wall

all:
	gcc $(INPUT_FILEPATH) $(CLAGS) -o $(OUTPUT_FILEPATH) `pkg-config --cflags --libs OpenCL`

build_run:
	gcc $(INPUT_FILEPATH) $(CLAGS) -o $(OUTPUT_FILEPATH) `pkg-config --cflags --libs OpenCL` && $(OUTPUT_FILEPATH)

run:
	$(OUTPUT_FILEPATH)