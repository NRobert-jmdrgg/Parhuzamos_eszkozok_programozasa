

CLAGS = -Wall

all:
	gcc $(NAME).c $(CLAGS) -o ./bin/$(NAME) `pkg-config --cflags --libs OpenCL`

build_run:
	gcc $(NAME).c $(CLAGS) -o ./bin/$(NAME) `pkg-config --cflags --libs OpenCL` && ./bin/$(NAME)

clean: 
	$(RM) ./bin/$(NAME)