RM			= \rm -f

ECHO			= /bin/echo -e

CC			= gcc

BIN_DIR			= ./bouding_boxes/bin

NAME			= $(BIN_DIR)/libmotion_detector_optimization.so

SRC_DIR			= src

SRCS			= $(SRC_DIR)/scanner_opti.c \
			$(SRC_DIR)/coord_list.c

OBJS			= $(SRCS:.c=.o)

CFLAGS			= -W -Wall -Werror -Wextra -fPIC -O3 \
			`pkg-config --cflags python` \
			-I$(SRC_DIR) \
			-I`python -m site --user-site`/numpy/core/include

LIBS			= `pkg-config --libs python`

all:            	$(NAME)

$(NAME):		$(OBJS)
			@mkdir -p $(BIN_DIR)
			$(CC) $(OBJS) -o $(NAME) -shared $(LIBS)

clean:
			$(RM) $(OBJS)

fclean:         	clean
			$(RM) -r $(BIN_DIR)

re:             	fclean all

.c.o:			%.c
			$(CC) -c $< -o $@ $(CFLAGS)

.PHONY:         	all clean fclean re