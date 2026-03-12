# Compiler and flags
CC = gcc
CFLAGS := -Wall -Werror -std=c99 -g -Iinclude
LDFLAGS = -lm

# Directories
OBJDIR = obj
INCDIR = include

# Sources
SRCS = main.c \
       candas.c \
       numc.c \
       arena.c \
       prng.c

# Automatically grab all headers in include/
HDRS = $(wildcard $(INCDIR)/*.h)

# Target executable
TARGET = main

# Object files inside obj/
OBJS = $(SRCS:%.c=$(OBJDIR)/%.o)

.PHONY: all clean

all: $(TARGET)

# Link executable
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile source -> obj/
$(OBJDIR)/%.o: %.c $(HDRS) | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create obj directory if it doesn't exist
$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	$(RM) -r $(OBJDIR) $(TARGET)
