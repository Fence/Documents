from ctypes import *
lib = CDLL('./hsp.so')
argv = ['hsp', 'problem.pddl', 'domain.pddl']
lib.main(len(argv), (c_char_p*len(argv))(*argv))

