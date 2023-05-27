"""
Miscellaneous functions that do not belong in any particular folder.

Created August 17, 2022
Last edited August 19, 2022
"""

def getArgsBase(argv, argOrder = [], boolKeys = [], intKeys = [], floatKeys = []):
	"""Get arguments from the command line and return them as a dictionary.

	parameters:
	argv: the arguments from argv
	argOrder: the names of each argument, in the order that the correspond to args
	boolKeys: the names of the arguments that should be booleans, unordered
	intKeys: "" for arguments that should be ints
	floatKeys: "" for arguments that should be floats
	"""
	if(len(argv) != len(argOrder) + 1):
		raise Exception("length of argv "
			"({}) and length of argOrder ({}) + 1 not equal.".format(
				len(argv), len(argOrder)))

	args = {}

	#load in the arguments by their order		
	for i in range(len(argOrder)):
		args[argOrder[i]] = argv[i + 1]
	
	#adjust the format of various values	
	#values that need to be bools
	for boolKey in boolKeys:
		args[boolKey] = bool(int(args[boolKey]))
	
	#values that need to be ints
	for intKey in intKeys:
		args[intKey] = int(args[intKey])

	#values that need to be floats
	for floatKey in floatKeys:
		args[floatKey] = float(args[floatKey])

	#print out the arguments
	print("called with arguments:", flush = True)
	for key in args:
		print("\t{}: {}".format(key, args[key]), flush = True)
	
	return args