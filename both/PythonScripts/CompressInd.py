from IndDataCompression import compress
from sys import argv

inTXT = argv[1]
outNPZ = argv[2]

compress(inTXT, outNPZ)