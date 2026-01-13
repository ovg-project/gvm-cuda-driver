import argparse
import lief
import json

def parse_args():
	parser = argparse.ArgumentParser(description="Symbol mapping generator")
	parser.add_argument("-in", "--input", required=True, help="Path to the input object")
	parser.add_argument("-map", "--mapping", required=True, help="Path to the input mapping")
	parser.add_argument("-t", "--type", required=True, choices=["cuda", "wrapper"], help="Type of input object")
	parser.add_argument("-out", "--output", required=True, help="Path to the output object")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	mapping = None
	with open(args.mapping, "r") as f:
		mapping = json.load(f)[args.type]

	bin = lief.parse(args.input)
	for sym in bin.symbols:
		if sym.name in mapping.keys():
			sym.name = mapping[sym.name]

	for sym in bin.dynamic_symbols:
		if sym.name in mapping.keys():
			sym.name = mapping[sym.name]

	bin.write(args.output)
