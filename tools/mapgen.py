import argparse
import lief
import json

def parse_args():
	parser = argparse.ArgumentParser(description="Symbol mapping generator")
	parser.add_argument("-obj", "--object", required=True, help="Path to the input object")
	parser.add_argument("-map", "--mapping", required=True, help="Path to the output mapping")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()

	cudamap = {}
	wrappermap = {}
	bin = lief.parse(args.object)
	for sym in bin.symbols:
		if sym.name.endswith("_WRAPPER"):
			cudamap[sym.name.removesuffix("_WRAPPER")] = sym.name.removesuffix("_WRAPPER") + "_IMPL"
			wrappermap[sym.name] = sym.name.removesuffix("_WRAPPER")

	for sym in bin.symbols:
		if sym.name.endswith("_IMPL"):
			if sym.name.removesuffix("_IMPL") not in cudamap.keys():
				wrappermap[sym.name] = sym.name.removesuffix("_IMPL")

	with open(args.mapping, "w") as f:
		json.dump({ "cuda" : cudamap, "wrapper" : wrappermap }, f, indent=4)
