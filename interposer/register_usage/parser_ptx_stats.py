import re
import csv

filename = "./modified_stats.txt"  # Replace with the actual path to your file

with open(filename, "r") as file:
    content = file.read()

matches = re.findall(r"ptxas info\s+:\s+Function properties for\s+(_Z\S+)\n\s+0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info\s+:\s+Used (\d+) registers, (\d+) bytes smem, (\d+) bytes cmem\[0\]", content)

with open("output.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Function", "Used Registers", "Shared Memory", "Constant Memory"])
    writer.writerows(matches)

