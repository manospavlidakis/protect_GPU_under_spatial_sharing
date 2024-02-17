#! /usr/bin/env python3
import csv

file1 = "regUsageModified.csv"  # Replace with the actual path to your first file
file2 = "regUsageNative.csv"  # Replace with the actual path to your second file
output_file = "regUsage_merged.csv"  # Replace with the desired path for the output file

# Read the functions from the first file and store them in a dictionary
functions_dict = {}

with open(file1, "r") as file1_csv:
    reader1 = csv.DictReader(file1_csv)
    for row in reader1:
        function = row["Function"]
        functions_dict[function] = row

# Open the second file and append matching lines to the output file
with open(file2, "r") as file2_csv, open(output_file, "w", newline="") as output_csv:
    reader2 = csv.DictReader(file2_csv)
    fieldnames = reader2.fieldnames

    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
    writer.writeheader()
    cluster_reg = {}
    cluster_const = {}
    for row in reader2:
        function = row["Function"]
        if function in functions_dict:
             if functions_dict[function]["Function"] != row["Function"]:
                        print("Not matching functions!!")
                        break
             diff = int(functions_dict[function]["Used Registers"])-int(row["Used Registers"])
             diff_const = int(functions_dict[function]["Constant Memory"])-int(row["Constant Memory"])
             if diff > -4 and diff < 20:
                     if diff not in cluster_reg: 
                             cluster_reg[diff] = 1
                     else:
                             cluster_reg[diff] +=1
             if diff_const not in cluster_reg:
                     cluster_const[diff_const] = 1
             else:
                     cluster_const[diff_const] +=1

             print(functions_dict[function]["Function"]+","+functions_dict[function]["Used Registers"]+","+row["Used Registers"]+","+str(diff)+","+functions_dict[function]["Constant Memory"]+","+row["Constant Memory"]+"\n")
for key, value in cluster_reg.items():
      print(key, value)
print("--- CONST  ---")
for key1, value1 in cluster_const.items():
      print(key1, value1)

print(cluster_reg)
print(cluster_const)
#            print(row)
#            writer.writerow(functions_dict[function]+","+str(row))
#            writer.writerow(row)

