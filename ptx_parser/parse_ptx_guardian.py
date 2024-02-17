#! /usr/bin/env python3  
import re
import sys
import time
instructions_not_supported = []
kernel_param_map = {}
g_index = 0 
def search_ptr_arthm(string):
	char = ['+', '-', '+-', '']
	for c in char:
		index = string.find(c)
		# Found
		if (index != -1):
			global g_index
			g_index = int(index)
			return c
		# Not found
		else:
			continue

def adjust_func(ptx_file, out_ptx):
	print("\n############ ADJUST FUNC ###############\n")
	#print("MAP:"+str(kernel_param_map))
	with open(ptx_file, 'r') as f:
		ptx = f.readlines()
	prev_line = ''
	lines_array = []
	param_line = ''
	for i, line in enumerate(ptx):
		if 'call.uni' in line:
			# Append desired string to previous line
			#ptx[i-2] = prev_line.rstrip() + ' desired_string\n'
			ptx[i-1] = prev_line.rstrip() + '.param .b64 param_mask;\nst.param.f64 [param_mask], %maskreg1;\n'
			#print("Prev line " +str(ptx[i-1]))
			out_ptx.write(ptx[i-1])
			next_line_num = i + 1
			#print("Next line "+ str(next_line_num))
			if next_line_num < len(ptx):
				next_line = ptx[next_line_num]
				#print("Next line "+ str(next_line))
				next_line_no_comma = next_line.replace(',', '').replace(' ','').replace('\n','')
				if next_line_no_comma.strip() == "vprintf":
					print ("vprintf remove it!!!!!!!!!!!!!!")
					out_ptx.write(line)
					continue
				#print("Next line no comma "+ str(next_line_no_comma))
				par_count = kernel_param_map[next_line_no_comma]
				#print("Count: " + str(par_count) + " i: "+str(i) + " total: " +str(par_count+i+2))
				#third_line = ptx[i+par_count+2]
				#out_ptx[i+par_count+2].write("param_mask")
				#print ("Extra param: " + ptx[i+par_count+2])
				param_line = i+par_count+2
				#print ("Line with last param: " + str(param_line))
		if (i == param_line):
			#print ("--- Param: "+ str(line))
			out_ptx.write("param_mask,\n")
		out_ptx.write(line)				
			#prev_line = line
			
def load_mask(ptx_file, out_ptx):
	print("\n############ LOAD MASK ###############\n")
	with open(ptx_file, 'r') as f:
		ptx = f.readlines()
	for line in ptx:
			#print("Line: "+line)
			if '//argAND' in line:
				#print("Line: "+line)
				tmp_reg = line.split()
				my_reg_and = tmp_reg[1].replace(",","")
			if '//argOR' in line:
				#print("Line: "+line)
				tmp_reg1 = line.split()
				my_reg_or = tmp_reg1[1].replace(",","")
				#print("Reg: "+str(my_reg_or))
			if 'ld.global.v' in line or 'ld.global.nc.v' in line:
#				load_instructions.append(line)
				register = line.split()
				sizeOfRegisterList = len(register)
				#print("Register: "+str(register))
				#print("Size of register list : "+str(len(register)))

				if (sizeOfRegisterList == 2):
					regSz2 = register[1].split(",")
					#print("Register cutted: "+str(regSz2))
					concat_reg = regSz2[1].replace("[","").replace("];","")
					#print("S2: "+str(concat_reg))
				elif (sizeOfRegisterList == 3):
					concat_reg = register[2].replace("[","").replace("];","")
					#print("S3: "+str(concat_reg))
				elif (sizeOfRegisterList == 4):
					concat_reg = register[3].replace("[","").replace("];","").replace("}","")
					#print("S4: "+str(concat_reg))
				elif (sizeOfRegisterList == 5):
					concat_reg = register[4].replace("[","").replace("];","")
				elif (sizeOfRegisterList == 6):
					concat_reg = register[5].replace("[","").replace("];","")
					#print("S6: "+str(concat_reg))
				else:
					print("Size greater than 6. Exit! "+str(sizeOfRegisterList))
					exit()

				
				#print("register: "+str(concat_reg))
				character = search_ptr_arthm(concat_reg)
				#print ("Load C: "+ character)
				#print("concat: "+concat_reg)
				if (character == ''):
					# Bitwise AND with 000 000 fff fff
					out_ptx.write("and.b64 " + concat_reg + ", " + 
						concat_reg + ", " + my_reg_and +";\n")
					out_ptx.write("or.b64 " + concat_reg + ", " + 
						concat_reg + ", " + my_reg_or +";\n")
					out_ptx.write(line)
				# if there is +, - in ld and st instructions
				elif (character == '+'): 
					#print ("G index "+str(int(g_index)))
					without_arithm = concat_reg[0:int(g_index)]
					number = concat_reg[int(g_index):]
					#print("St Num: " + number)
					out_ptx.write("\tadd.s64 \t" + without_arithm + ", "
						+ without_arithm
						+ ", "+ number + "; //+or- mask\n")

					out_ptx.write("\tand.b64 \t" + without_arithm + ", " + 
						without_arithm +", "+ my_reg_and +";\n")
					out_ptx.write("\tor.b64 \t" + without_arithm + ", " + 
						without_arithm +", "+ my_reg_or +";\n")
					#print("without_arithm: "+str(without_arithm))
					split = line.split();
					#print("Split: "+str(split))
					operation = split[0]
					#print("Operation: "+operation)
					v_split = operation.split(".")
					#print("v_plit len: "+str(len(v_split)))
					sizeOp = len(v_split)
					v = v_split[sizeOp-2]
					#print("Version: "+v)
					if ( v == 'v2'  ):
						final_reg = split[1]+split[2]
					elif ( v == 'v4' ):
						final_reg = split[1]+split[2]+split[3]+split[4]
					else:
						print("Index: "+str(int(g_index))+" Not supported. Exit!!!")
						print("Line: "+str(line))
						exit()
					#print("final reg: "+final_reg)
					out_ptx.write("\t"+operation+ "\t"
						+ final_reg + " [" + without_arithm +"];\n")
					out_ptx.write("\tsub.s64 \t" + without_arithm 
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					#print("LD Remove: "+without_arithm)
					#out_ptx.write(line)
				else:
					print("Character in:"+line+" is - or +- please fix it!!")
					out_ptx.write(line)
			# load 
			elif 'ld.global.' in line or 'ld.volatile.global' in line:
#				load_instructions.append(line)
				register = line.split()
				sizeOfRegisterList = len(register)
				#print("Register: "+str(register))
				#print("Size of register list : "+str(len(register)))

				if (sizeOfRegisterList == 2):
					regSz2 = register[1].split(",")
					#print("Register cutted: "+str(regSz2))
					concat_reg = regSz2[1].replace("[","").replace("];","")
					#print("S2: "+str(concat_reg))
				elif (sizeOfRegisterList == 3):
					concat_reg = register[2].replace("[","").replace("];","")
					#print("S3: "+str(concat_reg))
				elif (sizeOfRegisterList == 4):
					concat_reg = register[3].replace("[","").replace("];","")
					#print("S4: "+str(concat_reg))
				elif (sizeOfRegisterList == 5):
					concat_reg = register[4].replace("[","").replace("];","")
				elif (sizeOfRegisterList == 6):
					concat_reg = register[5].replace("[","").replace("];","")
				else:
					print("Size greater than 6. Exit! "+str(sizeOfRegisterList))
					exit()

				
				#print("register: "+str(concat_reg))
				character = search_ptr_arthm(concat_reg)
				#print ("Load C: "+ character)
				#print("concat: "+concat_reg)
				if (character == ''):
					out_ptx.write("and.b64 " + concat_reg + ", " + 
						concat_reg + ", " + my_reg_and +";\n")
					out_ptx.write("or.b64 " + concat_reg + ", " + 
						concat_reg + ", " + my_reg_or +";\n")
					out_ptx.write(line)
				# if there is +, - in ld and st instructions
				elif (character == '+'): 
					#print ("G index "+str(int(g_index)))
					without_arithm = concat_reg[0:int(g_index)]
					number = concat_reg[int(g_index):]
                                        #print("St Num: " + number)
					out_ptx.write("\tadd.s64 \t" + without_arithm + ", "
						+ without_arithm
						+ ", "+ number + "; //+or- mask\n")

					out_ptx.write("\tand.b64 \t" + without_arithm + ", " + 
						without_arithm +", "+ my_reg_and +";\n")
					out_ptx.write("\tor.b64 \t" + without_arithm + ", " + 
						without_arithm +", "+ my_reg_or +";\n")
			
					#print("without_arithm: "+str(without_arithm))
					split = line.split();
					operation = split[0]
					#print("Operation: "+operation)
					final_reg = split[1]
					#print("final reg: "+final_reg)
					out_ptx.write("\t"+operation+ "\t"
						+ final_reg + " [" + without_arithm +"];\n")
					out_ptx.write("\tsub.s64 \t" + without_arithm 
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					#print("LD Remove: "+without_arithm)
					#out_ptx.write(line)
				else:
					print("Character in:"+line+" is - or +- please fix it!!")
					out_ptx.write(line)

			elif '@p0 st.global' in line :
				#store_instructions.append(line)
				# split instruction line 
				register = line.split()
				# keep only the register used
				concat_reg = register[2].replace("[",
						"").replace("],","").replace(";","")
				#print("concat: "+concat_reg)
				# For the case %rd12+4: get +, -, +-
				character = search_ptr_arthm(concat_reg)
				#print ("C: "+ character)
				if (character == ''):
				# append to the new ptx the bitwise op
					out_ptx.write("\tand.b64 \t" + concat_reg + ", " + 
						concat_reg + ", "  + my_reg_and +";\n")	
					out_ptx.write("\tor.b64 \t" + concat_reg + ", " + 
						concat_reg + ", "  + my_reg_or +";\n")	
					out_ptx.write(line)
				# if there is +, - in ld and st instructions
				elif (character == '+'): 
					#print ("G index "+str(int(g_index)))
					without_arithm = concat_reg[0:int(g_index)]
					#print("Register "+without_arithm)
					number = concat_reg[int(g_index):]
					#print("St Num: " + number)
					out_ptx.write("\tadd.s64 \t" + without_arithm
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					out_ptx.write("\tand.b64 \t" + without_arithm + ", " + 
						without_arithm +", " + my_reg_and +";\n")
					out_ptx.write("\tor.b64 \t" + without_arithm + ", " + 
						without_arithm +", " + my_reg_or +";\n")
					split = line.split();
					#print("Split: "+str(split))
					operation = split[0]
					#print("Operation: "+operation)
					v_split = operation.split(".")
					v = v_split[2]
					#print("Version: "+v)
					if ( v == 'v2'  ):
						final_reg = split[2]+split[3]
					elif ( v == 'v4' ):
						final_reg = split[2]+split[3]+split[4]+split[5]
					else:
						final_reg = split[2]
					#print("final reg: "+final_reg)
					out_ptx.write("\t"+operation+ "\t[" 
						+ without_arithm + "], " + final_reg +"\n")
					out_ptx.write("\tsub.s64 \t" + without_arithm
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					#out_ptx.write(line)
			# store 
			elif 'st.global' in line or 'st.volatile.global' in line:
				#store_instructions.append(line)
				# split instruction line 
				register = line.split()
				# keep only the register used
				concat_reg = register[1].replace("[",
						"").replace("],","").replace(";","")
				#print("concat: "+concat_reg)
				# For the case %rd12+4: get +, -, +-
				character = search_ptr_arthm(concat_reg)
				#print ("C: "+ character)
				if (character == ''):
				# append to the new ptx the bitwise op
					out_ptx.write("\tand.b64 \t" + concat_reg + ", " + 
						concat_reg + ", "  + my_reg_and +";\n")	
					out_ptx.write("\tor.b64 \t" + concat_reg + ", " + 
						concat_reg + ", "  + my_reg_or +";\n")	
					out_ptx.write(line)
				# if there is +, - in ld and st instructions
				elif (character == '+'): 
					#print ("G index "+str(int(g_index)))
					without_arithm = concat_reg[0:int(g_index)]
					#print("Register "+without_arithm)
					number = concat_reg[int(g_index):]
					#print("St Num: " + number)
					out_ptx.write("\tadd.s64 \t" + without_arithm
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					out_ptx.write("\tand.b64 \t" + without_arithm + ", " + 
						without_arithm +", " + my_reg_and +";\n")
					out_ptx.write("\tor.b64 \t" + without_arithm + ", " + 
						without_arithm +", " + my_reg_or +";\n")
					split = line.split();
					#print("Split: "+str(split))
					operation = split[0]
					#print("Operation: "+operation)
					v_split = operation.split(".")
					v = v_split[2]
					#print("Version: "+v)
					if ( v == 'v2'  ):
						final_reg = split[2]+split[3]
					elif ( v == 'v4' ):
						final_reg = split[2]+split[3]+split[4]+split[5]
					else:
						final_reg = split[2]
					#print("final reg: "+final_reg)
					out_ptx.write("\t"+operation+ "\t[" 
						+ without_arithm + "], " + final_reg +"\n")
					out_ptx.write("\tsub.s64 \t" + without_arithm
						+ ", " + without_arithm
						+ ", "+ number + "; //+or- mask\n")
					#out_ptx.write(line)
				else:
					print("Character in:"+line+" is - or +- please fix it!!")
					out_ptx.write(line)

					#print(line)
			elif 'global' in line and 'ld' in line:
				instructions_not_supported.append(line)
				out_ptx.write(line)
			else:
				out_ptx.write(line)
def find_char_index(s, char):
    for i, c in enumerate(s):
        if c == char:
            return i
    return -1

def find_param_num(ptx_file):
	print("\n############ FIND PARAM NUM ###############\n")
	all_parameters = []	
	param_count = 0
	func = 0
	with open(ptx_file, 'r') as f:
		ptx = f.readlines()
		current_kernel = ""
		ignore_function = ""
		for line in ptx:                      
			if '.entry' in line:
				func = 0
				param_count=0;
				del all_parameters[:]
				kernel = line.split()
				#print ("1.Kernel: "+str(kernel))
				if '.visible' in kernel:
					current_kernel = kernel[2].replace("(","")
				elif '.weak' in kernel:
					current_kernel = kernel[2].replace("(","")
				else:
					current_kernel = kernel[1].replace("(","")
				#print ("1. Kernel name: "+str(current_kernel))
			if '.func' in line and not 'vprintf' in line:
				first = 1
				func = 1
				found_reg = 1
				kernel = line.split()
				
				if '.visible' in kernel:
					current_kernel = kernel[5].replace("(","")
				elif '.weak' in kernel:
					current_kernel = kernel[5].replace("(","")
				else:
					current_kernel = kernel[4].replace("(","")
				#print("2. Current kernel: "+ str(current_kernel))

			#if '.param' in line and ignore_function not in line:
			if '.param' in line and '_param_' in line:
				if 'ld.param.' not in line:
					#and ignore_function not in line:
					#print("\n -- Line: "+str(line))
					#print("1. --> Param Current kernel: "+current_kernel)
					kernel_parameter = line.split()
					#print("1. Kernekl+Parameter " + str(kernel_parameter))
					parameterSize = len(kernel_parameter)
					#print("1. Parameter Size: " + str(parameterSize))
					if (parameterSize == 3):
						parameter = re.split(str(current_kernel),
								str(kernel_parameter[2]))
						#print("- S3 Parameter: "+str(parameter))
					elif (parameterSize == 4):
						parameter = re.split(str(current_kernel),
								str(kernel_parameter[3]))
						#print("-- S4 Parameter: "+str(parameter))
					elif (parameterSize == 5):
						tmpParam = re.split(str(current_kernel),
								str(kernel_parameter[4]))
						#print("--- S5 Parameter: "+str(tmpParam))
						paramSplit = tmpParam[1].split("[")
						parameter = paramSplit[0]

					else:
						print("Parameter Size greater than 5")
	
					#print("Parameter: "+str(parameter))
					clear_param = str(parameter).replace("['', '_","").replace(",']","").replace("']","")
					#print("Clear Parameter: "+str(clear_param))
					param_count=param_count+1

			if 'ret;' in line:
				#print ("Ret: " + current_kernel)
				kernel_param_map[current_kernel] = param_count
				param_count = 0
				#print("count "+str(param_count))

def add_extras(ptx_file, out_ptx):
	find_param_num(in_ptx_file)
	print("\n############ ADD EXTRAS ###############\n")
	#print("MAP:"+str(kernel_param_map))
	with open(ptx_file, 'r') as f:
		ptx = f.readlines()
		current_kernel = ""
		ignore_function = ""
		for line in ptx:
			if '.entry' in line:
				first = 1
				func = 0
				found_reg = 1
				kernel = line.split()
				
				if '.visible' in kernel:
					current_kernel = kernel[2].replace("(","")
				elif '.weak' in kernel:
					current_kernel = kernel[2].replace("(","")
				else:
					current_kernel = kernel[1].replace("(","")
				#print("2. Current kernel: "+ str(current_kernel))

				out_ptx.write(line)
			elif '.func' in line and not 'vprintf' in line:
				first = 1
				func = 1
				found_reg = 1
				kernel = line.split()
				
				if '.visible' in kernel:
					current_kernel = kernel[5].replace("(","")
				elif '.weak' in kernel:
					current_kernel = kernel[5].replace("(","")
				else:
					current_kernel = kernel[4].replace("(","")
				#print("2. Current function: "+ str(current_kernel))
				out_ptx.write(line)
			elif 'vprintf' in line:
				out_ptx.write(line)
			# Add the extra parameter
			elif '.reg .b64' in line or '.reg .f64' in line:
				#print("Line: " + line)
				new_registers = []
				new_registers.append(line)
				if found_reg == 1:
					#print("NEW reg: "+str(new_registers))
					new_registers.append("\t.reg .b64 \t%maskreg<3>;\n")
					#print("2. NEW reg: "+str(new_registers))
					out_ptx.writelines(new_registers)
					found_reg = 0 
				else:
					out_ptx.writelines(line)
				#print("Line: "+line)

			#elif '.param' in line and ignore_function not in line:
			elif 'mov.u64' in line and current_kernel in line:
				#print("######## kernel: "+str(current_kernel))
				kernel_parameter = line.split()
				#print("kernel parameter: "+str(kernel_parameter))
				kernel_args_len = len(kernel_parameter)
				#print("Kernel Args len: " + str(kernel_args_len ))
				#print("!!!!!!!!!!!!! ld.param in line"+str(line))
				new_lines = []
				new_lines.append(line)
				#print("Kernel: "+str(current_kernel))
				#print("New line: "+str(new_lines))
				if first == 1:
					#print("--- First "+str(line))
					new_lines.append("\tld.param.u64\t%maskreg1, ["
						+str(current_kernel)+"_param_"
						+str(par_count)+"];"+"   //argAND"+"\n")
					new_lines.append("\tld.param.u64\t%maskreg2, ["
						+str(current_kernel)+"_param_"
						+str(par_count)+"];"+"   //argOR"+"\n")
					out_ptx.writelines(new_lines)
					first = 0
				else:
					out_ptx.write(line)
					#print("!!! Not first "+str(line))
                
			elif '.param' in line and '_param_' in line:
			#elif '.param' in line:
				kernel_parameter = line.split()
				#print("kernel parameter: "+str(kernel_parameter))
				kernel_args_len = len(kernel_parameter)
				#print("Kernel Args len: " + str(kernel_args_len ))
				if 'ld.param.' not in line:
					#print("Kernel+Parameter " + str(kernel_parameter[kernel_args_len-1]))
					parameter = re.split(str(current_kernel),
							str(kernel_parameter[kernel_args_len-1]))
					#print("Parameter: "+str(parameter))
					tmp_clear_param = str(parameter).replace("['', '_","").replace(",']","").replace("']","")
					clear_param = re.sub(r'\[\d+\]', '', tmp_clear_param)
					#print("Clear param: "+str(clear_param))
					par_count = kernel_param_map[current_kernel]
					#print ("Count: "+str(par_count))
					pattern = 'param_'+str(par_count-1)
					#print("My pattern: "+pattern)
					#print("Par count "+str(par_count))
					if clear_param == pattern:
						out_ptx.write("\t.param .u64 " 
							+ str(current_kernel) 
							+ str(parameter[1])+",\n")
							#"_param_"+ str(par_count-1)+",\n")	
						out_ptx.write("\t.param .u64 " 
							+ str(current_kernel) + "_param_" 
							+ str(par_count) + ",\n")
						out_ptx.write("\t.param .u64 " 
							+ str(current_kernel) + "_param_" 
							+ str(par_count+1))
						#print("----- Clear Param == pattern")
					else:
						out_ptx.write(line)
						#print(" Clear Param != pattern")
				elif 'ld.param.' in line:
					#print("!!!!!!!!!!!!! ld.param in line"+str(line))
					new_lines = []
					new_lines.append(line)
					#print("Kernel: "+str(current_kernel))
					#print("New line: "+str(new_lines))
					if first == 1:
						#print("--- First "+str(line))
						new_lines.append("\tld.param.u64\t%maskreg1, ["
							+str(current_kernel)+"_param_"
							+str(par_count)+"];"+"   //argAND"+"\n")
						new_lines.append("\tld.param.u64\t%maskreg2, ["
							+str(current_kernel)+"_param_"
							+str(par_count+1)+"];"+"   //argOR"+"\n")
						out_ptx.writelines(new_lines)
						first = 0
					else:
						out_ptx.write(line)
						#print("!!! Not first "+str(line))

			else:
				out_ptx.write(line)
					
			
if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage: python script.py <in_ptx_file> <out_ptx_file>')
		sys.exit()

	in_ptx_file = sys.argv[1]
	out_ptx_file = sys.argv[2]
	out_ptx = open(out_ptx_file, 'w')
	tmp_ptx = open("tmp_ptx.ptx", 'w')
	mask_ptx = open("mask_ptx.ptx", 'w')
	
	#Add parameter + registers + ld for parameter
	add_extras(in_ptx_file, tmp_ptx)
#	in_ptx_file.close()
	tmp_ptx.close()

	load_mask("tmp_ptx.ptx", mask_ptx)
	mask_ptx.close()
	adjust_func("mask_ptx.ptx", out_ptx)
	for ld_inst in instructions_not_supported:
		print(ld_inst)
		
#	for st_inst in store_instructions:
#		print(st_inst)
