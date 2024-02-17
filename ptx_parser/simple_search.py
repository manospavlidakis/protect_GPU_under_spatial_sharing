#! /usr/bin/env python3
import re

ptx_code = """
add.u64 %rd82, %SP, 4;
{ 
    .reg .b32 temp_param_reg;
.param .b64 param0;
st.param.f64 [param0+0], %fd374;
.param .b64 param1;
st.param.b64 [param1+0], %rd82;
.param .b64 retval0;
call.uni (retval0), 
__internal_trig_reduction_slowpathd, 
(
param0, 
param1
);
ld.param.f64 %fd375, [retval0+0];
} 
"""

# Define the regular expression pattern
pattern = r'call\.uni.*\n\s*(.*?)\n'

# Search for the pattern and extract the next line
match = re.search(pattern, ptx_code)
if match:
    next_line = match.group(1)
    print(next_line)
else:
    print("Pattern not found")

