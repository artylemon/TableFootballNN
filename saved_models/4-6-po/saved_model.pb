¬
¼
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018´Õ	

Adam/dense_320/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_320/bias/v
{
)Adam/dense_320/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_320/bias/v*
_output_shapes
:*
dtype0

Adam/dense_320/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_320/kernel/v

+Adam/dense_320/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_320/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_319/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_319/bias/v
{
)Adam/dense_319/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_319/bias/v*
_output_shapes
:*
dtype0

Adam/dense_319/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_319/kernel/v

+Adam/dense_319/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_319/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_318/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_318/bias/v
{
)Adam/dense_318/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_318/bias/v*
_output_shapes
:*
dtype0

Adam/dense_318/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_318/kernel/v

+Adam/dense_318/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_318/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_317/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_317/bias/v
{
)Adam/dense_317/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_317/bias/v*
_output_shapes
:*
dtype0

Adam/dense_317/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_317/kernel/v

+Adam/dense_317/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_317/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_316/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_316/bias/v
{
)Adam/dense_316/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_316/bias/v*
_output_shapes
:*
dtype0

Adam/dense_316/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_316/kernel/v

+Adam/dense_316/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_316/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_315/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_315/bias/v
{
)Adam/dense_315/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_315/bias/v*
_output_shapes
:*
dtype0

Adam/dense_315/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_315/kernel/v

+Adam/dense_315/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_315/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_314/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_314/bias/v
{
)Adam/dense_314/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_314/bias/v*
_output_shapes
:*
dtype0

Adam/dense_314/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_314/kernel/v

+Adam/dense_314/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_314/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_320/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_320/bias/m
{
)Adam/dense_320/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_320/bias/m*
_output_shapes
:*
dtype0

Adam/dense_320/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_320/kernel/m

+Adam/dense_320/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_320/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_319/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_319/bias/m
{
)Adam/dense_319/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_319/bias/m*
_output_shapes
:*
dtype0

Adam/dense_319/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_319/kernel/m

+Adam/dense_319/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_319/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_318/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_318/bias/m
{
)Adam/dense_318/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_318/bias/m*
_output_shapes
:*
dtype0

Adam/dense_318/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_318/kernel/m

+Adam/dense_318/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_318/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_317/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_317/bias/m
{
)Adam/dense_317/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_317/bias/m*
_output_shapes
:*
dtype0

Adam/dense_317/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_317/kernel/m

+Adam/dense_317/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_317/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_316/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_316/bias/m
{
)Adam/dense_316/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_316/bias/m*
_output_shapes
:*
dtype0

Adam/dense_316/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_316/kernel/m

+Adam/dense_316/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_316/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_315/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_315/bias/m
{
)Adam/dense_315/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_315/bias/m*
_output_shapes
:*
dtype0

Adam/dense_315/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_315/kernel/m

+Adam/dense_315/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_315/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_314/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_314/bias/m
{
)Adam/dense_314/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_314/bias/m*
_output_shapes
:*
dtype0

Adam/dense_314/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_314/kernel/m

+Adam/dense_314/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_314/kernel/m*
_output_shapes

:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
t
dense_320/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_320/bias
m
"dense_320/bias/Read/ReadVariableOpReadVariableOpdense_320/bias*
_output_shapes
:*
dtype0
|
dense_320/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_320/kernel
u
$dense_320/kernel/Read/ReadVariableOpReadVariableOpdense_320/kernel*
_output_shapes

:*
dtype0
t
dense_319/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_319/bias
m
"dense_319/bias/Read/ReadVariableOpReadVariableOpdense_319/bias*
_output_shapes
:*
dtype0
|
dense_319/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_319/kernel
u
$dense_319/kernel/Read/ReadVariableOpReadVariableOpdense_319/kernel*
_output_shapes

:*
dtype0
t
dense_318/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_318/bias
m
"dense_318/bias/Read/ReadVariableOpReadVariableOpdense_318/bias*
_output_shapes
:*
dtype0
|
dense_318/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_318/kernel
u
$dense_318/kernel/Read/ReadVariableOpReadVariableOpdense_318/kernel*
_output_shapes

:*
dtype0
t
dense_317/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_317/bias
m
"dense_317/bias/Read/ReadVariableOpReadVariableOpdense_317/bias*
_output_shapes
:*
dtype0
|
dense_317/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_317/kernel
u
$dense_317/kernel/Read/ReadVariableOpReadVariableOpdense_317/kernel*
_output_shapes

:*
dtype0
t
dense_316/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_316/bias
m
"dense_316/bias/Read/ReadVariableOpReadVariableOpdense_316/bias*
_output_shapes
:*
dtype0
|
dense_316/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_316/kernel
u
$dense_316/kernel/Read/ReadVariableOpReadVariableOpdense_316/kernel*
_output_shapes

:*
dtype0
t
dense_315/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_315/bias
m
"dense_315/bias/Read/ReadVariableOpReadVariableOpdense_315/bias*
_output_shapes
:*
dtype0
|
dense_315/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_315/kernel
u
$dense_315/kernel/Read/ReadVariableOpReadVariableOpdense_315/kernel*
_output_shapes

:*
dtype0
t
dense_314/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_314/bias
m
"dense_314/bias/Read/ReadVariableOpReadVariableOpdense_314/bias*
_output_shapes
:*
dtype0
|
dense_314/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_314/kernel
u
$dense_314/kernel/Read/ReadVariableOpReadVariableOpdense_314/kernel*
_output_shapes

:*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
^
ConstConst*
_output_shapes

:*
dtype0*!
valueB"þÒB^'C
`
Const_1Const*
_output_shapes

:*
dtype0*!
valueB"ëoEDÏE

NoOpNoOp
]
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*Ã\
value¹\B¶\ B¯\

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¾
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias*
¦
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias*
¦
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias*
¦
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
¦
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias*
¦
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias*
¦
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*

0
1
2
!3
"4
)5
*6
17
28
99
:10
A11
B12
I13
J14
Q15
R16*
j
!0
"1
)2
*3
14
25
96
:7
A8
B9
I10
J11
Q12
R13*
* 
°
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
6
\trace_0
]trace_1
^trace_2
_trace_3* 
* 
Ü
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rate!m"m)m*m 1m¡2m¢9m£:m¤Am¥Bm¦Im§Jm¨Qm©Rmª!v«"v¬)v­*v®1v¯2v°9v±:v²Av³Bv´IvµJv¶Qv·Rv¸*

eserving_default* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_15layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE*

ftrace_0* 

!0
"1*

!0
"1*
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 
`Z
VARIABLE_VALUEdense_314/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_314/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

strace_0* 

ttrace_0* 
`Z
VARIABLE_VALUEdense_315/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_315/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 

unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

ztrace_0* 

{trace_0* 
`Z
VARIABLE_VALUEdense_316/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_316/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 

|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_317/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_317/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_318/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_318/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

I0
J1*

I0
J1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_319/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_319/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1*

Q0
R1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEdense_320/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_320/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
<
0
1
2
3
4
5
6
7*

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_314/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_314/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_315/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_315/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_316/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_316/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_317/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_317/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_318/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_318/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_319/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_319/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_320/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_320/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_314/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_314/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_315/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_315/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_316/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_316/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_317/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_317/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_318/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_318/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_319/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_319/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_320/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_320/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_normalization_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Û
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConstConst_1dense_314/kerneldense_314/biasdense_315/kerneldense_315/biasdense_316/kerneldense_316/biasdense_317/kerneldense_317/biasdense_318/kerneldense_318/biasdense_319/kerneldense_319/biasdense_320/kerneldense_320/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1527204
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount_1/Read/ReadVariableOp$dense_314/kernel/Read/ReadVariableOp"dense_314/bias/Read/ReadVariableOp$dense_315/kernel/Read/ReadVariableOp"dense_315/bias/Read/ReadVariableOp$dense_316/kernel/Read/ReadVariableOp"dense_316/bias/Read/ReadVariableOp$dense_317/kernel/Read/ReadVariableOp"dense_317/bias/Read/ReadVariableOp$dense_318/kernel/Read/ReadVariableOp"dense_318/bias/Read/ReadVariableOp$dense_319/kernel/Read/ReadVariableOp"dense_319/bias/Read/ReadVariableOp$dense_320/kernel/Read/ReadVariableOp"dense_320/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_314/kernel/m/Read/ReadVariableOp)Adam/dense_314/bias/m/Read/ReadVariableOp+Adam/dense_315/kernel/m/Read/ReadVariableOp)Adam/dense_315/bias/m/Read/ReadVariableOp+Adam/dense_316/kernel/m/Read/ReadVariableOp)Adam/dense_316/bias/m/Read/ReadVariableOp+Adam/dense_317/kernel/m/Read/ReadVariableOp)Adam/dense_317/bias/m/Read/ReadVariableOp+Adam/dense_318/kernel/m/Read/ReadVariableOp)Adam/dense_318/bias/m/Read/ReadVariableOp+Adam/dense_319/kernel/m/Read/ReadVariableOp)Adam/dense_319/bias/m/Read/ReadVariableOp+Adam/dense_320/kernel/m/Read/ReadVariableOp)Adam/dense_320/bias/m/Read/ReadVariableOp+Adam/dense_314/kernel/v/Read/ReadVariableOp)Adam/dense_314/bias/v/Read/ReadVariableOp+Adam/dense_315/kernel/v/Read/ReadVariableOp)Adam/dense_315/bias/v/Read/ReadVariableOp+Adam/dense_316/kernel/v/Read/ReadVariableOp)Adam/dense_316/bias/v/Read/ReadVariableOp+Adam/dense_317/kernel/v/Read/ReadVariableOp)Adam/dense_317/bias/v/Read/ReadVariableOp+Adam/dense_318/kernel/v/Read/ReadVariableOp)Adam/dense_318/bias/v/Read/ReadVariableOp+Adam/dense_319/kernel/v/Read/ReadVariableOp)Adam/dense_319/bias/v/Read/ReadVariableOp+Adam/dense_320/kernel/v/Read/ReadVariableOp)Adam/dense_320/bias/v/Read/ReadVariableOpConst_2*A
Tin:
826		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1527716
Å

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_314/kerneldense_314/biasdense_315/kerneldense_315/biasdense_316/kerneldense_316/biasdense_317/kerneldense_317/biasdense_318/kerneldense_318/biasdense_319/kerneldense_319/biasdense_320/kerneldense_320/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_314/kernel/mAdam/dense_314/bias/mAdam/dense_315/kernel/mAdam/dense_315/bias/mAdam/dense_316/kernel/mAdam/dense_316/bias/mAdam/dense_317/kernel/mAdam/dense_317/bias/mAdam/dense_318/kernel/mAdam/dense_318/bias/mAdam/dense_319/kernel/mAdam/dense_319/bias/mAdam/dense_320/kernel/mAdam/dense_320/bias/mAdam/dense_314/kernel/vAdam/dense_314/bias/vAdam/dense_315/kernel/vAdam/dense_315/bias/vAdam/dense_316/kernel/vAdam/dense_316/bias/vAdam/dense_317/kernel/vAdam/dense_317/bias/vAdam/dense_318/kernel/vAdam/dense_318/bias/vAdam/dense_319/kernel/vAdam/dense_319/bias/vAdam/dense_320/kernel/vAdam/dense_320/bias/v*@
Tin9
725*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1527882Ýî
É	
÷
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_314_layer_call_fn_1527405

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


/__inference_sequential_68_layer_call_fn_1526840
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_68_layer_call_and_return_conditional_losses_1526805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:


÷
F__inference_dense_314_layer_call_and_return_conditional_losses_1527416

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_317_layer_call_fn_1527465

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_319_layer_call_and_return_conditional_losses_1527516

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

/__inference_sequential_68_layer_call_fn_1527241

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_68_layer_call_and_return_conditional_losses_1526805o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:


÷
F__inference_dense_318_layer_call_and_return_conditional_losses_1527496

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

/__inference_sequential_68_layer_call_fn_1527278

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_68_layer_call_and_return_conditional_losses_1526995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_315_layer_call_fn_1527425

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý+

J__inference_sequential_68_layer_call_and_return_conditional_losses_1526995

inputs
normalization_sub_y
normalization_sqrt_x#
dense_314_1526959:
dense_314_1526961:#
dense_315_1526964:
dense_315_1526966:#
dense_316_1526969:
dense_316_1526971:#
dense_317_1526974:
dense_317_1526976:#
dense_318_1526979:
dense_318_1526981:#
dense_319_1526984:
dense_319_1526986:#
dense_320_1526989:
dense_320_1526991:
identity¢!dense_314/StatefulPartitionedCall¢!dense_315/StatefulPartitionedCall¢!dense_316/StatefulPartitionedCall¢!dense_317/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢!dense_320/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_314/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_314_1526959dense_314_1526961*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697
!dense_315/StatefulPartitionedCallStatefulPartitionedCall*dense_314/StatefulPartitionedCall:output:0dense_315_1526964dense_315_1526966*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714
!dense_316/StatefulPartitionedCallStatefulPartitionedCall*dense_315/StatefulPartitionedCall:output:0dense_316_1526969dense_316_1526971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731
!dense_317/StatefulPartitionedCallStatefulPartitionedCall*dense_316/StatefulPartitionedCall:output:0dense_317_1526974dense_317_1526976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748
!dense_318/StatefulPartitionedCallStatefulPartitionedCall*dense_317/StatefulPartitionedCall:output:0dense_318_1526979dense_318_1526981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765
!dense_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0dense_319_1526984dense_319_1526986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782
!dense_320/StatefulPartitionedCallStatefulPartitionedCall*dense_319/StatefulPartitionedCall:output:0dense_320_1526989dense_320_1526991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798y
IdentityIdentity*dense_320/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_314/StatefulPartitionedCall"^dense_315/StatefulPartitionedCall"^dense_316/StatefulPartitionedCall"^dense_317/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall"^dense_320/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_314/StatefulPartitionedCall!dense_314/StatefulPartitionedCall2F
!dense_315/StatefulPartitionedCall!dense_315/StatefulPartitionedCall2F
!dense_316/StatefulPartitionedCall!dense_316/StatefulPartitionedCall2F
!dense_317/StatefulPartitionedCall!dense_317/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2F
!dense_320/StatefulPartitionedCall!dense_320/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
ÓU
»
"__inference__wrapped_model_1526672
normalization_input%
!sequential_68_normalization_sub_y&
"sequential_68_normalization_sqrt_xH
6sequential_68_dense_314_matmul_readvariableop_resource:E
7sequential_68_dense_314_biasadd_readvariableop_resource:H
6sequential_68_dense_315_matmul_readvariableop_resource:E
7sequential_68_dense_315_biasadd_readvariableop_resource:H
6sequential_68_dense_316_matmul_readvariableop_resource:E
7sequential_68_dense_316_biasadd_readvariableop_resource:H
6sequential_68_dense_317_matmul_readvariableop_resource:E
7sequential_68_dense_317_biasadd_readvariableop_resource:H
6sequential_68_dense_318_matmul_readvariableop_resource:E
7sequential_68_dense_318_biasadd_readvariableop_resource:H
6sequential_68_dense_319_matmul_readvariableop_resource:E
7sequential_68_dense_319_biasadd_readvariableop_resource:H
6sequential_68_dense_320_matmul_readvariableop_resource:E
7sequential_68_dense_320_biasadd_readvariableop_resource:
identity¢.sequential_68/dense_314/BiasAdd/ReadVariableOp¢-sequential_68/dense_314/MatMul/ReadVariableOp¢.sequential_68/dense_315/BiasAdd/ReadVariableOp¢-sequential_68/dense_315/MatMul/ReadVariableOp¢.sequential_68/dense_316/BiasAdd/ReadVariableOp¢-sequential_68/dense_316/MatMul/ReadVariableOp¢.sequential_68/dense_317/BiasAdd/ReadVariableOp¢-sequential_68/dense_317/MatMul/ReadVariableOp¢.sequential_68/dense_318/BiasAdd/ReadVariableOp¢-sequential_68/dense_318/MatMul/ReadVariableOp¢.sequential_68/dense_319/BiasAdd/ReadVariableOp¢-sequential_68/dense_319/MatMul/ReadVariableOp¢.sequential_68/dense_320/BiasAdd/ReadVariableOp¢-sequential_68/dense_320/MatMul/ReadVariableOp
sequential_68/normalization/subSubnormalization_input!sequential_68_normalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 sequential_68/normalization/SqrtSqrt"sequential_68_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_68/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3­
#sequential_68/normalization/MaximumMaximum$sequential_68/normalization/Sqrt:y:0.sequential_68/normalization/Maximum/y:output:0*
T0*
_output_shapes

:®
#sequential_68/normalization/truedivRealDiv#sequential_68/normalization/sub:z:0'sequential_68/normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_314/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_314_matmul_readvariableop_resource*
_output_shapes

:*
dtype0º
sequential_68/dense_314/MatMulMatMul'sequential_68/normalization/truediv:z:05sequential_68/dense_314/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_314/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_314_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_314/BiasAddBiasAdd(sequential_68/dense_314/MatMul:product:06sequential_68/dense_314/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_314/ReluRelu(sequential_68/dense_314/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_315/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_315_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_315/MatMulMatMul*sequential_68/dense_314/Relu:activations:05sequential_68/dense_315/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_315/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_315_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_315/BiasAddBiasAdd(sequential_68/dense_315/MatMul:product:06sequential_68/dense_315/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_315/ReluRelu(sequential_68/dense_315/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_316/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_316_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_316/MatMulMatMul*sequential_68/dense_315/Relu:activations:05sequential_68/dense_316/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_316/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_316_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_316/BiasAddBiasAdd(sequential_68/dense_316/MatMul:product:06sequential_68/dense_316/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_316/ReluRelu(sequential_68/dense_316/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_317/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_317_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_317/MatMulMatMul*sequential_68/dense_316/Relu:activations:05sequential_68/dense_317/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_317/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_317_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_317/BiasAddBiasAdd(sequential_68/dense_317/MatMul:product:06sequential_68/dense_317/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_317/ReluRelu(sequential_68/dense_317/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_318/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_318_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_318/MatMulMatMul*sequential_68/dense_317/Relu:activations:05sequential_68/dense_318/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_318/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_318_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_318/BiasAddBiasAdd(sequential_68/dense_318/MatMul:product:06sequential_68/dense_318/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_318/ReluRelu(sequential_68/dense_318/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_319/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_319_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_319/MatMulMatMul*sequential_68/dense_318/Relu:activations:05sequential_68/dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_319/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_319/BiasAddBiasAdd(sequential_68/dense_319/MatMul:product:06sequential_68/dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_68/dense_319/ReluRelu(sequential_68/dense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_68/dense_320/MatMul/ReadVariableOpReadVariableOp6sequential_68_dense_320_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_68/dense_320/MatMulMatMul*sequential_68/dense_319/Relu:activations:05sequential_68/dense_320/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_68/dense_320/BiasAdd/ReadVariableOpReadVariableOp7sequential_68_dense_320_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_68/dense_320/BiasAddBiasAdd(sequential_68/dense_320/MatMul:product:06sequential_68/dense_320/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_68/dense_320/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp/^sequential_68/dense_314/BiasAdd/ReadVariableOp.^sequential_68/dense_314/MatMul/ReadVariableOp/^sequential_68/dense_315/BiasAdd/ReadVariableOp.^sequential_68/dense_315/MatMul/ReadVariableOp/^sequential_68/dense_316/BiasAdd/ReadVariableOp.^sequential_68/dense_316/MatMul/ReadVariableOp/^sequential_68/dense_317/BiasAdd/ReadVariableOp.^sequential_68/dense_317/MatMul/ReadVariableOp/^sequential_68/dense_318/BiasAdd/ReadVariableOp.^sequential_68/dense_318/MatMul/ReadVariableOp/^sequential_68/dense_319/BiasAdd/ReadVariableOp.^sequential_68/dense_319/MatMul/ReadVariableOp/^sequential_68/dense_320/BiasAdd/ReadVariableOp.^sequential_68/dense_320/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2`
.sequential_68/dense_314/BiasAdd/ReadVariableOp.sequential_68/dense_314/BiasAdd/ReadVariableOp2^
-sequential_68/dense_314/MatMul/ReadVariableOp-sequential_68/dense_314/MatMul/ReadVariableOp2`
.sequential_68/dense_315/BiasAdd/ReadVariableOp.sequential_68/dense_315/BiasAdd/ReadVariableOp2^
-sequential_68/dense_315/MatMul/ReadVariableOp-sequential_68/dense_315/MatMul/ReadVariableOp2`
.sequential_68/dense_316/BiasAdd/ReadVariableOp.sequential_68/dense_316/BiasAdd/ReadVariableOp2^
-sequential_68/dense_316/MatMul/ReadVariableOp-sequential_68/dense_316/MatMul/ReadVariableOp2`
.sequential_68/dense_317/BiasAdd/ReadVariableOp.sequential_68/dense_317/BiasAdd/ReadVariableOp2^
-sequential_68/dense_317/MatMul/ReadVariableOp-sequential_68/dense_317/MatMul/ReadVariableOp2`
.sequential_68/dense_318/BiasAdd/ReadVariableOp.sequential_68/dense_318/BiasAdd/ReadVariableOp2^
-sequential_68/dense_318/MatMul/ReadVariableOp-sequential_68/dense_318/MatMul/ReadVariableOp2`
.sequential_68/dense_319/BiasAdd/ReadVariableOp.sequential_68/dense_319/BiasAdd/ReadVariableOp2^
-sequential_68/dense_319/MatMul/ReadVariableOp-sequential_68/dense_319/MatMul/ReadVariableOp2`
.sequential_68/dense_320/BiasAdd/ReadVariableOp.sequential_68/dense_320/BiasAdd/ReadVariableOp2^
-sequential_68/dense_320/MatMul/ReadVariableOp-sequential_68/dense_320/MatMul/ReadVariableOp:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
'
Á
__inference_adapt_step_21695
iterator%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢IteratorGetNext¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢add/ReadVariableOp
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*
_output_shapes

: *
output_shapes

: *
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*
_output_shapes

: l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 ^
ShapeConst*
_output_shapes
:*
dtype0	*%
valueB	"               Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:¥
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0*
validate_shape(
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*
validate_shape(*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator
¹C
²
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527337

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_314_matmul_readvariableop_resource:7
)dense_314_biasadd_readvariableop_resource::
(dense_315_matmul_readvariableop_resource:7
)dense_315_biasadd_readvariableop_resource::
(dense_316_matmul_readvariableop_resource:7
)dense_316_biasadd_readvariableop_resource::
(dense_317_matmul_readvariableop_resource:7
)dense_317_biasadd_readvariableop_resource::
(dense_318_matmul_readvariableop_resource:7
)dense_318_biasadd_readvariableop_resource::
(dense_319_matmul_readvariableop_resource:7
)dense_319_biasadd_readvariableop_resource::
(dense_320_matmul_readvariableop_resource:7
)dense_320_biasadd_readvariableop_resource:
identity¢ dense_314/BiasAdd/ReadVariableOp¢dense_314/MatMul/ReadVariableOp¢ dense_315/BiasAdd/ReadVariableOp¢dense_315/MatMul/ReadVariableOp¢ dense_316/BiasAdd/ReadVariableOp¢dense_316/MatMul/ReadVariableOp¢ dense_317/BiasAdd/ReadVariableOp¢dense_317/MatMul/ReadVariableOp¢ dense_318/BiasAdd/ReadVariableOp¢dense_318/MatMul/ReadVariableOp¢ dense_319/BiasAdd/ReadVariableOp¢dense_319/MatMul/ReadVariableOp¢ dense_320/BiasAdd/ReadVariableOp¢dense_320/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_314/MatMul/ReadVariableOpReadVariableOp(dense_314_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_314/MatMulMatMulnormalization/truediv:z:0'dense_314/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_314/BiasAdd/ReadVariableOpReadVariableOp)dense_314_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_314/BiasAddBiasAdddense_314/MatMul:product:0(dense_314/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_314/ReluReludense_314/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_315/MatMul/ReadVariableOpReadVariableOp(dense_315_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_315/MatMulMatMuldense_314/Relu:activations:0'dense_315/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_315/BiasAdd/ReadVariableOpReadVariableOp)dense_315_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_315/BiasAddBiasAdddense_315/MatMul:product:0(dense_315/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_315/ReluReludense_315/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_316/MatMul/ReadVariableOpReadVariableOp(dense_316_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_316/MatMulMatMuldense_315/Relu:activations:0'dense_316/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_316/BiasAdd/ReadVariableOpReadVariableOp)dense_316_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_316/BiasAddBiasAdddense_316/MatMul:product:0(dense_316/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_316/ReluReludense_316/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_317/MatMul/ReadVariableOpReadVariableOp(dense_317_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_317/MatMulMatMuldense_316/Relu:activations:0'dense_317/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_317/BiasAdd/ReadVariableOpReadVariableOp)dense_317_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_317/BiasAddBiasAdddense_317/MatMul:product:0(dense_317/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_317/ReluReludense_317/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_318/MatMul/ReadVariableOpReadVariableOp(dense_318_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_318/MatMulMatMuldense_317/Relu:activations:0'dense_318/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_318/BiasAdd/ReadVariableOpReadVariableOp)dense_318_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_318/BiasAddBiasAdddense_318/MatMul:product:0(dense_318/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_318/ReluReludense_318/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_319/MatMul/ReadVariableOpReadVariableOp(dense_319_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_319/MatMulMatMuldense_318/Relu:activations:0'dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_319/BiasAdd/ReadVariableOpReadVariableOp)dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_319/BiasAddBiasAdddense_319/MatMul:product:0(dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_319/ReluReludense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_320/MatMul/ReadVariableOpReadVariableOp(dense_320_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_320/MatMulMatMuldense_319/Relu:activations:0'dense_320/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_320/BiasAdd/ReadVariableOpReadVariableOp)dense_320_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_320/BiasAddBiasAdddense_320/MatMul:product:0(dense_320/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_320/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp!^dense_314/BiasAdd/ReadVariableOp ^dense_314/MatMul/ReadVariableOp!^dense_315/BiasAdd/ReadVariableOp ^dense_315/MatMul/ReadVariableOp!^dense_316/BiasAdd/ReadVariableOp ^dense_316/MatMul/ReadVariableOp!^dense_317/BiasAdd/ReadVariableOp ^dense_317/MatMul/ReadVariableOp!^dense_318/BiasAdd/ReadVariableOp ^dense_318/MatMul/ReadVariableOp!^dense_319/BiasAdd/ReadVariableOp ^dense_319/MatMul/ReadVariableOp!^dense_320/BiasAdd/ReadVariableOp ^dense_320/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2D
 dense_314/BiasAdd/ReadVariableOp dense_314/BiasAdd/ReadVariableOp2B
dense_314/MatMul/ReadVariableOpdense_314/MatMul/ReadVariableOp2D
 dense_315/BiasAdd/ReadVariableOp dense_315/BiasAdd/ReadVariableOp2B
dense_315/MatMul/ReadVariableOpdense_315/MatMul/ReadVariableOp2D
 dense_316/BiasAdd/ReadVariableOp dense_316/BiasAdd/ReadVariableOp2B
dense_316/MatMul/ReadVariableOpdense_316/MatMul/ReadVariableOp2D
 dense_317/BiasAdd/ReadVariableOp dense_317/BiasAdd/ReadVariableOp2B
dense_317/MatMul/ReadVariableOpdense_317/MatMul/ReadVariableOp2D
 dense_318/BiasAdd/ReadVariableOp dense_318/BiasAdd/ReadVariableOp2B
dense_318/MatMul/ReadVariableOpdense_318/MatMul/ReadVariableOp2D
 dense_319/BiasAdd/ReadVariableOp dense_319/BiasAdd/ReadVariableOp2B
dense_319/MatMul/ReadVariableOpdense_319/MatMul/ReadVariableOp2D
 dense_320/BiasAdd/ReadVariableOp dense_320/BiasAdd/ReadVariableOp2B
dense_320/MatMul/ReadVariableOpdense_320/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_320_layer_call_fn_1527525

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


/__inference_sequential_68_layer_call_fn_1527067
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_sequential_68_layer_call_and_return_conditional_losses_1526995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:


÷
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_316_layer_call_fn_1527445

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_317_layer_call_and_return_conditional_losses_1527476

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

%__inference_signature_wrapper_1527204
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_1526672o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
Æ

+__inference_dense_318_layer_call_fn_1527485

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_319_layer_call_fn_1527505

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹C
²
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527396

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_314_matmul_readvariableop_resource:7
)dense_314_biasadd_readvariableop_resource::
(dense_315_matmul_readvariableop_resource:7
)dense_315_biasadd_readvariableop_resource::
(dense_316_matmul_readvariableop_resource:7
)dense_316_biasadd_readvariableop_resource::
(dense_317_matmul_readvariableop_resource:7
)dense_317_biasadd_readvariableop_resource::
(dense_318_matmul_readvariableop_resource:7
)dense_318_biasadd_readvariableop_resource::
(dense_319_matmul_readvariableop_resource:7
)dense_319_biasadd_readvariableop_resource::
(dense_320_matmul_readvariableop_resource:7
)dense_320_biasadd_readvariableop_resource:
identity¢ dense_314/BiasAdd/ReadVariableOp¢dense_314/MatMul/ReadVariableOp¢ dense_315/BiasAdd/ReadVariableOp¢dense_315/MatMul/ReadVariableOp¢ dense_316/BiasAdd/ReadVariableOp¢dense_316/MatMul/ReadVariableOp¢ dense_317/BiasAdd/ReadVariableOp¢dense_317/MatMul/ReadVariableOp¢ dense_318/BiasAdd/ReadVariableOp¢dense_318/MatMul/ReadVariableOp¢ dense_319/BiasAdd/ReadVariableOp¢dense_319/MatMul/ReadVariableOp¢ dense_320/BiasAdd/ReadVariableOp¢dense_320/MatMul/ReadVariableOpg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_314/MatMul/ReadVariableOpReadVariableOp(dense_314_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_314/MatMulMatMulnormalization/truediv:z:0'dense_314/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_314/BiasAdd/ReadVariableOpReadVariableOp)dense_314_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_314/BiasAddBiasAdddense_314/MatMul:product:0(dense_314/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_314/ReluReludense_314/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_315/MatMul/ReadVariableOpReadVariableOp(dense_315_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_315/MatMulMatMuldense_314/Relu:activations:0'dense_315/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_315/BiasAdd/ReadVariableOpReadVariableOp)dense_315_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_315/BiasAddBiasAdddense_315/MatMul:product:0(dense_315/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_315/ReluReludense_315/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_316/MatMul/ReadVariableOpReadVariableOp(dense_316_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_316/MatMulMatMuldense_315/Relu:activations:0'dense_316/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_316/BiasAdd/ReadVariableOpReadVariableOp)dense_316_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_316/BiasAddBiasAdddense_316/MatMul:product:0(dense_316/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_316/ReluReludense_316/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_317/MatMul/ReadVariableOpReadVariableOp(dense_317_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_317/MatMulMatMuldense_316/Relu:activations:0'dense_317/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_317/BiasAdd/ReadVariableOpReadVariableOp)dense_317_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_317/BiasAddBiasAdddense_317/MatMul:product:0(dense_317/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_317/ReluReludense_317/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_318/MatMul/ReadVariableOpReadVariableOp(dense_318_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_318/MatMulMatMuldense_317/Relu:activations:0'dense_318/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_318/BiasAdd/ReadVariableOpReadVariableOp)dense_318_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_318/BiasAddBiasAdddense_318/MatMul:product:0(dense_318/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_318/ReluReludense_318/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_319/MatMul/ReadVariableOpReadVariableOp(dense_319_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_319/MatMulMatMuldense_318/Relu:activations:0'dense_319/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_319/BiasAdd/ReadVariableOpReadVariableOp)dense_319_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_319/BiasAddBiasAdddense_319/MatMul:product:0(dense_319/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_319/ReluReludense_319/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_320/MatMul/ReadVariableOpReadVariableOp(dense_320_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_320/MatMulMatMuldense_319/Relu:activations:0'dense_320/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_320/BiasAdd/ReadVariableOpReadVariableOp)dense_320_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_320/BiasAddBiasAdddense_320/MatMul:product:0(dense_320/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_320/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp!^dense_314/BiasAdd/ReadVariableOp ^dense_314/MatMul/ReadVariableOp!^dense_315/BiasAdd/ReadVariableOp ^dense_315/MatMul/ReadVariableOp!^dense_316/BiasAdd/ReadVariableOp ^dense_316/MatMul/ReadVariableOp!^dense_317/BiasAdd/ReadVariableOp ^dense_317/MatMul/ReadVariableOp!^dense_318/BiasAdd/ReadVariableOp ^dense_318/MatMul/ReadVariableOp!^dense_319/BiasAdd/ReadVariableOp ^dense_319/MatMul/ReadVariableOp!^dense_320/BiasAdd/ReadVariableOp ^dense_320/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2D
 dense_314/BiasAdd/ReadVariableOp dense_314/BiasAdd/ReadVariableOp2B
dense_314/MatMul/ReadVariableOpdense_314/MatMul/ReadVariableOp2D
 dense_315/BiasAdd/ReadVariableOp dense_315/BiasAdd/ReadVariableOp2B
dense_315/MatMul/ReadVariableOpdense_315/MatMul/ReadVariableOp2D
 dense_316/BiasAdd/ReadVariableOp dense_316/BiasAdd/ReadVariableOp2B
dense_316/MatMul/ReadVariableOpdense_316/MatMul/ReadVariableOp2D
 dense_317/BiasAdd/ReadVariableOp dense_317/BiasAdd/ReadVariableOp2B
dense_317/MatMul/ReadVariableOpdense_317/MatMul/ReadVariableOp2D
 dense_318/BiasAdd/ReadVariableOp dense_318/BiasAdd/ReadVariableOp2B
dense_318/MatMul/ReadVariableOpdense_318/MatMul/ReadVariableOp2D
 dense_319/BiasAdd/ReadVariableOp dense_319/BiasAdd/ReadVariableOp2B
dense_319/MatMul/ReadVariableOpdense_319/MatMul/ReadVariableOp2D
 dense_320/BiasAdd/ReadVariableOp dense_320/BiasAdd/ReadVariableOp2B
dense_320/MatMul/ReadVariableOpdense_320/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:


÷
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_315_layer_call_and_return_conditional_losses_1527436

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_316_layer_call_and_return_conditional_losses_1527456

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_320_layer_call_and_return_conditional_losses_1527535

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤,

J__inference_sequential_68_layer_call_and_return_conditional_losses_1527113
normalization_input
normalization_sub_y
normalization_sqrt_x#
dense_314_1527077:
dense_314_1527079:#
dense_315_1527082:
dense_315_1527084:#
dense_316_1527087:
dense_316_1527089:#
dense_317_1527092:
dense_317_1527094:#
dense_318_1527097:
dense_318_1527099:#
dense_319_1527102:
dense_319_1527104:#
dense_320_1527107:
dense_320_1527109:
identity¢!dense_314/StatefulPartitionedCall¢!dense_315/StatefulPartitionedCall¢!dense_316/StatefulPartitionedCall¢!dense_317/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢!dense_320/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_314/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_314_1527077dense_314_1527079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697
!dense_315/StatefulPartitionedCallStatefulPartitionedCall*dense_314/StatefulPartitionedCall:output:0dense_315_1527082dense_315_1527084*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714
!dense_316/StatefulPartitionedCallStatefulPartitionedCall*dense_315/StatefulPartitionedCall:output:0dense_316_1527087dense_316_1527089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731
!dense_317/StatefulPartitionedCallStatefulPartitionedCall*dense_316/StatefulPartitionedCall:output:0dense_317_1527092dense_317_1527094*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748
!dense_318/StatefulPartitionedCallStatefulPartitionedCall*dense_317/StatefulPartitionedCall:output:0dense_318_1527097dense_318_1527099*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765
!dense_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0dense_319_1527102dense_319_1527104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782
!dense_320/StatefulPartitionedCallStatefulPartitionedCall*dense_319/StatefulPartitionedCall:output:0dense_320_1527107dense_320_1527109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798y
IdentityIdentity*dense_320/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_314/StatefulPartitionedCall"^dense_315/StatefulPartitionedCall"^dense_316/StatefulPartitionedCall"^dense_317/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall"^dense_320/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_314/StatefulPartitionedCall!dense_314/StatefulPartitionedCall2F
!dense_315/StatefulPartitionedCall!dense_315/StatefulPartitionedCall2F
!dense_316/StatefulPartitionedCall!dense_316/StatefulPartitionedCall2F
!dense_317/StatefulPartitionedCall!dense_317/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2F
!dense_320/StatefulPartitionedCall!dense_320/StatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
¤,

J__inference_sequential_68_layer_call_and_return_conditional_losses_1527159
normalization_input
normalization_sub_y
normalization_sqrt_x#
dense_314_1527123:
dense_314_1527125:#
dense_315_1527128:
dense_315_1527130:#
dense_316_1527133:
dense_316_1527135:#
dense_317_1527138:
dense_317_1527140:#
dense_318_1527143:
dense_318_1527145:#
dense_319_1527148:
dense_319_1527150:#
dense_320_1527153:
dense_320_1527155:
identity¢!dense_314/StatefulPartitionedCall¢!dense_315/StatefulPartitionedCall¢!dense_316/StatefulPartitionedCall¢!dense_317/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢!dense_320/StatefulPartitionedCallt
normalization/subSubnormalization_inputnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_314/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_314_1527123dense_314_1527125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697
!dense_315/StatefulPartitionedCallStatefulPartitionedCall*dense_314/StatefulPartitionedCall:output:0dense_315_1527128dense_315_1527130*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714
!dense_316/StatefulPartitionedCallStatefulPartitionedCall*dense_315/StatefulPartitionedCall:output:0dense_316_1527133dense_316_1527135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731
!dense_317/StatefulPartitionedCallStatefulPartitionedCall*dense_316/StatefulPartitionedCall:output:0dense_317_1527138dense_317_1527140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748
!dense_318/StatefulPartitionedCallStatefulPartitionedCall*dense_317/StatefulPartitionedCall:output:0dense_318_1527143dense_318_1527145*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765
!dense_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0dense_319_1527148dense_319_1527150*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782
!dense_320/StatefulPartitionedCallStatefulPartitionedCall*dense_319/StatefulPartitionedCall:output:0dense_320_1527153dense_320_1527155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798y
IdentityIdentity*dense_320/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_314/StatefulPartitionedCall"^dense_315/StatefulPartitionedCall"^dense_316/StatefulPartitionedCall"^dense_317/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall"^dense_320/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_314/StatefulPartitionedCall!dense_314/StatefulPartitionedCall2F
!dense_315/StatefulPartitionedCall!dense_315/StatefulPartitionedCall2F
!dense_316/StatefulPartitionedCall!dense_316/StatefulPartitionedCall2F
!dense_317/StatefulPartitionedCall!dense_317/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2F
!dense_320/StatefulPartitionedCall!dense_320/StatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
åÎ
À
#__inference__traced_restore_1527882
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_1:	 5
#assignvariableop_3_dense_314_kernel:/
!assignvariableop_4_dense_314_bias:5
#assignvariableop_5_dense_315_kernel:/
!assignvariableop_6_dense_315_bias:5
#assignvariableop_7_dense_316_kernel:/
!assignvariableop_8_dense_316_bias:5
#assignvariableop_9_dense_317_kernel:0
"assignvariableop_10_dense_317_bias:6
$assignvariableop_11_dense_318_kernel:0
"assignvariableop_12_dense_318_bias:6
$assignvariableop_13_dense_319_kernel:0
"assignvariableop_14_dense_319_bias:6
$assignvariableop_15_dense_320_kernel:0
"assignvariableop_16_dense_320_bias:'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: 0
&assignvariableop_21_adam_learning_rate: #
assignvariableop_22_total: #
assignvariableop_23_count: =
+assignvariableop_24_adam_dense_314_kernel_m:7
)assignvariableop_25_adam_dense_314_bias_m:=
+assignvariableop_26_adam_dense_315_kernel_m:7
)assignvariableop_27_adam_dense_315_bias_m:=
+assignvariableop_28_adam_dense_316_kernel_m:7
)assignvariableop_29_adam_dense_316_bias_m:=
+assignvariableop_30_adam_dense_317_kernel_m:7
)assignvariableop_31_adam_dense_317_bias_m:=
+assignvariableop_32_adam_dense_318_kernel_m:7
)assignvariableop_33_adam_dense_318_bias_m:=
+assignvariableop_34_adam_dense_319_kernel_m:7
)assignvariableop_35_adam_dense_319_bias_m:=
+assignvariableop_36_adam_dense_320_kernel_m:7
)assignvariableop_37_adam_dense_320_bias_m:=
+assignvariableop_38_adam_dense_314_kernel_v:7
)assignvariableop_39_adam_dense_314_bias_v:=
+assignvariableop_40_adam_dense_315_kernel_v:7
)assignvariableop_41_adam_dense_315_bias_v:=
+assignvariableop_42_adam_dense_316_kernel_v:7
)assignvariableop_43_adam_dense_316_bias_v:=
+assignvariableop_44_adam_dense_317_kernel_v:7
)assignvariableop_45_adam_dense_317_bias_v:=
+assignvariableop_46_adam_dense_318_kernel_v:7
)assignvariableop_47_adam_dense_318_bias_v:=
+assignvariableop_48_adam_dense_319_kernel_v:7
)assignvariableop_49_adam_dense_319_bias_v:=
+assignvariableop_50_adam_dense_320_kernel_v:7
)assignvariableop_51_adam_dense_320_bias_v:
identity_53¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*Á
value·B´5B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÚ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ª
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ê
_output_shapes×
Ô:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_count_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_314_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_314_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_315_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_315_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_316_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_316_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_317_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_317_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_318_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_318_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_319_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_319_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_320_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_320_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_iterIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dense_314_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_314_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_315_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_315_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_316_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_316_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_317_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_317_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_318_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_318_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_dense_319_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_319_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_dense_320_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_320_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_dense_314_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_314_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_315_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_315_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_dense_316_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_316_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense_317_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_317_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_dense_318_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_318_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dense_319_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_319_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_dense_320_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_320_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ç	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: ´	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_53Identity_53:output:0*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ý+

J__inference_sequential_68_layer_call_and_return_conditional_losses_1526805

inputs
normalization_sub_y
normalization_sqrt_x#
dense_314_1526698:
dense_314_1526700:#
dense_315_1526715:
dense_315_1526717:#
dense_316_1526732:
dense_316_1526734:#
dense_317_1526749:
dense_317_1526751:#
dense_318_1526766:
dense_318_1526768:#
dense_319_1526783:
dense_319_1526785:#
dense_320_1526799:
dense_320_1526801:
identity¢!dense_314/StatefulPartitionedCall¢!dense_315/StatefulPartitionedCall¢!dense_316/StatefulPartitionedCall¢!dense_317/StatefulPartitionedCall¢!dense_318/StatefulPartitionedCall¢!dense_319/StatefulPartitionedCall¢!dense_320/StatefulPartitionedCallg
normalization/subSubinputsnormalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
normalization/SqrtSqrtnormalization_sqrt_x*
T0*
_output_shapes

:\
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!dense_314/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_314_1526698dense_314_1526700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_314_layer_call_and_return_conditional_losses_1526697
!dense_315/StatefulPartitionedCallStatefulPartitionedCall*dense_314/StatefulPartitionedCall:output:0dense_315_1526715dense_315_1526717*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_315_layer_call_and_return_conditional_losses_1526714
!dense_316/StatefulPartitionedCallStatefulPartitionedCall*dense_315/StatefulPartitionedCall:output:0dense_316_1526732dense_316_1526734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_316_layer_call_and_return_conditional_losses_1526731
!dense_317/StatefulPartitionedCallStatefulPartitionedCall*dense_316/StatefulPartitionedCall:output:0dense_317_1526749dense_317_1526751*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_317_layer_call_and_return_conditional_losses_1526748
!dense_318/StatefulPartitionedCallStatefulPartitionedCall*dense_317/StatefulPartitionedCall:output:0dense_318_1526766dense_318_1526768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_318_layer_call_and_return_conditional_losses_1526765
!dense_319/StatefulPartitionedCallStatefulPartitionedCall*dense_318/StatefulPartitionedCall:output:0dense_319_1526783dense_319_1526785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_319_layer_call_and_return_conditional_losses_1526782
!dense_320/StatefulPartitionedCallStatefulPartitionedCall*dense_319/StatefulPartitionedCall:output:0dense_320_1526799dense_320_1526801*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_320_layer_call_and_return_conditional_losses_1526798y
IdentityIdentity*dense_320/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_314/StatefulPartitionedCall"^dense_315/StatefulPartitionedCall"^dense_316/StatefulPartitionedCall"^dense_317/StatefulPartitionedCall"^dense_318/StatefulPartitionedCall"^dense_319/StatefulPartitionedCall"^dense_320/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_314/StatefulPartitionedCall!dense_314/StatefulPartitionedCall2F
!dense_315/StatefulPartitionedCall!dense_315/StatefulPartitionedCall2F
!dense_316/StatefulPartitionedCall!dense_316/StatefulPartitionedCall2F
!dense_317/StatefulPartitionedCall!dense_317/StatefulPartitionedCall2F
!dense_318/StatefulPartitionedCall!dense_318/StatefulPartitionedCall2F
!dense_319/StatefulPartitionedCall!dense_319/StatefulPartitionedCall2F
!dense_320/StatefulPartitionedCall!dense_320/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
g
µ
 __inference__traced_save_1527716
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop&
"savev2_count_1_read_readvariableop	/
+savev2_dense_314_kernel_read_readvariableop-
)savev2_dense_314_bias_read_readvariableop/
+savev2_dense_315_kernel_read_readvariableop-
)savev2_dense_315_bias_read_readvariableop/
+savev2_dense_316_kernel_read_readvariableop-
)savev2_dense_316_bias_read_readvariableop/
+savev2_dense_317_kernel_read_readvariableop-
)savev2_dense_317_bias_read_readvariableop/
+savev2_dense_318_kernel_read_readvariableop-
)savev2_dense_318_bias_read_readvariableop/
+savev2_dense_319_kernel_read_readvariableop-
)savev2_dense_319_bias_read_readvariableop/
+savev2_dense_320_kernel_read_readvariableop-
)savev2_dense_320_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_314_kernel_m_read_readvariableop4
0savev2_adam_dense_314_bias_m_read_readvariableop6
2savev2_adam_dense_315_kernel_m_read_readvariableop4
0savev2_adam_dense_315_bias_m_read_readvariableop6
2savev2_adam_dense_316_kernel_m_read_readvariableop4
0savev2_adam_dense_316_bias_m_read_readvariableop6
2savev2_adam_dense_317_kernel_m_read_readvariableop4
0savev2_adam_dense_317_bias_m_read_readvariableop6
2savev2_adam_dense_318_kernel_m_read_readvariableop4
0savev2_adam_dense_318_bias_m_read_readvariableop6
2savev2_adam_dense_319_kernel_m_read_readvariableop4
0savev2_adam_dense_319_bias_m_read_readvariableop6
2savev2_adam_dense_320_kernel_m_read_readvariableop4
0savev2_adam_dense_320_bias_m_read_readvariableop6
2savev2_adam_dense_314_kernel_v_read_readvariableop4
0savev2_adam_dense_314_bias_v_read_readvariableop6
2savev2_adam_dense_315_kernel_v_read_readvariableop4
0savev2_adam_dense_315_bias_v_read_readvariableop6
2savev2_adam_dense_316_kernel_v_read_readvariableop4
0savev2_adam_dense_316_bias_v_read_readvariableop6
2savev2_adam_dense_317_kernel_v_read_readvariableop4
0savev2_adam_dense_317_bias_v_read_readvariableop6
2savev2_adam_dense_318_kernel_v_read_readvariableop4
0savev2_adam_dense_318_bias_v_read_readvariableop6
2savev2_adam_dense_319_kernel_v_read_readvariableop4
0savev2_adam_dense_319_bias_v_read_readvariableop6
2savev2_adam_dense_320_kernel_v_read_readvariableop4
0savev2_adam_dense_320_bias_v_read_readvariableop
savev2_const_2

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*Á
value·B´5B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH×
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ü
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop"savev2_count_1_read_readvariableop+savev2_dense_314_kernel_read_readvariableop)savev2_dense_314_bias_read_readvariableop+savev2_dense_315_kernel_read_readvariableop)savev2_dense_315_bias_read_readvariableop+savev2_dense_316_kernel_read_readvariableop)savev2_dense_316_bias_read_readvariableop+savev2_dense_317_kernel_read_readvariableop)savev2_dense_317_bias_read_readvariableop+savev2_dense_318_kernel_read_readvariableop)savev2_dense_318_bias_read_readvariableop+savev2_dense_319_kernel_read_readvariableop)savev2_dense_319_bias_read_readvariableop+savev2_dense_320_kernel_read_readvariableop)savev2_dense_320_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_314_kernel_m_read_readvariableop0savev2_adam_dense_314_bias_m_read_readvariableop2savev2_adam_dense_315_kernel_m_read_readvariableop0savev2_adam_dense_315_bias_m_read_readvariableop2savev2_adam_dense_316_kernel_m_read_readvariableop0savev2_adam_dense_316_bias_m_read_readvariableop2savev2_adam_dense_317_kernel_m_read_readvariableop0savev2_adam_dense_317_bias_m_read_readvariableop2savev2_adam_dense_318_kernel_m_read_readvariableop0savev2_adam_dense_318_bias_m_read_readvariableop2savev2_adam_dense_319_kernel_m_read_readvariableop0savev2_adam_dense_319_bias_m_read_readvariableop2savev2_adam_dense_320_kernel_m_read_readvariableop0savev2_adam_dense_320_bias_m_read_readvariableop2savev2_adam_dense_314_kernel_v_read_readvariableop0savev2_adam_dense_314_bias_v_read_readvariableop2savev2_adam_dense_315_kernel_v_read_readvariableop0savev2_adam_dense_315_bias_v_read_readvariableop2savev2_adam_dense_316_kernel_v_read_readvariableop0savev2_adam_dense_316_bias_v_read_readvariableop2savev2_adam_dense_317_kernel_v_read_readvariableop0savev2_adam_dense_317_bias_v_read_readvariableop2savev2_adam_dense_318_kernel_v_read_readvariableop0savev2_adam_dense_318_bias_v_read_readvariableop2savev2_adam_dense_319_kernel_v_read_readvariableop0savev2_adam_dense_319_bias_v_read_readvariableop2savev2_adam_dense_320_kernel_v_read_readvariableop0savev2_adam_dense_320_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *C
dtypes9
725		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesó
ð: ::: ::::::::::::::: : : : : : : ::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:: .

_output_shapes
::$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:: 2

_output_shapes
::$3 

_output_shapes

:: 4

_output_shapes
::5

_output_shapes
: "¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
\
normalization_inputE
%serving_default_normalization_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ=
	dense_3200
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ä

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ó
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
»
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses

)kernel
*bias"
_tf_keras_layer
»
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses

1kernel
2bias"
_tf_keras_layer
»
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
»
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias"
_tf_keras_layer
»
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses

Ikernel
Jbias"
_tf_keras_layer
»
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer

0
1
2
!3
"4
)5
*6
17
28
99
:10
A11
B12
I13
J14
Q15
R16"
trackable_list_wrapper

!0
"1
)2
*3
14
25
96
:7
A8
B9
I10
J11
Q12
R13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32
/__inference_sequential_68_layer_call_fn_1526840
/__inference_sequential_68_layer_call_fn_1527241
/__inference_sequential_68_layer_call_fn_1527278
/__inference_sequential_68_layer_call_fn_1527067À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
Þ
\trace_0
]trace_1
^trace_2
_trace_32ó
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527337
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527396
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527113
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527159À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z\trace_0z]trace_1z^trace_2z_trace_3
ÙBÖ
"__inference__wrapped_model_1526672normalization_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ë
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rate!m"m)m*m 1m¡2m¢9m£:m¤Am¥Bm¦Im§Jm¨Qm©Rmª!v«"v¬)v­*v®1v¯2v°9v±:v²Av³Bv´IvµJv¶Qv·Rv¸"
	optimizer
,
eserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
Ø
ftrace_02»
__inference_adapt_step_21695
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zftrace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
ï
ltrace_02Ò
+__inference_dense_314_layer_call_fn_1527405¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zltrace_0

mtrace_02í
F__inference_dense_314_layer_call_and_return_conditional_losses_1527416¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zmtrace_0
": 2dense_314/kernel
:2dense_314/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ï
strace_02Ò
+__inference_dense_315_layer_call_fn_1527425¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zstrace_0

ttrace_02í
F__inference_dense_315_layer_call_and_return_conditional_losses_1527436¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zttrace_0
": 2dense_315/kernel
:2dense_315/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
­
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
ï
ztrace_02Ò
+__inference_dense_316_layer_call_fn_1527445¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zztrace_0

{trace_02í
F__inference_dense_316_layer_call_and_return_conditional_losses_1527456¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z{trace_0
": 2dense_316/kernel
:2dense_316/bias
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
®
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_317_layer_call_fn_1527465¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_317_layer_call_and_return_conditional_losses_1527476¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": 2dense_317/kernel
:2dense_317/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_318_layer_call_fn_1527485¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_318_layer_call_and_return_conditional_losses_1527496¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": 2dense_318/kernel
:2dense_318/bias
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_319_layer_call_fn_1527505¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_319_layer_call_and_return_conditional_losses_1527516¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": 2dense_319/kernel
:2dense_319/bias
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
ñ
trace_02Ò
+__inference_dense_320_layer_call_fn_1527525¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02í
F__inference_dense_320_layer_call_and_return_conditional_losses_1527535¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": 2dense_320/kernel
:2dense_320/bias
5
0
1
2"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
/__inference_sequential_68_layer_call_fn_1526840normalization_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
/__inference_sequential_68_layer_call_fn_1527241inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Bþ
/__inference_sequential_68_layer_call_fn_1527278inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
/__inference_sequential_68_layer_call_fn_1527067normalization_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527337inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527396inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
©B¦
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527113normalization_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
©B¦
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527159normalization_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ØBÕ
%__inference_signature_wrapper_1527204normalization_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÊBÇ
__inference_adapt_step_21695iterator"
²
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_314_layer_call_fn_1527405inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_314_layer_call_and_return_conditional_losses_1527416inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_315_layer_call_fn_1527425inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_315_layer_call_and_return_conditional_losses_1527436inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_316_layer_call_fn_1527445inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_316_layer_call_and_return_conditional_losses_1527456inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_317_layer_call_fn_1527465inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_317_layer_call_and_return_conditional_losses_1527476inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_318_layer_call_fn_1527485inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_318_layer_call_and_return_conditional_losses_1527496inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_319_layer_call_fn_1527505inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_319_layer_call_and_return_conditional_losses_1527516inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_dense_320_layer_call_fn_1527525inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_dense_320_layer_call_and_return_conditional_losses_1527535inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
':%2Adam/dense_314/kernel/m
!:2Adam/dense_314/bias/m
':%2Adam/dense_315/kernel/m
!:2Adam/dense_315/bias/m
':%2Adam/dense_316/kernel/m
!:2Adam/dense_316/bias/m
':%2Adam/dense_317/kernel/m
!:2Adam/dense_317/bias/m
':%2Adam/dense_318/kernel/m
!:2Adam/dense_318/bias/m
':%2Adam/dense_319/kernel/m
!:2Adam/dense_319/bias/m
':%2Adam/dense_320/kernel/m
!:2Adam/dense_320/bias/m
':%2Adam/dense_314/kernel/v
!:2Adam/dense_314/bias/v
':%2Adam/dense_315/kernel/v
!:2Adam/dense_315/bias/v
':%2Adam/dense_316/kernel/v
!:2Adam/dense_316/bias/v
':%2Adam/dense_317/kernel/v
!:2Adam/dense_317/bias/v
':%2Adam/dense_318/kernel/v
!:2Adam/dense_318/bias/v
':%2Adam/dense_319/kernel/v
!:2Adam/dense_319/bias/v
':%2Adam/dense_320/kernel/v
!:2Adam/dense_320/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant¹
"__inference__wrapped_model_1526672¹º!")*129:ABIJQRE¢B
;¢8
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_320# 
	dense_320ÿÿÿÿÿÿÿÿÿe
__inference_adapt_step_21695E:¢7
0¢-
+(¢
 IteratorSpec 
ª "
 ¦
F__inference_dense_314_layer_call_and_return_conditional_losses_1527416\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_314_layer_call_fn_1527405O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_315_layer_call_and_return_conditional_losses_1527436\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_315_layer_call_fn_1527425O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_316_layer_call_and_return_conditional_losses_1527456\12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_316_layer_call_fn_1527445O12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_317_layer_call_and_return_conditional_losses_1527476\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_317_layer_call_fn_1527465O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_318_layer_call_and_return_conditional_losses_1527496\AB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_318_layer_call_fn_1527485OAB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_319_layer_call_and_return_conditional_losses_1527516\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_319_layer_call_fn_1527505OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_320_layer_call_and_return_conditional_losses_1527535\QR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_320_layer_call_fn_1527525OQR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÙ
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527113¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ù
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527159¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527337}¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ë
J__inference_sequential_68_layer_call_and_return_conditional_losses_1527396}¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 °
/__inference_sequential_68_layer_call_fn_1526840}¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
/__inference_sequential_68_layer_call_fn_1527067}¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
/__inference_sequential_68_layer_call_fn_1527241p¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
/__inference_sequential_68_layer_call_fn_1527278p¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÓ
%__inference_signature_wrapper_1527204©¹º!")*129:ABIJQR\¢Y
¢ 
RªO
M
normalization_input63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_320# 
	dense_320ÿÿÿÿÿÿÿÿÿ