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
Adam/dense_341/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_341/bias/v
{
)Adam/dense_341/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_341/bias/v*
_output_shapes
:*
dtype0

Adam/dense_341/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_341/kernel/v

+Adam/dense_341/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_341/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_340/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_340/bias/v
{
)Adam/dense_340/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_340/bias/v*
_output_shapes
:*
dtype0

Adam/dense_340/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_340/kernel/v

+Adam/dense_340/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_340/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_339/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_339/bias/v
{
)Adam/dense_339/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_339/bias/v*
_output_shapes
:*
dtype0

Adam/dense_339/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_339/kernel/v

+Adam/dense_339/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_339/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_338/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_338/bias/v
{
)Adam/dense_338/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_338/bias/v*
_output_shapes
:*
dtype0

Adam/dense_338/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_338/kernel/v

+Adam/dense_338/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_338/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_337/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_337/bias/v
{
)Adam/dense_337/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_337/bias/v*
_output_shapes
:*
dtype0

Adam/dense_337/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_337/kernel/v

+Adam/dense_337/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_337/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_336/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_336/bias/v
{
)Adam/dense_336/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_336/bias/v*
_output_shapes
:*
dtype0

Adam/dense_336/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_336/kernel/v

+Adam/dense_336/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_336/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_335/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_335/bias/v
{
)Adam/dense_335/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_335/bias/v*
_output_shapes
:*
dtype0

Adam/dense_335/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_335/kernel/v

+Adam/dense_335/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_335/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_341/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_341/bias/m
{
)Adam/dense_341/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_341/bias/m*
_output_shapes
:*
dtype0

Adam/dense_341/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_341/kernel/m

+Adam/dense_341/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_341/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_340/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_340/bias/m
{
)Adam/dense_340/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_340/bias/m*
_output_shapes
:*
dtype0

Adam/dense_340/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_340/kernel/m

+Adam/dense_340/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_340/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_339/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_339/bias/m
{
)Adam/dense_339/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_339/bias/m*
_output_shapes
:*
dtype0

Adam/dense_339/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_339/kernel/m

+Adam/dense_339/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_339/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_338/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_338/bias/m
{
)Adam/dense_338/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_338/bias/m*
_output_shapes
:*
dtype0

Adam/dense_338/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_338/kernel/m

+Adam/dense_338/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_338/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_337/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_337/bias/m
{
)Adam/dense_337/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_337/bias/m*
_output_shapes
:*
dtype0

Adam/dense_337/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_337/kernel/m

+Adam/dense_337/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_337/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_336/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_336/bias/m
{
)Adam/dense_336/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_336/bias/m*
_output_shapes
:*
dtype0

Adam/dense_336/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_336/kernel/m

+Adam/dense_336/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_336/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_335/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_335/bias/m
{
)Adam/dense_335/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_335/bias/m*
_output_shapes
:*
dtype0

Adam/dense_335/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_335/kernel/m

+Adam/dense_335/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_335/kernel/m*
_output_shapes

:*
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
dense_341/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_341/bias
m
"dense_341/bias/Read/ReadVariableOpReadVariableOpdense_341/bias*
_output_shapes
:*
dtype0
|
dense_341/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_341/kernel
u
$dense_341/kernel/Read/ReadVariableOpReadVariableOpdense_341/kernel*
_output_shapes

:*
dtype0
t
dense_340/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_340/bias
m
"dense_340/bias/Read/ReadVariableOpReadVariableOpdense_340/bias*
_output_shapes
:*
dtype0
|
dense_340/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_340/kernel
u
$dense_340/kernel/Read/ReadVariableOpReadVariableOpdense_340/kernel*
_output_shapes

:*
dtype0
t
dense_339/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_339/bias
m
"dense_339/bias/Read/ReadVariableOpReadVariableOpdense_339/bias*
_output_shapes
:*
dtype0
|
dense_339/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_339/kernel
u
$dense_339/kernel/Read/ReadVariableOpReadVariableOpdense_339/kernel*
_output_shapes

:*
dtype0
t
dense_338/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_338/bias
m
"dense_338/bias/Read/ReadVariableOpReadVariableOpdense_338/bias*
_output_shapes
:*
dtype0
|
dense_338/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_338/kernel
u
$dense_338/kernel/Read/ReadVariableOpReadVariableOpdense_338/kernel*
_output_shapes

:*
dtype0
t
dense_337/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_337/bias
m
"dense_337/bias/Read/ReadVariableOpReadVariableOpdense_337/bias*
_output_shapes
:*
dtype0
|
dense_337/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_337/kernel
u
$dense_337/kernel/Read/ReadVariableOpReadVariableOpdense_337/kernel*
_output_shapes

:*
dtype0
t
dense_336/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_336/bias
m
"dense_336/bias/Read/ReadVariableOpReadVariableOpdense_336/bias*
_output_shapes
:*
dtype0
|
dense_336/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_336/kernel
u
$dense_336/kernel/Read/ReadVariableOpReadVariableOpdense_336/kernel*
_output_shapes

:*
dtype0
t
dense_335/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_335/bias
m
"dense_335/bias/Read/ReadVariableOpReadVariableOpdense_335/bias*
_output_shapes
:*
dtype0
|
dense_335/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_335/kernel
u
$dense_335/kernel/Read/ReadVariableOpReadVariableOpdense_335/kernel*
_output_shapes

:*
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
VARIABLE_VALUEdense_335/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_335/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_336/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_336/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_337/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_337/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_338/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_338/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_339/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_339/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_340/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_340/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_341/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_341/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_335/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_335/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_336/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_336/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_337/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_337/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_338/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_338/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_339/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_339/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_340/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_340/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_341/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_341/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_335/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_335/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_336/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_336/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_337/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_337/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_338/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_338/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_339/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_339/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_340/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_340/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_341/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_341/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_normalization_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Û
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConstConst_1dense_335/kerneldense_335/biasdense_336/kerneldense_336/biasdense_337/kerneldense_337/biasdense_338/kerneldense_338/biasdense_339/kerneldense_339/biasdense_340/kerneldense_340/biasdense_341/kerneldense_341/bias*
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
%__inference_signature_wrapper_1595568
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount_1/Read/ReadVariableOp$dense_335/kernel/Read/ReadVariableOp"dense_335/bias/Read/ReadVariableOp$dense_336/kernel/Read/ReadVariableOp"dense_336/bias/Read/ReadVariableOp$dense_337/kernel/Read/ReadVariableOp"dense_337/bias/Read/ReadVariableOp$dense_338/kernel/Read/ReadVariableOp"dense_338/bias/Read/ReadVariableOp$dense_339/kernel/Read/ReadVariableOp"dense_339/bias/Read/ReadVariableOp$dense_340/kernel/Read/ReadVariableOp"dense_340/bias/Read/ReadVariableOp$dense_341/kernel/Read/ReadVariableOp"dense_341/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_335/kernel/m/Read/ReadVariableOp)Adam/dense_335/bias/m/Read/ReadVariableOp+Adam/dense_336/kernel/m/Read/ReadVariableOp)Adam/dense_336/bias/m/Read/ReadVariableOp+Adam/dense_337/kernel/m/Read/ReadVariableOp)Adam/dense_337/bias/m/Read/ReadVariableOp+Adam/dense_338/kernel/m/Read/ReadVariableOp)Adam/dense_338/bias/m/Read/ReadVariableOp+Adam/dense_339/kernel/m/Read/ReadVariableOp)Adam/dense_339/bias/m/Read/ReadVariableOp+Adam/dense_340/kernel/m/Read/ReadVariableOp)Adam/dense_340/bias/m/Read/ReadVariableOp+Adam/dense_341/kernel/m/Read/ReadVariableOp)Adam/dense_341/bias/m/Read/ReadVariableOp+Adam/dense_335/kernel/v/Read/ReadVariableOp)Adam/dense_335/bias/v/Read/ReadVariableOp+Adam/dense_336/kernel/v/Read/ReadVariableOp)Adam/dense_336/bias/v/Read/ReadVariableOp+Adam/dense_337/kernel/v/Read/ReadVariableOp)Adam/dense_337/bias/v/Read/ReadVariableOp+Adam/dense_338/kernel/v/Read/ReadVariableOp)Adam/dense_338/bias/v/Read/ReadVariableOp+Adam/dense_339/kernel/v/Read/ReadVariableOp)Adam/dense_339/bias/v/Read/ReadVariableOp+Adam/dense_340/kernel/v/Read/ReadVariableOp)Adam/dense_340/bias/v/Read/ReadVariableOp+Adam/dense_341/kernel/v/Read/ReadVariableOp)Adam/dense_341/bias/v/Read/ReadVariableOpConst_2*A
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
 __inference__traced_save_1596080
Å

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_335/kerneldense_335/biasdense_336/kerneldense_336/biasdense_337/kerneldense_337/biasdense_338/kerneldense_338/biasdense_339/kerneldense_339/biasdense_340/kerneldense_340/biasdense_341/kerneldense_341/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_335/kernel/mAdam/dense_335/bias/mAdam/dense_336/kernel/mAdam/dense_336/bias/mAdam/dense_337/kernel/mAdam/dense_337/bias/mAdam/dense_338/kernel/mAdam/dense_338/bias/mAdam/dense_339/kernel/mAdam/dense_339/bias/mAdam/dense_340/kernel/mAdam/dense_340/bias/mAdam/dense_341/kernel/mAdam/dense_341/bias/mAdam/dense_335/kernel/vAdam/dense_335/bias/vAdam/dense_336/kernel/vAdam/dense_336/bias/vAdam/dense_337/kernel/vAdam/dense_337/bias/vAdam/dense_338/kernel/vAdam/dense_338/bias/vAdam/dense_339/kernel/vAdam/dense_339/bias/vAdam/dense_340/kernel/vAdam/dense_340/bias/vAdam/dense_341/kernel/vAdam/dense_341/bias/v*@
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
#__inference__traced_restore_1596246Ýî
É	
÷
F__inference_dense_341_layer_call_and_return_conditional_losses_1595899

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_335_layer_call_and_return_conditional_losses_1595780

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
¹C
²
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595760

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_335_matmul_readvariableop_resource:7
)dense_335_biasadd_readvariableop_resource::
(dense_336_matmul_readvariableop_resource:7
)dense_336_biasadd_readvariableop_resource::
(dense_337_matmul_readvariableop_resource:7
)dense_337_biasadd_readvariableop_resource::
(dense_338_matmul_readvariableop_resource:7
)dense_338_biasadd_readvariableop_resource::
(dense_339_matmul_readvariableop_resource:7
)dense_339_biasadd_readvariableop_resource::
(dense_340_matmul_readvariableop_resource:7
)dense_340_biasadd_readvariableop_resource::
(dense_341_matmul_readvariableop_resource:7
)dense_341_biasadd_readvariableop_resource:
identity¢ dense_335/BiasAdd/ReadVariableOp¢dense_335/MatMul/ReadVariableOp¢ dense_336/BiasAdd/ReadVariableOp¢dense_336/MatMul/ReadVariableOp¢ dense_337/BiasAdd/ReadVariableOp¢dense_337/MatMul/ReadVariableOp¢ dense_338/BiasAdd/ReadVariableOp¢dense_338/MatMul/ReadVariableOp¢ dense_339/BiasAdd/ReadVariableOp¢dense_339/MatMul/ReadVariableOp¢ dense_340/BiasAdd/ReadVariableOp¢dense_340/MatMul/ReadVariableOp¢ dense_341/BiasAdd/ReadVariableOp¢dense_341/MatMul/ReadVariableOpg
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
dense_335/MatMul/ReadVariableOpReadVariableOp(dense_335_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_335/MatMulMatMulnormalization/truediv:z:0'dense_335/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_335/BiasAdd/ReadVariableOpReadVariableOp)dense_335_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_335/BiasAddBiasAdddense_335/MatMul:product:0(dense_335/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_335/ReluReludense_335/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_336/MatMul/ReadVariableOpReadVariableOp(dense_336_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_336/MatMulMatMuldense_335/Relu:activations:0'dense_336/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_336/BiasAdd/ReadVariableOpReadVariableOp)dense_336_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_336/BiasAddBiasAdddense_336/MatMul:product:0(dense_336/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_336/ReluReludense_336/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_337/MatMul/ReadVariableOpReadVariableOp(dense_337_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_337/MatMulMatMuldense_336/Relu:activations:0'dense_337/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_337/BiasAdd/ReadVariableOpReadVariableOp)dense_337_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_337/BiasAddBiasAdddense_337/MatMul:product:0(dense_337/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_337/ReluReludense_337/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_338/MatMul/ReadVariableOpReadVariableOp(dense_338_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_338/MatMulMatMuldense_337/Relu:activations:0'dense_338/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_338/BiasAdd/ReadVariableOpReadVariableOp)dense_338_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_338/BiasAddBiasAdddense_338/MatMul:product:0(dense_338/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_338/ReluReludense_338/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_339/MatMul/ReadVariableOpReadVariableOp(dense_339_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_339/MatMulMatMuldense_338/Relu:activations:0'dense_339/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_339/BiasAdd/ReadVariableOpReadVariableOp)dense_339_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_339/BiasAddBiasAdddense_339/MatMul:product:0(dense_339/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_339/ReluReludense_339/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_340/MatMul/ReadVariableOpReadVariableOp(dense_340_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_340/MatMulMatMuldense_339/Relu:activations:0'dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_340/BiasAdd/ReadVariableOpReadVariableOp)dense_340_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_340/BiasAddBiasAdddense_340/MatMul:product:0(dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_340/ReluReludense_340/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_341/MatMul/ReadVariableOpReadVariableOp(dense_341_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_341/MatMulMatMuldense_340/Relu:activations:0'dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_341/BiasAdd/ReadVariableOpReadVariableOp)dense_341_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_341/BiasAddBiasAdddense_341/MatMul:product:0(dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_341/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp!^dense_335/BiasAdd/ReadVariableOp ^dense_335/MatMul/ReadVariableOp!^dense_336/BiasAdd/ReadVariableOp ^dense_336/MatMul/ReadVariableOp!^dense_337/BiasAdd/ReadVariableOp ^dense_337/MatMul/ReadVariableOp!^dense_338/BiasAdd/ReadVariableOp ^dense_338/MatMul/ReadVariableOp!^dense_339/BiasAdd/ReadVariableOp ^dense_339/MatMul/ReadVariableOp!^dense_340/BiasAdd/ReadVariableOp ^dense_340/MatMul/ReadVariableOp!^dense_341/BiasAdd/ReadVariableOp ^dense_341/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2D
 dense_335/BiasAdd/ReadVariableOp dense_335/BiasAdd/ReadVariableOp2B
dense_335/MatMul/ReadVariableOpdense_335/MatMul/ReadVariableOp2D
 dense_336/BiasAdd/ReadVariableOp dense_336/BiasAdd/ReadVariableOp2B
dense_336/MatMul/ReadVariableOpdense_336/MatMul/ReadVariableOp2D
 dense_337/BiasAdd/ReadVariableOp dense_337/BiasAdd/ReadVariableOp2B
dense_337/MatMul/ReadVariableOpdense_337/MatMul/ReadVariableOp2D
 dense_338/BiasAdd/ReadVariableOp dense_338/BiasAdd/ReadVariableOp2B
dense_338/MatMul/ReadVariableOpdense_338/MatMul/ReadVariableOp2D
 dense_339/BiasAdd/ReadVariableOp dense_339/BiasAdd/ReadVariableOp2B
dense_339/MatMul/ReadVariableOpdense_339/MatMul/ReadVariableOp2D
 dense_340/BiasAdd/ReadVariableOp dense_340/BiasAdd/ReadVariableOp2B
dense_340/MatMul/ReadVariableOpdense_340/MatMul/ReadVariableOp2D
 dense_341/BiasAdd/ReadVariableOp dense_341/BiasAdd/ReadVariableOp2B
dense_341/MatMul/ReadVariableOpdense_341/MatMul/ReadVariableOp:X T
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
F__inference_dense_337_layer_call_and_return_conditional_losses_1595820

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_340_layer_call_and_return_conditional_losses_1595880

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_338_layer_call_fn_1595829

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


/__inference_sequential_71_layer_call_fn_1595431
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595359o
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
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


/__inference_sequential_71_layer_call_fn_1595204
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595169o
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
+__inference_dense_337_layer_call_fn_1595809

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
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
Æ

+__inference_dense_341_layer_call_fn_1595889

inputs
unknown:
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_335_layer_call_fn_1595769

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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
¹C
²
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595701

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_335_matmul_readvariableop_resource:7
)dense_335_biasadd_readvariableop_resource::
(dense_336_matmul_readvariableop_resource:7
)dense_336_biasadd_readvariableop_resource::
(dense_337_matmul_readvariableop_resource:7
)dense_337_biasadd_readvariableop_resource::
(dense_338_matmul_readvariableop_resource:7
)dense_338_biasadd_readvariableop_resource::
(dense_339_matmul_readvariableop_resource:7
)dense_339_biasadd_readvariableop_resource::
(dense_340_matmul_readvariableop_resource:7
)dense_340_biasadd_readvariableop_resource::
(dense_341_matmul_readvariableop_resource:7
)dense_341_biasadd_readvariableop_resource:
identity¢ dense_335/BiasAdd/ReadVariableOp¢dense_335/MatMul/ReadVariableOp¢ dense_336/BiasAdd/ReadVariableOp¢dense_336/MatMul/ReadVariableOp¢ dense_337/BiasAdd/ReadVariableOp¢dense_337/MatMul/ReadVariableOp¢ dense_338/BiasAdd/ReadVariableOp¢dense_338/MatMul/ReadVariableOp¢ dense_339/BiasAdd/ReadVariableOp¢dense_339/MatMul/ReadVariableOp¢ dense_340/BiasAdd/ReadVariableOp¢dense_340/MatMul/ReadVariableOp¢ dense_341/BiasAdd/ReadVariableOp¢dense_341/MatMul/ReadVariableOpg
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
dense_335/MatMul/ReadVariableOpReadVariableOp(dense_335_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_335/MatMulMatMulnormalization/truediv:z:0'dense_335/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_335/BiasAdd/ReadVariableOpReadVariableOp)dense_335_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_335/BiasAddBiasAdddense_335/MatMul:product:0(dense_335/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_335/ReluReludense_335/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_336/MatMul/ReadVariableOpReadVariableOp(dense_336_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_336/MatMulMatMuldense_335/Relu:activations:0'dense_336/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_336/BiasAdd/ReadVariableOpReadVariableOp)dense_336_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_336/BiasAddBiasAdddense_336/MatMul:product:0(dense_336/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_336/ReluReludense_336/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_337/MatMul/ReadVariableOpReadVariableOp(dense_337_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_337/MatMulMatMuldense_336/Relu:activations:0'dense_337/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_337/BiasAdd/ReadVariableOpReadVariableOp)dense_337_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_337/BiasAddBiasAdddense_337/MatMul:product:0(dense_337/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_337/ReluReludense_337/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_338/MatMul/ReadVariableOpReadVariableOp(dense_338_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_338/MatMulMatMuldense_337/Relu:activations:0'dense_338/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_338/BiasAdd/ReadVariableOpReadVariableOp)dense_338_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_338/BiasAddBiasAdddense_338/MatMul:product:0(dense_338/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_338/ReluReludense_338/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_339/MatMul/ReadVariableOpReadVariableOp(dense_339_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_339/MatMulMatMuldense_338/Relu:activations:0'dense_339/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_339/BiasAdd/ReadVariableOpReadVariableOp)dense_339_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_339/BiasAddBiasAdddense_339/MatMul:product:0(dense_339/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_339/ReluReludense_339/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_340/MatMul/ReadVariableOpReadVariableOp(dense_340_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_340/MatMulMatMuldense_339/Relu:activations:0'dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_340/BiasAdd/ReadVariableOpReadVariableOp)dense_340_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_340/BiasAddBiasAdddense_340/MatMul:product:0(dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_340/ReluReludense_340/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_341/MatMul/ReadVariableOpReadVariableOp(dense_341_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_341/MatMulMatMuldense_340/Relu:activations:0'dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_341/BiasAdd/ReadVariableOpReadVariableOp)dense_341_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_341/BiasAddBiasAdddense_341/MatMul:product:0(dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_341/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp!^dense_335/BiasAdd/ReadVariableOp ^dense_335/MatMul/ReadVariableOp!^dense_336/BiasAdd/ReadVariableOp ^dense_336/MatMul/ReadVariableOp!^dense_337/BiasAdd/ReadVariableOp ^dense_337/MatMul/ReadVariableOp!^dense_338/BiasAdd/ReadVariableOp ^dense_338/MatMul/ReadVariableOp!^dense_339/BiasAdd/ReadVariableOp ^dense_339/MatMul/ReadVariableOp!^dense_340/BiasAdd/ReadVariableOp ^dense_340/MatMul/ReadVariableOp!^dense_341/BiasAdd/ReadVariableOp ^dense_341/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2D
 dense_335/BiasAdd/ReadVariableOp dense_335/BiasAdd/ReadVariableOp2B
dense_335/MatMul/ReadVariableOpdense_335/MatMul/ReadVariableOp2D
 dense_336/BiasAdd/ReadVariableOp dense_336/BiasAdd/ReadVariableOp2B
dense_336/MatMul/ReadVariableOpdense_336/MatMul/ReadVariableOp2D
 dense_337/BiasAdd/ReadVariableOp dense_337/BiasAdd/ReadVariableOp2B
dense_337/MatMul/ReadVariableOpdense_337/MatMul/ReadVariableOp2D
 dense_338/BiasAdd/ReadVariableOp dense_338/BiasAdd/ReadVariableOp2B
dense_338/MatMul/ReadVariableOpdense_338/MatMul/ReadVariableOp2D
 dense_339/BiasAdd/ReadVariableOp dense_339/BiasAdd/ReadVariableOp2B
dense_339/MatMul/ReadVariableOpdense_339/MatMul/ReadVariableOp2D
 dense_340/BiasAdd/ReadVariableOp dense_340/BiasAdd/ReadVariableOp2B
dense_340/MatMul/ReadVariableOpdense_340/MatMul/ReadVariableOp2D
 dense_341/BiasAdd/ReadVariableOp dense_341/BiasAdd/ReadVariableOp2B
dense_341/MatMul/ReadVariableOpdense_341/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
åÎ
À
#__inference__traced_restore_1596246
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_1:	 5
#assignvariableop_3_dense_335_kernel:/
!assignvariableop_4_dense_335_bias:5
#assignvariableop_5_dense_336_kernel:/
!assignvariableop_6_dense_336_bias:5
#assignvariableop_7_dense_337_kernel:/
!assignvariableop_8_dense_337_bias:5
#assignvariableop_9_dense_338_kernel:0
"assignvariableop_10_dense_338_bias:6
$assignvariableop_11_dense_339_kernel:0
"assignvariableop_12_dense_339_bias:6
$assignvariableop_13_dense_340_kernel:0
"assignvariableop_14_dense_340_bias:6
$assignvariableop_15_dense_341_kernel:0
"assignvariableop_16_dense_341_bias:'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: 0
&assignvariableop_21_adam_learning_rate: #
assignvariableop_22_total: #
assignvariableop_23_count: =
+assignvariableop_24_adam_dense_335_kernel_m:7
)assignvariableop_25_adam_dense_335_bias_m:=
+assignvariableop_26_adam_dense_336_kernel_m:7
)assignvariableop_27_adam_dense_336_bias_m:=
+assignvariableop_28_adam_dense_337_kernel_m:7
)assignvariableop_29_adam_dense_337_bias_m:=
+assignvariableop_30_adam_dense_338_kernel_m:7
)assignvariableop_31_adam_dense_338_bias_m:=
+assignvariableop_32_adam_dense_339_kernel_m:7
)assignvariableop_33_adam_dense_339_bias_m:=
+assignvariableop_34_adam_dense_340_kernel_m:7
)assignvariableop_35_adam_dense_340_bias_m:=
+assignvariableop_36_adam_dense_341_kernel_m:7
)assignvariableop_37_adam_dense_341_bias_m:=
+assignvariableop_38_adam_dense_335_kernel_v:7
)assignvariableop_39_adam_dense_335_bias_v:=
+assignvariableop_40_adam_dense_336_kernel_v:7
)assignvariableop_41_adam_dense_336_bias_v:=
+assignvariableop_42_adam_dense_337_kernel_v:7
)assignvariableop_43_adam_dense_337_bias_v:=
+assignvariableop_44_adam_dense_338_kernel_v:7
)assignvariableop_45_adam_dense_338_bias_v:=
+assignvariableop_46_adam_dense_339_kernel_v:7
)assignvariableop_47_adam_dense_339_bias_v:=
+assignvariableop_48_adam_dense_340_kernel_v:7
)assignvariableop_49_adam_dense_340_bias_v:=
+assignvariableop_50_adam_dense_341_kernel_v:7
)assignvariableop_51_adam_dense_341_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_335_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_335_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_336_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_336_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_337_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_337_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_338_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_338_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_339_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_339_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_340_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_340_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp$assignvariableop_15_dense_341_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_341_biasIdentity_16:output:0"/device:CPU:0*
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
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dense_335_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_335_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_336_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_336_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_337_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_337_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_338_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_338_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_339_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_339_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_dense_340_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_340_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_dense_341_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_341_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_dense_335_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_335_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_dense_336_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_336_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_dense_337_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_337_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_dense_338_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp)assignvariableop_45_adam_dense_338_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp+assignvariableop_46_adam_dense_339_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_339_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp+assignvariableop_48_adam_dense_340_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_340_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp+assignvariableop_50_adam_dense_341_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adam_dense_341_bias_vIdentity_51:output:0"/device:CPU:0*
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


÷
F__inference_dense_336_layer_call_and_return_conditional_losses_1595800

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý+

J__inference_sequential_71_layer_call_and_return_conditional_losses_1595359

inputs
normalization_sub_y
normalization_sqrt_x#
dense_335_1595323:
dense_335_1595325:#
dense_336_1595328:
dense_336_1595330:#
dense_337_1595333:
dense_337_1595335:#
dense_338_1595338:
dense_338_1595340:#
dense_339_1595343:
dense_339_1595345:#
dense_340_1595348:
dense_340_1595350:#
dense_341_1595353:
dense_341_1595355:
identity¢!dense_335/StatefulPartitionedCall¢!dense_336/StatefulPartitionedCall¢!dense_337/StatefulPartitionedCall¢!dense_338/StatefulPartitionedCall¢!dense_339/StatefulPartitionedCall¢!dense_340/StatefulPartitionedCall¢!dense_341/StatefulPartitionedCallg
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
!dense_335/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_335_1595323dense_335_1595325*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061
!dense_336/StatefulPartitionedCallStatefulPartitionedCall*dense_335/StatefulPartitionedCall:output:0dense_336_1595328dense_336_1595330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078
!dense_337/StatefulPartitionedCallStatefulPartitionedCall*dense_336/StatefulPartitionedCall:output:0dense_337_1595333dense_337_1595335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095
!dense_338/StatefulPartitionedCallStatefulPartitionedCall*dense_337/StatefulPartitionedCall:output:0dense_338_1595338dense_338_1595340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112
!dense_339/StatefulPartitionedCallStatefulPartitionedCall*dense_338/StatefulPartitionedCall:output:0dense_339_1595343dense_339_1595345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129
!dense_340/StatefulPartitionedCallStatefulPartitionedCall*dense_339/StatefulPartitionedCall:output:0dense_340_1595348dense_340_1595350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1595353dense_341_1595355*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162y
IdentityIdentity*dense_341/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_335/StatefulPartitionedCall"^dense_336/StatefulPartitionedCall"^dense_337/StatefulPartitionedCall"^dense_338/StatefulPartitionedCall"^dense_339/StatefulPartitionedCall"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_335/StatefulPartitionedCall!dense_335/StatefulPartitionedCall2F
!dense_336/StatefulPartitionedCall!dense_336/StatefulPartitionedCall2F
!dense_337/StatefulPartitionedCall!dense_337/StatefulPartitionedCall2F
!dense_338/StatefulPartitionedCall!dense_338/StatefulPartitionedCall2F
!dense_339/StatefulPartitionedCall!dense_339/StatefulPartitionedCall2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
í

%__inference_signature_wrapper_1595568
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

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
"__inference__wrapped_model_1595036o
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
¤,

J__inference_sequential_71_layer_call_and_return_conditional_losses_1595523
normalization_input
normalization_sub_y
normalization_sqrt_x#
dense_335_1595487:
dense_335_1595489:#
dense_336_1595492:
dense_336_1595494:#
dense_337_1595497:
dense_337_1595499:#
dense_338_1595502:
dense_338_1595504:#
dense_339_1595507:
dense_339_1595509:#
dense_340_1595512:
dense_340_1595514:#
dense_341_1595517:
dense_341_1595519:
identity¢!dense_335/StatefulPartitionedCall¢!dense_336/StatefulPartitionedCall¢!dense_337/StatefulPartitionedCall¢!dense_338/StatefulPartitionedCall¢!dense_339/StatefulPartitionedCall¢!dense_340/StatefulPartitionedCall¢!dense_341/StatefulPartitionedCallt
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
!dense_335/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_335_1595487dense_335_1595489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061
!dense_336/StatefulPartitionedCallStatefulPartitionedCall*dense_335/StatefulPartitionedCall:output:0dense_336_1595492dense_336_1595494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078
!dense_337/StatefulPartitionedCallStatefulPartitionedCall*dense_336/StatefulPartitionedCall:output:0dense_337_1595497dense_337_1595499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095
!dense_338/StatefulPartitionedCallStatefulPartitionedCall*dense_337/StatefulPartitionedCall:output:0dense_338_1595502dense_338_1595504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112
!dense_339/StatefulPartitionedCallStatefulPartitionedCall*dense_338/StatefulPartitionedCall:output:0dense_339_1595507dense_339_1595509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129
!dense_340/StatefulPartitionedCallStatefulPartitionedCall*dense_339/StatefulPartitionedCall:output:0dense_340_1595512dense_340_1595514*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1595517dense_341_1595519*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162y
IdentityIdentity*dense_341/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_335/StatefulPartitionedCall"^dense_336/StatefulPartitionedCall"^dense_337/StatefulPartitionedCall"^dense_338/StatefulPartitionedCall"^dense_339/StatefulPartitionedCall"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_335/StatefulPartitionedCall!dense_335/StatefulPartitionedCall2F
!dense_336/StatefulPartitionedCall!dense_336/StatefulPartitionedCall2F
!dense_337/StatefulPartitionedCall!dense_337/StatefulPartitionedCall2F
!dense_338/StatefulPartitionedCall!dense_338/StatefulPartitionedCall2F
!dense_339/StatefulPartitionedCall!dense_339/StatefulPartitionedCall2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall:e a
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
F__inference_dense_339_layer_call_and_return_conditional_losses_1595860

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

/__inference_sequential_71_layer_call_fn_1595605

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595169o
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
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_dense_339_layer_call_fn_1595849

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É	
÷
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
ý+

J__inference_sequential_71_layer_call_and_return_conditional_losses_1595169

inputs
normalization_sub_y
normalization_sqrt_x#
dense_335_1595062:
dense_335_1595064:#
dense_336_1595079:
dense_336_1595081:#
dense_337_1595096:
dense_337_1595098:#
dense_338_1595113:
dense_338_1595115:#
dense_339_1595130:
dense_339_1595132:#
dense_340_1595147:
dense_340_1595149:#
dense_341_1595163:
dense_341_1595165:
identity¢!dense_335/StatefulPartitionedCall¢!dense_336/StatefulPartitionedCall¢!dense_337/StatefulPartitionedCall¢!dense_338/StatefulPartitionedCall¢!dense_339/StatefulPartitionedCall¢!dense_340/StatefulPartitionedCall¢!dense_341/StatefulPartitionedCallg
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
!dense_335/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_335_1595062dense_335_1595064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061
!dense_336/StatefulPartitionedCallStatefulPartitionedCall*dense_335/StatefulPartitionedCall:output:0dense_336_1595079dense_336_1595081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078
!dense_337/StatefulPartitionedCallStatefulPartitionedCall*dense_336/StatefulPartitionedCall:output:0dense_337_1595096dense_337_1595098*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095
!dense_338/StatefulPartitionedCallStatefulPartitionedCall*dense_337/StatefulPartitionedCall:output:0dense_338_1595113dense_338_1595115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112
!dense_339/StatefulPartitionedCallStatefulPartitionedCall*dense_338/StatefulPartitionedCall:output:0dense_339_1595130dense_339_1595132*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129
!dense_340/StatefulPartitionedCallStatefulPartitionedCall*dense_339/StatefulPartitionedCall:output:0dense_340_1595147dense_340_1595149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1595163dense_341_1595165*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162y
IdentityIdentity*dense_341/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_335/StatefulPartitionedCall"^dense_336/StatefulPartitionedCall"^dense_337/StatefulPartitionedCall"^dense_338/StatefulPartitionedCall"^dense_339/StatefulPartitionedCall"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_335/StatefulPartitionedCall!dense_335/StatefulPartitionedCall2F
!dense_336/StatefulPartitionedCall!dense_336/StatefulPartitionedCall2F
!dense_337/StatefulPartitionedCall!dense_337/StatefulPartitionedCall2F
!dense_338/StatefulPartitionedCall!dense_338/StatefulPartitionedCall2F
!dense_339/StatefulPartitionedCall!dense_339/StatefulPartitionedCall2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall:X T
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
"__inference__wrapped_model_1595036
normalization_input%
!sequential_71_normalization_sub_y&
"sequential_71_normalization_sqrt_xH
6sequential_71_dense_335_matmul_readvariableop_resource:E
7sequential_71_dense_335_biasadd_readvariableop_resource:H
6sequential_71_dense_336_matmul_readvariableop_resource:E
7sequential_71_dense_336_biasadd_readvariableop_resource:H
6sequential_71_dense_337_matmul_readvariableop_resource:E
7sequential_71_dense_337_biasadd_readvariableop_resource:H
6sequential_71_dense_338_matmul_readvariableop_resource:E
7sequential_71_dense_338_biasadd_readvariableop_resource:H
6sequential_71_dense_339_matmul_readvariableop_resource:E
7sequential_71_dense_339_biasadd_readvariableop_resource:H
6sequential_71_dense_340_matmul_readvariableop_resource:E
7sequential_71_dense_340_biasadd_readvariableop_resource:H
6sequential_71_dense_341_matmul_readvariableop_resource:E
7sequential_71_dense_341_biasadd_readvariableop_resource:
identity¢.sequential_71/dense_335/BiasAdd/ReadVariableOp¢-sequential_71/dense_335/MatMul/ReadVariableOp¢.sequential_71/dense_336/BiasAdd/ReadVariableOp¢-sequential_71/dense_336/MatMul/ReadVariableOp¢.sequential_71/dense_337/BiasAdd/ReadVariableOp¢-sequential_71/dense_337/MatMul/ReadVariableOp¢.sequential_71/dense_338/BiasAdd/ReadVariableOp¢-sequential_71/dense_338/MatMul/ReadVariableOp¢.sequential_71/dense_339/BiasAdd/ReadVariableOp¢-sequential_71/dense_339/MatMul/ReadVariableOp¢.sequential_71/dense_340/BiasAdd/ReadVariableOp¢-sequential_71/dense_340/MatMul/ReadVariableOp¢.sequential_71/dense_341/BiasAdd/ReadVariableOp¢-sequential_71/dense_341/MatMul/ReadVariableOp
sequential_71/normalization/subSubnormalization_input!sequential_71_normalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 sequential_71/normalization/SqrtSqrt"sequential_71_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_71/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3­
#sequential_71/normalization/MaximumMaximum$sequential_71/normalization/Sqrt:y:0.sequential_71/normalization/Maximum/y:output:0*
T0*
_output_shapes

:®
#sequential_71/normalization/truedivRealDiv#sequential_71/normalization/sub:z:0'sequential_71/normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_335/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_335_matmul_readvariableop_resource*
_output_shapes

:*
dtype0º
sequential_71/dense_335/MatMulMatMul'sequential_71/normalization/truediv:z:05sequential_71/dense_335/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_335/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_335_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_335/BiasAddBiasAdd(sequential_71/dense_335/MatMul:product:06sequential_71/dense_335/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_335/ReluRelu(sequential_71/dense_335/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_336/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_336_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_336/MatMulMatMul*sequential_71/dense_335/Relu:activations:05sequential_71/dense_336/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_336/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_336_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_336/BiasAddBiasAdd(sequential_71/dense_336/MatMul:product:06sequential_71/dense_336/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_336/ReluRelu(sequential_71/dense_336/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_337/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_337_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_337/MatMulMatMul*sequential_71/dense_336/Relu:activations:05sequential_71/dense_337/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_337/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_337_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_337/BiasAddBiasAdd(sequential_71/dense_337/MatMul:product:06sequential_71/dense_337/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_337/ReluRelu(sequential_71/dense_337/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_338/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_338_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_338/MatMulMatMul*sequential_71/dense_337/Relu:activations:05sequential_71/dense_338/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_338/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_338_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_338/BiasAddBiasAdd(sequential_71/dense_338/MatMul:product:06sequential_71/dense_338/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_338/ReluRelu(sequential_71/dense_338/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_339/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_339_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_339/MatMulMatMul*sequential_71/dense_338/Relu:activations:05sequential_71/dense_339/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_339/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_339_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_339/BiasAddBiasAdd(sequential_71/dense_339/MatMul:product:06sequential_71/dense_339/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_339/ReluRelu(sequential_71/dense_339/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_340/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_340_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_340/MatMulMatMul*sequential_71/dense_339/Relu:activations:05sequential_71/dense_340/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_340/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_340_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_340/BiasAddBiasAdd(sequential_71/dense_340/MatMul:product:06sequential_71/dense_340/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_71/dense_340/ReluRelu(sequential_71/dense_340/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_71/dense_341/MatMul/ReadVariableOpReadVariableOp6sequential_71_dense_341_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_71/dense_341/MatMulMatMul*sequential_71/dense_340/Relu:activations:05sequential_71/dense_341/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_71/dense_341/BiasAdd/ReadVariableOpReadVariableOp7sequential_71_dense_341_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_71/dense_341/BiasAddBiasAdd(sequential_71/dense_341/MatMul:product:06sequential_71/dense_341/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_71/dense_341/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
NoOpNoOp/^sequential_71/dense_335/BiasAdd/ReadVariableOp.^sequential_71/dense_335/MatMul/ReadVariableOp/^sequential_71/dense_336/BiasAdd/ReadVariableOp.^sequential_71/dense_336/MatMul/ReadVariableOp/^sequential_71/dense_337/BiasAdd/ReadVariableOp.^sequential_71/dense_337/MatMul/ReadVariableOp/^sequential_71/dense_338/BiasAdd/ReadVariableOp.^sequential_71/dense_338/MatMul/ReadVariableOp/^sequential_71/dense_339/BiasAdd/ReadVariableOp.^sequential_71/dense_339/MatMul/ReadVariableOp/^sequential_71/dense_340/BiasAdd/ReadVariableOp.^sequential_71/dense_340/MatMul/ReadVariableOp/^sequential_71/dense_341/BiasAdd/ReadVariableOp.^sequential_71/dense_341/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2`
.sequential_71/dense_335/BiasAdd/ReadVariableOp.sequential_71/dense_335/BiasAdd/ReadVariableOp2^
-sequential_71/dense_335/MatMul/ReadVariableOp-sequential_71/dense_335/MatMul/ReadVariableOp2`
.sequential_71/dense_336/BiasAdd/ReadVariableOp.sequential_71/dense_336/BiasAdd/ReadVariableOp2^
-sequential_71/dense_336/MatMul/ReadVariableOp-sequential_71/dense_336/MatMul/ReadVariableOp2`
.sequential_71/dense_337/BiasAdd/ReadVariableOp.sequential_71/dense_337/BiasAdd/ReadVariableOp2^
-sequential_71/dense_337/MatMul/ReadVariableOp-sequential_71/dense_337/MatMul/ReadVariableOp2`
.sequential_71/dense_338/BiasAdd/ReadVariableOp.sequential_71/dense_338/BiasAdd/ReadVariableOp2^
-sequential_71/dense_338/MatMul/ReadVariableOp-sequential_71/dense_338/MatMul/ReadVariableOp2`
.sequential_71/dense_339/BiasAdd/ReadVariableOp.sequential_71/dense_339/BiasAdd/ReadVariableOp2^
-sequential_71/dense_339/MatMul/ReadVariableOp-sequential_71/dense_339/MatMul/ReadVariableOp2`
.sequential_71/dense_340/BiasAdd/ReadVariableOp.sequential_71/dense_340/BiasAdd/ReadVariableOp2^
-sequential_71/dense_340/MatMul/ReadVariableOp-sequential_71/dense_340/MatMul/ReadVariableOp2`
.sequential_71/dense_341/BiasAdd/ReadVariableOp.sequential_71/dense_341/BiasAdd/ReadVariableOp2^
-sequential_71/dense_341/MatMul/ReadVariableOp-sequential_71/dense_341/MatMul/ReadVariableOp:e a
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
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤,

J__inference_sequential_71_layer_call_and_return_conditional_losses_1595477
normalization_input
normalization_sub_y
normalization_sqrt_x#
dense_335_1595441:
dense_335_1595443:#
dense_336_1595446:
dense_336_1595448:#
dense_337_1595451:
dense_337_1595453:#
dense_338_1595456:
dense_338_1595458:#
dense_339_1595461:
dense_339_1595463:#
dense_340_1595466:
dense_340_1595468:#
dense_341_1595471:
dense_341_1595473:
identity¢!dense_335/StatefulPartitionedCall¢!dense_336/StatefulPartitionedCall¢!dense_337/StatefulPartitionedCall¢!dense_338/StatefulPartitionedCall¢!dense_339/StatefulPartitionedCall¢!dense_340/StatefulPartitionedCall¢!dense_341/StatefulPartitionedCallt
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
!dense_335/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_335_1595441dense_335_1595443*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_335_layer_call_and_return_conditional_losses_1595061
!dense_336/StatefulPartitionedCallStatefulPartitionedCall*dense_335/StatefulPartitionedCall:output:0dense_336_1595446dense_336_1595448*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078
!dense_337/StatefulPartitionedCallStatefulPartitionedCall*dense_336/StatefulPartitionedCall:output:0dense_337_1595451dense_337_1595453*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_337_layer_call_and_return_conditional_losses_1595095
!dense_338/StatefulPartitionedCallStatefulPartitionedCall*dense_337/StatefulPartitionedCall:output:0dense_338_1595456dense_338_1595458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112
!dense_339/StatefulPartitionedCallStatefulPartitionedCall*dense_338/StatefulPartitionedCall:output:0dense_339_1595461dense_339_1595463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_339_layer_call_and_return_conditional_losses_1595129
!dense_340/StatefulPartitionedCallStatefulPartitionedCall*dense_339/StatefulPartitionedCall:output:0dense_340_1595466dense_340_1595468*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146
!dense_341/StatefulPartitionedCallStatefulPartitionedCall*dense_340/StatefulPartitionedCall:output:0dense_341_1595471dense_341_1595473*
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595162y
IdentityIdentity*dense_341/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp"^dense_335/StatefulPartitionedCall"^dense_336/StatefulPartitionedCall"^dense_337/StatefulPartitionedCall"^dense_338/StatefulPartitionedCall"^dense_339/StatefulPartitionedCall"^dense_340/StatefulPartitionedCall"^dense_341/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : : : : : 2F
!dense_335/StatefulPartitionedCall!dense_335/StatefulPartitionedCall2F
!dense_336/StatefulPartitionedCall!dense_336/StatefulPartitionedCall2F
!dense_337/StatefulPartitionedCall!dense_337/StatefulPartitionedCall2F
!dense_338/StatefulPartitionedCall!dense_338/StatefulPartitionedCall2F
!dense_339/StatefulPartitionedCall!dense_339/StatefulPartitionedCall2F
!dense_340/StatefulPartitionedCall!dense_340/StatefulPartitionedCall2F
!dense_341/StatefulPartitionedCall!dense_341/StatefulPartitionedCall:e a
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
+__inference_dense_336_layer_call_fn_1595789

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_336_layer_call_and_return_conditional_losses_1595078o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
g
µ
 __inference__traced_save_1596080
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop&
"savev2_count_1_read_readvariableop	/
+savev2_dense_335_kernel_read_readvariableop-
)savev2_dense_335_bias_read_readvariableop/
+savev2_dense_336_kernel_read_readvariableop-
)savev2_dense_336_bias_read_readvariableop/
+savev2_dense_337_kernel_read_readvariableop-
)savev2_dense_337_bias_read_readvariableop/
+savev2_dense_338_kernel_read_readvariableop-
)savev2_dense_338_bias_read_readvariableop/
+savev2_dense_339_kernel_read_readvariableop-
)savev2_dense_339_bias_read_readvariableop/
+savev2_dense_340_kernel_read_readvariableop-
)savev2_dense_340_bias_read_readvariableop/
+savev2_dense_341_kernel_read_readvariableop-
)savev2_dense_341_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_335_kernel_m_read_readvariableop4
0savev2_adam_dense_335_bias_m_read_readvariableop6
2savev2_adam_dense_336_kernel_m_read_readvariableop4
0savev2_adam_dense_336_bias_m_read_readvariableop6
2savev2_adam_dense_337_kernel_m_read_readvariableop4
0savev2_adam_dense_337_bias_m_read_readvariableop6
2savev2_adam_dense_338_kernel_m_read_readvariableop4
0savev2_adam_dense_338_bias_m_read_readvariableop6
2savev2_adam_dense_339_kernel_m_read_readvariableop4
0savev2_adam_dense_339_bias_m_read_readvariableop6
2savev2_adam_dense_340_kernel_m_read_readvariableop4
0savev2_adam_dense_340_bias_m_read_readvariableop6
2savev2_adam_dense_341_kernel_m_read_readvariableop4
0savev2_adam_dense_341_bias_m_read_readvariableop6
2savev2_adam_dense_335_kernel_v_read_readvariableop4
0savev2_adam_dense_335_bias_v_read_readvariableop6
2savev2_adam_dense_336_kernel_v_read_readvariableop4
0savev2_adam_dense_336_bias_v_read_readvariableop6
2savev2_adam_dense_337_kernel_v_read_readvariableop4
0savev2_adam_dense_337_bias_v_read_readvariableop6
2savev2_adam_dense_338_kernel_v_read_readvariableop4
0savev2_adam_dense_338_bias_v_read_readvariableop6
2savev2_adam_dense_339_kernel_v_read_readvariableop4
0savev2_adam_dense_339_bias_v_read_readvariableop6
2savev2_adam_dense_340_kernel_v_read_readvariableop4
0savev2_adam_dense_340_bias_v_read_readvariableop6
2savev2_adam_dense_341_kernel_v_read_readvariableop4
0savev2_adam_dense_341_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop"savev2_count_1_read_readvariableop+savev2_dense_335_kernel_read_readvariableop)savev2_dense_335_bias_read_readvariableop+savev2_dense_336_kernel_read_readvariableop)savev2_dense_336_bias_read_readvariableop+savev2_dense_337_kernel_read_readvariableop)savev2_dense_337_bias_read_readvariableop+savev2_dense_338_kernel_read_readvariableop)savev2_dense_338_bias_read_readvariableop+savev2_dense_339_kernel_read_readvariableop)savev2_dense_339_bias_read_readvariableop+savev2_dense_340_kernel_read_readvariableop)savev2_dense_340_bias_read_readvariableop+savev2_dense_341_kernel_read_readvariableop)savev2_dense_341_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_335_kernel_m_read_readvariableop0savev2_adam_dense_335_bias_m_read_readvariableop2savev2_adam_dense_336_kernel_m_read_readvariableop0savev2_adam_dense_336_bias_m_read_readvariableop2savev2_adam_dense_337_kernel_m_read_readvariableop0savev2_adam_dense_337_bias_m_read_readvariableop2savev2_adam_dense_338_kernel_m_read_readvariableop0savev2_adam_dense_338_bias_m_read_readvariableop2savev2_adam_dense_339_kernel_m_read_readvariableop0savev2_adam_dense_339_bias_m_read_readvariableop2savev2_adam_dense_340_kernel_m_read_readvariableop0savev2_adam_dense_340_bias_m_read_readvariableop2savev2_adam_dense_341_kernel_m_read_readvariableop0savev2_adam_dense_341_bias_m_read_readvariableop2savev2_adam_dense_335_kernel_v_read_readvariableop0savev2_adam_dense_335_bias_v_read_readvariableop2savev2_adam_dense_336_kernel_v_read_readvariableop0savev2_adam_dense_336_bias_v_read_readvariableop2savev2_adam_dense_337_kernel_v_read_readvariableop0savev2_adam_dense_337_bias_v_read_readvariableop2savev2_adam_dense_338_kernel_v_read_readvariableop0savev2_adam_dense_338_bias_v_read_readvariableop2savev2_adam_dense_339_kernel_v_read_readvariableop0savev2_adam_dense_339_bias_v_read_readvariableop2savev2_adam_dense_340_kernel_v_read_readvariableop0savev2_adam_dense_340_bias_v_read_readvariableop2savev2_adam_dense_341_kernel_v_read_readvariableop0savev2_adam_dense_341_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
ð: ::: ::::::::::::::: : : : : : : ::::::::::::::::::::::::::::: 2(
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:: .

_output_shapes
::$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:: 2

_output_shapes
::$3 

_output_shapes

:: 4

_output_shapes
::5

_output_shapes
: 


÷
F__inference_dense_338_layer_call_and_return_conditional_losses_1595840

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


÷
F__inference_dense_338_layer_call_and_return_conditional_losses_1595112

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

/__inference_sequential_71_layer_call_fn_1595642

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595359o
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
+__inference_dense_340_layer_call_fn_1595869

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_340_layer_call_and_return_conditional_losses_1595146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¿L
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
	dense_3410
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
/__inference_sequential_71_layer_call_fn_1595204
/__inference_sequential_71_layer_call_fn_1595605
/__inference_sequential_71_layer_call_fn_1595642
/__inference_sequential_71_layer_call_fn_1595431À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595701
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595760
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595477
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595523À
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
"__inference__wrapped_model_1595036normalization_input"
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
+__inference_dense_335_layer_call_fn_1595769¢
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
F__inference_dense_335_layer_call_and_return_conditional_losses_1595780¢
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
": 2dense_335/kernel
:2dense_335/bias
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
+__inference_dense_336_layer_call_fn_1595789¢
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
F__inference_dense_336_layer_call_and_return_conditional_losses_1595800¢
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
": 2dense_336/kernel
:2dense_336/bias
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
+__inference_dense_337_layer_call_fn_1595809¢
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
F__inference_dense_337_layer_call_and_return_conditional_losses_1595820¢
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
": 2dense_337/kernel
:2dense_337/bias
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
+__inference_dense_338_layer_call_fn_1595829¢
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
F__inference_dense_338_layer_call_and_return_conditional_losses_1595840¢
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
": 2dense_338/kernel
:2dense_338/bias
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
+__inference_dense_339_layer_call_fn_1595849¢
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
F__inference_dense_339_layer_call_and_return_conditional_losses_1595860¢
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
": 2dense_339/kernel
:2dense_339/bias
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
+__inference_dense_340_layer_call_fn_1595869¢
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
F__inference_dense_340_layer_call_and_return_conditional_losses_1595880¢
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
": 2dense_340/kernel
:2dense_340/bias
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
+__inference_dense_341_layer_call_fn_1595889¢
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595899¢
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
": 2dense_341/kernel
:2dense_341/bias
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
/__inference_sequential_71_layer_call_fn_1595204normalization_input"À
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
/__inference_sequential_71_layer_call_fn_1595605inputs"À
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
/__inference_sequential_71_layer_call_fn_1595642inputs"À
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
/__inference_sequential_71_layer_call_fn_1595431normalization_input"À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595701inputs"À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595760inputs"À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595477normalization_input"À
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595523normalization_input"À
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
%__inference_signature_wrapper_1595568normalization_input"
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
+__inference_dense_335_layer_call_fn_1595769inputs"¢
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
F__inference_dense_335_layer_call_and_return_conditional_losses_1595780inputs"¢
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
+__inference_dense_336_layer_call_fn_1595789inputs"¢
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
F__inference_dense_336_layer_call_and_return_conditional_losses_1595800inputs"¢
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
+__inference_dense_337_layer_call_fn_1595809inputs"¢
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
F__inference_dense_337_layer_call_and_return_conditional_losses_1595820inputs"¢
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
+__inference_dense_338_layer_call_fn_1595829inputs"¢
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
F__inference_dense_338_layer_call_and_return_conditional_losses_1595840inputs"¢
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
+__inference_dense_339_layer_call_fn_1595849inputs"¢
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
F__inference_dense_339_layer_call_and_return_conditional_losses_1595860inputs"¢
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
+__inference_dense_340_layer_call_fn_1595869inputs"¢
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
F__inference_dense_340_layer_call_and_return_conditional_losses_1595880inputs"¢
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
+__inference_dense_341_layer_call_fn_1595889inputs"¢
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
F__inference_dense_341_layer_call_and_return_conditional_losses_1595899inputs"¢
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
':%2Adam/dense_335/kernel/m
!:2Adam/dense_335/bias/m
':%2Adam/dense_336/kernel/m
!:2Adam/dense_336/bias/m
':%2Adam/dense_337/kernel/m
!:2Adam/dense_337/bias/m
':%2Adam/dense_338/kernel/m
!:2Adam/dense_338/bias/m
':%2Adam/dense_339/kernel/m
!:2Adam/dense_339/bias/m
':%2Adam/dense_340/kernel/m
!:2Adam/dense_340/bias/m
':%2Adam/dense_341/kernel/m
!:2Adam/dense_341/bias/m
':%2Adam/dense_335/kernel/v
!:2Adam/dense_335/bias/v
':%2Adam/dense_336/kernel/v
!:2Adam/dense_336/bias/v
':%2Adam/dense_337/kernel/v
!:2Adam/dense_337/bias/v
':%2Adam/dense_338/kernel/v
!:2Adam/dense_338/bias/v
':%2Adam/dense_339/kernel/v
!:2Adam/dense_339/bias/v
':%2Adam/dense_340/kernel/v
!:2Adam/dense_340/bias/v
':%2Adam/dense_341/kernel/v
!:2Adam/dense_341/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant¹
"__inference__wrapped_model_1595036¹º!")*129:ABIJQRE¢B
;¢8
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_341# 
	dense_341ÿÿÿÿÿÿÿÿÿe
__inference_adapt_step_21695E:¢7
0¢-
+(¢
 IteratorSpec 
ª "
 ¦
F__inference_dense_335_layer_call_and_return_conditional_losses_1595780\!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_335_layer_call_fn_1595769O!"/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_336_layer_call_and_return_conditional_losses_1595800\)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_336_layer_call_fn_1595789O)*/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_337_layer_call_and_return_conditional_losses_1595820\12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_337_layer_call_fn_1595809O12/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_338_layer_call_and_return_conditional_losses_1595840\9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_338_layer_call_fn_1595829O9:/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_339_layer_call_and_return_conditional_losses_1595860\AB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_339_layer_call_fn_1595849OAB/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_340_layer_call_and_return_conditional_losses_1595880\IJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_340_layer_call_fn_1595869OIJ/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
F__inference_dense_341_layer_call_and_return_conditional_losses_1595899\QR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ~
+__inference_dense_341_layer_call_fn_1595889OQR/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÙ
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595477¹º!")*129:ABIJQRM¢J
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595523¹º!")*129:ABIJQRM¢J
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595701}¹º!")*129:ABIJQR@¢=
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
J__inference_sequential_71_layer_call_and_return_conditional_losses_1595760}¹º!")*129:ABIJQR@¢=
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
/__inference_sequential_71_layer_call_fn_1595204}¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ°
/__inference_sequential_71_layer_call_fn_1595431}¹º!")*129:ABIJQRM¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ£
/__inference_sequential_71_layer_call_fn_1595605p¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ£
/__inference_sequential_71_layer_call_fn_1595642p¹º!")*129:ABIJQR@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÓ
%__inference_signature_wrapper_1595568©¹º!")*129:ABIJQR\¢Y
¢ 
RªO
M
normalization_input63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_341# 
	dense_341ÿÿÿÿÿÿÿÿÿ