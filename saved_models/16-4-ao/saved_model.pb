Ëí
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
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018ä½

Adam/dense_171/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_171/bias/v
{
)Adam/dense_171/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_171/bias/v*
_output_shapes
:*
dtype0

Adam/dense_171/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_171/kernel/v

+Adam/dense_171/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_171/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_170/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_170/bias/v
{
)Adam/dense_170/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_170/bias/v*
_output_shapes
:*
dtype0

Adam/dense_170/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_170/kernel/v

+Adam/dense_170/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_170/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_169/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_169/bias/v
{
)Adam/dense_169/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_169/bias/v*
_output_shapes
:*
dtype0

Adam/dense_169/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_169/kernel/v

+Adam/dense_169/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_169/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_168/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_168/bias/v
{
)Adam/dense_168/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_168/bias/v*
_output_shapes
:*
dtype0

Adam/dense_168/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_168/kernel/v

+Adam/dense_168/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_168/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_167/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_167/bias/v
{
)Adam/dense_167/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/v*
_output_shapes
:*
dtype0

Adam/dense_167/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_167/kernel/v

+Adam/dense_167/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_171/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_171/bias/m
{
)Adam/dense_171/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_171/bias/m*
_output_shapes
:*
dtype0

Adam/dense_171/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_171/kernel/m

+Adam/dense_171/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_171/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_170/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_170/bias/m
{
)Adam/dense_170/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_170/bias/m*
_output_shapes
:*
dtype0

Adam/dense_170/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_170/kernel/m

+Adam/dense_170/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_170/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_169/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_169/bias/m
{
)Adam/dense_169/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_169/bias/m*
_output_shapes
:*
dtype0

Adam/dense_169/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_169/kernel/m

+Adam/dense_169/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_169/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_168/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_168/bias/m
{
)Adam/dense_168/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_168/bias/m*
_output_shapes
:*
dtype0

Adam/dense_168/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_168/kernel/m

+Adam/dense_168/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_168/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_167/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_167/bias/m
{
)Adam/dense_167/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/m*
_output_shapes
:*
dtype0

Adam/dense_167/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_167/kernel/m

+Adam/dense_167/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/m*
_output_shapes

:*
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
dense_171/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_171/bias
m
"dense_171/bias/Read/ReadVariableOpReadVariableOpdense_171/bias*
_output_shapes
:*
dtype0
|
dense_171/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_171/kernel
u
$dense_171/kernel/Read/ReadVariableOpReadVariableOpdense_171/kernel*
_output_shapes

:*
dtype0
t
dense_170/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_170/bias
m
"dense_170/bias/Read/ReadVariableOpReadVariableOpdense_170/bias*
_output_shapes
:*
dtype0
|
dense_170/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_170/kernel
u
$dense_170/kernel/Read/ReadVariableOpReadVariableOpdense_170/kernel*
_output_shapes

:*
dtype0
t
dense_169/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_169/bias
m
"dense_169/bias/Read/ReadVariableOpReadVariableOpdense_169/bias*
_output_shapes
:*
dtype0
|
dense_169/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_169/kernel
u
$dense_169/kernel/Read/ReadVariableOpReadVariableOpdense_169/kernel*
_output_shapes

:*
dtype0
t
dense_168/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_168/bias
m
"dense_168/bias/Read/ReadVariableOpReadVariableOpdense_168/bias*
_output_shapes
:*
dtype0
|
dense_168/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_168/kernel
u
$dense_168/kernel/Read/ReadVariableOpReadVariableOpdense_168/kernel*
_output_shapes

:*
dtype0
t
dense_167/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_167/bias
m
"dense_167/bias/Read/ReadVariableOpReadVariableOpdense_167/bias*
_output_shapes
:*
dtype0
|
dense_167/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_167/kernel
u
$dense_167/kernel/Read/ReadVariableOpReadVariableOpdense_167/kernel*
_output_shapes

:*
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
G
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*ÅF
value»FB¸F B±F
¶
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¾
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias*
¦
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
¦
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
¦
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
¦
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
b
0
1
2
3
 4
'5
(6
/7
08
79
810
?11
@12*
J
0
 1
'2
(3
/4
05
76
87
?8
@9*
* 
°
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ftrace_0
Gtrace_1
Htrace_2
Itrace_3* 
6
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_3* 
* 

Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratem} m~'m(m/m0m7m8m?m@mv v'v(v/v0v7v8v?v@v*

Sserving_default* 
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
Ttrace_0* 

0
 1*

0
 1*
* 

Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ztrace_0* 

[trace_0* 
`Z
VARIABLE_VALUEdense_167/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_167/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 

\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

atrace_0* 

btrace_0* 
`Z
VARIABLE_VALUEdense_168/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_168/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
`Z
VARIABLE_VALUEdense_169/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_169/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
`Z
VARIABLE_VALUEdense_170/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_170/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
`Z
VARIABLE_VALUEdense_171/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_171/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*
.
0
1
2
3
4
5*

x0*
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
8
y	variables
z	keras_api
	{total
	|count*

{0
|1*

y	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_167/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_167/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_168/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_168/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_169/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_169/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_170/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_170/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_171/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_171/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_167/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_167/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_168/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_168/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_169/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_169/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_170/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_170/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/dense_171/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense_171/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

#serving_default_normalization_inputPlaceholder*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*%
shape:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputConstConst_1dense_167/kerneldense_167/biasdense_168/kerneldense_168/biasdense_169/kerneldense_169/biasdense_170/kerneldense_170/biasdense_171/kerneldense_171/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_964853
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount_1/Read/ReadVariableOp$dense_167/kernel/Read/ReadVariableOp"dense_167/bias/Read/ReadVariableOp$dense_168/kernel/Read/ReadVariableOp"dense_168/bias/Read/ReadVariableOp$dense_169/kernel/Read/ReadVariableOp"dense_169/bias/Read/ReadVariableOp$dense_170/kernel/Read/ReadVariableOp"dense_170/bias/Read/ReadVariableOp$dense_171/kernel/Read/ReadVariableOp"dense_171/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/dense_167/kernel/m/Read/ReadVariableOp)Adam/dense_167/bias/m/Read/ReadVariableOp+Adam/dense_168/kernel/m/Read/ReadVariableOp)Adam/dense_168/bias/m/Read/ReadVariableOp+Adam/dense_169/kernel/m/Read/ReadVariableOp)Adam/dense_169/bias/m/Read/ReadVariableOp+Adam/dense_170/kernel/m/Read/ReadVariableOp)Adam/dense_170/bias/m/Read/ReadVariableOp+Adam/dense_171/kernel/m/Read/ReadVariableOp)Adam/dense_171/bias/m/Read/ReadVariableOp+Adam/dense_167/kernel/v/Read/ReadVariableOp)Adam/dense_167/bias/v/Read/ReadVariableOp+Adam/dense_168/kernel/v/Read/ReadVariableOp)Adam/dense_168/bias/v/Read/ReadVariableOp+Adam/dense_169/kernel/v/Read/ReadVariableOp)Adam/dense_169/bias/v/Read/ReadVariableOp+Adam/dense_170/kernel/v/Read/ReadVariableOp)Adam/dense_170/bias/v/Read/ReadVariableOp+Adam/dense_171/kernel/v/Read/ReadVariableOp)Adam/dense_171/bias/v/Read/ReadVariableOpConst_2*5
Tin.
,2*		*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_965245
´
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecount_1dense_167/kerneldense_167/biasdense_168/kerneldense_168/biasdense_169/kerneldense_169/biasdense_170/kerneldense_170/biasdense_171/kerneldense_171/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_167/kernel/mAdam/dense_167/bias/mAdam/dense_168/kernel/mAdam/dense_168/bias/mAdam/dense_169/kernel/mAdam/dense_169/bias/mAdam/dense_170/kernel/mAdam/dense_170/bias/mAdam/dense_171/kernel/mAdam/dense_171/bias/mAdam/dense_167/kernel/vAdam/dense_167/bias/vAdam/dense_168/kernel/vAdam/dense_168/bias/vAdam/dense_169/kernel/vAdam/dense_169/bias/vAdam/dense_170/kernel/vAdam/dense_170/bias/vAdam/dense_171/kernel/vAdam/dense_171/bias/v*4
Tin-
+2)*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_965375ò
Ð2
½
I__inference_sequential_43_layer_call_and_return_conditional_losses_965001

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_167_matmul_readvariableop_resource:7
)dense_167_biasadd_readvariableop_resource::
(dense_168_matmul_readvariableop_resource:7
)dense_168_biasadd_readvariableop_resource::
(dense_169_matmul_readvariableop_resource:7
)dense_169_biasadd_readvariableop_resource::
(dense_170_matmul_readvariableop_resource:7
)dense_170_biasadd_readvariableop_resource::
(dense_171_matmul_readvariableop_resource:7
)dense_171_biasadd_readvariableop_resource:
identity¢ dense_167/BiasAdd/ReadVariableOp¢dense_167/MatMul/ReadVariableOp¢ dense_168/BiasAdd/ReadVariableOp¢dense_168/MatMul/ReadVariableOp¢ dense_169/BiasAdd/ReadVariableOp¢dense_169/MatMul/ReadVariableOp¢ dense_170/BiasAdd/ReadVariableOp¢dense_170/MatMul/ReadVariableOp¢ dense_171/BiasAdd/ReadVariableOp¢dense_171/MatMul/ReadVariableOpg
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
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_167/MatMulMatMulnormalization/truediv:z:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_167/ReluReludense_167/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_168/MatMul/ReadVariableOpReadVariableOp(dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_168/MatMulMatMuldense_167/Relu:activations:0'dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_168/BiasAdd/ReadVariableOpReadVariableOp)dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_168/BiasAddBiasAdddense_168/MatMul:product:0(dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_168/ReluReludense_168/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_169/MatMul/ReadVariableOpReadVariableOp(dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_169/MatMulMatMuldense_168/Relu:activations:0'dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_169/BiasAdd/ReadVariableOpReadVariableOp)dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_169/BiasAddBiasAdddense_169/MatMul:product:0(dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_169/ReluReludense_169/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_170/MatMul/ReadVariableOpReadVariableOp(dense_170_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_170/MatMulMatMuldense_169/Relu:activations:0'dense_170/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_170/BiasAdd/ReadVariableOpReadVariableOp)dense_170_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_170/BiasAddBiasAdddense_170/MatMul:product:0(dense_170/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_170/ReluReludense_170/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_171/MatMul/ReadVariableOpReadVariableOp(dense_171_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_171/MatMulMatMuldense_170/Relu:activations:0'dense_171/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_171/BiasAdd/ReadVariableOpReadVariableOp)dense_171_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_171/BiasAddBiasAdddense_171/MatMul:product:0(dense_171/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_171/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp!^dense_168/BiasAdd/ReadVariableOp ^dense_168/MatMul/ReadVariableOp!^dense_169/BiasAdd/ReadVariableOp ^dense_169/MatMul/ReadVariableOp!^dense_170/BiasAdd/ReadVariableOp ^dense_170/MatMul/ReadVariableOp!^dense_171/BiasAdd/ReadVariableOp ^dense_171/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp2D
 dense_168/BiasAdd/ReadVariableOp dense_168/BiasAdd/ReadVariableOp2B
dense_168/MatMul/ReadVariableOpdense_168/MatMul/ReadVariableOp2D
 dense_169/BiasAdd/ReadVariableOp dense_169/BiasAdd/ReadVariableOp2B
dense_169/MatMul/ReadVariableOpdense_169/MatMul/ReadVariableOp2D
 dense_170/BiasAdd/ReadVariableOp dense_170/BiasAdd/ReadVariableOp2B
dense_170/MatMul/ReadVariableOpdense_170/MatMul/ReadVariableOp2D
 dense_171/BiasAdd/ReadVariableOp dense_171/BiasAdd/ReadVariableOp2B
dense_171/MatMul/ReadVariableOpdense_171/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_168_layer_call_fn_965030

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_168_layer_call_and_return_conditional_losses_964487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
"
°
I__inference_sequential_43_layer_call_and_return_conditional_losses_964816
normalization_input
normalization_sub_y
normalization_sqrt_x"
dense_167_964790:
dense_167_964792:"
dense_168_964795:
dense_168_964797:"
dense_169_964800:
dense_169_964802:"
dense_170_964805:
dense_170_964807:"
dense_171_964810:
dense_171_964812:
identity¢!dense_167/StatefulPartitionedCall¢!dense_168/StatefulPartitionedCall¢!dense_169/StatefulPartitionedCall¢!dense_170/StatefulPartitionedCall¢!dense_171/StatefulPartitionedCallt
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
:ÿÿÿÿÿÿÿÿÿ
!dense_167/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_167_964790dense_167_964792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_167_layer_call_and_return_conditional_losses_964470
!dense_168/StatefulPartitionedCallStatefulPartitionedCall*dense_167/StatefulPartitionedCall:output:0dense_168_964795dense_168_964797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_168_layer_call_and_return_conditional_losses_964487
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_964800dense_169_964802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_169_layer_call_and_return_conditional_losses_964504
!dense_170/StatefulPartitionedCallStatefulPartitionedCall*dense_169/StatefulPartitionedCall:output:0dense_170_964805dense_170_964807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_170_layer_call_and_return_conditional_losses_964521
!dense_171/StatefulPartitionedCallStatefulPartitionedCall*dense_170/StatefulPartitionedCall:output:0dense_171_964810dense_171_964812*
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
GPU 2J 8 *N
fIRG
E__inference_dense_171_layer_call_and_return_conditional_losses_964537y
IdentityIdentity*dense_171/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp"^dense_167/StatefulPartitionedCall"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall"^dense_170/StatefulPartitionedCall"^dense_171/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall2F
!dense_170/StatefulPartitionedCall!dense_170/StatefulPartitionedCall2F
!dense_171/StatefulPartitionedCall!dense_171/StatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ð

"__inference__traced_restore_965375
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:$
assignvariableop_2_count_1:	 5
#assignvariableop_3_dense_167_kernel:/
!assignvariableop_4_dense_167_bias:5
#assignvariableop_5_dense_168_kernel:/
!assignvariableop_6_dense_168_bias:5
#assignvariableop_7_dense_169_kernel:/
!assignvariableop_8_dense_169_bias:5
#assignvariableop_9_dense_170_kernel:0
"assignvariableop_10_dense_170_bias:6
$assignvariableop_11_dense_171_kernel:0
"assignvariableop_12_dense_171_bias:'
assignvariableop_13_adam_iter:	 )
assignvariableop_14_adam_beta_1: )
assignvariableop_15_adam_beta_2: (
assignvariableop_16_adam_decay: 0
&assignvariableop_17_adam_learning_rate: #
assignvariableop_18_total: #
assignvariableop_19_count: =
+assignvariableop_20_adam_dense_167_kernel_m:7
)assignvariableop_21_adam_dense_167_bias_m:=
+assignvariableop_22_adam_dense_168_kernel_m:7
)assignvariableop_23_adam_dense_168_bias_m:=
+assignvariableop_24_adam_dense_169_kernel_m:7
)assignvariableop_25_adam_dense_169_bias_m:=
+assignvariableop_26_adam_dense_170_kernel_m:7
)assignvariableop_27_adam_dense_170_bias_m:=
+assignvariableop_28_adam_dense_171_kernel_m:7
)assignvariableop_29_adam_dense_171_bias_m:=
+assignvariableop_30_adam_dense_167_kernel_v:7
)assignvariableop_31_adam_dense_167_bias_v:=
+assignvariableop_32_adam_dense_168_kernel_v:7
)assignvariableop_33_adam_dense_168_bias_v:=
+assignvariableop_34_adam_dense_169_kernel_v:7
)assignvariableop_35_adam_dense_169_bias_v:=
+assignvariableop_36_adam_dense_170_kernel_v:7
)assignvariableop_37_adam_dense_170_bias_v:=
+assignvariableop_38_adam_dense_171_kernel_v:7
)assignvariableop_39_adam_dense_171_bias_v:
identity_41¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9§
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Í
valueÃBÀ)B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B î
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)		[
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_167_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_167_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_168_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_168_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_169_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_169_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_170_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_170_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_171_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_171_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_iterIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_2Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_decayIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp&assignvariableop_17_adam_learning_rateIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_dense_167_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_167_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_dense_168_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_168_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dense_169_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_169_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dense_170_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_170_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_dense_171_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_171_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dense_167_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_167_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_dense_168_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_168_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_dense_169_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_169_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_dense_170_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_170_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp+assignvariableop_38_adam_dense_171_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_171_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¿
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: ¬
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_41Identity_41:output:0*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ó!
£
I__inference_sequential_43_layer_call_and_return_conditional_losses_964688

inputs
normalization_sub_y
normalization_sqrt_x"
dense_167_964662:
dense_167_964664:"
dense_168_964667:
dense_168_964669:"
dense_169_964672:
dense_169_964674:"
dense_170_964677:
dense_170_964679:"
dense_171_964682:
dense_171_964684:
identity¢!dense_167/StatefulPartitionedCall¢!dense_168/StatefulPartitionedCall¢!dense_169/StatefulPartitionedCall¢!dense_170/StatefulPartitionedCall¢!dense_171/StatefulPartitionedCallg
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
:ÿÿÿÿÿÿÿÿÿ
!dense_167/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_167_964662dense_167_964664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_167_layer_call_and_return_conditional_losses_964470
!dense_168/StatefulPartitionedCallStatefulPartitionedCall*dense_167/StatefulPartitionedCall:output:0dense_168_964667dense_168_964669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_168_layer_call_and_return_conditional_losses_964487
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_964672dense_169_964674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_169_layer_call_and_return_conditional_losses_964504
!dense_170/StatefulPartitionedCallStatefulPartitionedCall*dense_169/StatefulPartitionedCall:output:0dense_170_964677dense_170_964679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_170_layer_call_and_return_conditional_losses_964521
!dense_171/StatefulPartitionedCallStatefulPartitionedCall*dense_170/StatefulPartitionedCall:output:0dense_171_964682dense_171_964684*
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
GPU 2J 8 *N
fIRG
E__inference_dense_171_layer_call_and_return_conditional_losses_964537y
IdentityIdentity*dense_171/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp"^dense_167/StatefulPartitionedCall"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall"^dense_170/StatefulPartitionedCall"^dense_171/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall2F
!dense_170/StatefulPartitionedCall!dense_170/StatefulPartitionedCall2F
!dense_171/StatefulPartitionedCall!dense_171/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_170_layer_call_fn_965070

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_170_layer_call_and_return_conditional_losses_964521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_170_layer_call_and_return_conditional_losses_965081

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

*__inference_dense_169_layer_call_fn_965050

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_169_layer_call_and_return_conditional_losses_964504o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

.__inference_sequential_43_layer_call_fn_964911

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_43_layer_call_and_return_conditional_losses_964688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
í

.__inference_sequential_43_layer_call_fn_964744
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_43_layer_call_and_return_conditional_losses_964688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 22
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
Ð2
½
I__inference_sequential_43_layer_call_and_return_conditional_losses_964956

inputs
normalization_sub_y
normalization_sqrt_x:
(dense_167_matmul_readvariableop_resource:7
)dense_167_biasadd_readvariableop_resource::
(dense_168_matmul_readvariableop_resource:7
)dense_168_biasadd_readvariableop_resource::
(dense_169_matmul_readvariableop_resource:7
)dense_169_biasadd_readvariableop_resource::
(dense_170_matmul_readvariableop_resource:7
)dense_170_biasadd_readvariableop_resource::
(dense_171_matmul_readvariableop_resource:7
)dense_171_biasadd_readvariableop_resource:
identity¢ dense_167/BiasAdd/ReadVariableOp¢dense_167/MatMul/ReadVariableOp¢ dense_168/BiasAdd/ReadVariableOp¢dense_168/MatMul/ReadVariableOp¢ dense_169/BiasAdd/ReadVariableOp¢dense_169/MatMul/ReadVariableOp¢ dense_170/BiasAdd/ReadVariableOp¢dense_170/MatMul/ReadVariableOp¢ dense_171/BiasAdd/ReadVariableOp¢dense_171/MatMul/ReadVariableOpg
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
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_167/MatMulMatMulnormalization/truediv:z:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_167/ReluReludense_167/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_168/MatMul/ReadVariableOpReadVariableOp(dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_168/MatMulMatMuldense_167/Relu:activations:0'dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_168/BiasAdd/ReadVariableOpReadVariableOp)dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_168/BiasAddBiasAdddense_168/MatMul:product:0(dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_168/ReluReludense_168/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_169/MatMul/ReadVariableOpReadVariableOp(dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_169/MatMulMatMuldense_168/Relu:activations:0'dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_169/BiasAdd/ReadVariableOpReadVariableOp)dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_169/BiasAddBiasAdddense_169/MatMul:product:0(dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_169/ReluReludense_169/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_170/MatMul/ReadVariableOpReadVariableOp(dense_170_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_170/MatMulMatMuldense_169/Relu:activations:0'dense_170/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_170/BiasAdd/ReadVariableOpReadVariableOp)dense_170_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_170/BiasAddBiasAdddense_170/MatMul:product:0(dense_170/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_170/ReluReludense_170/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_171/MatMul/ReadVariableOpReadVariableOp(dense_171_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_171/MatMulMatMuldense_170/Relu:activations:0'dense_171/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 dense_171/BiasAdd/ReadVariableOpReadVariableOp)dense_171_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_171/BiasAddBiasAdddense_171/MatMul:product:0(dense_171/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_171/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp!^dense_168/BiasAdd/ReadVariableOp ^dense_168/MatMul/ReadVariableOp!^dense_169/BiasAdd/ReadVariableOp ^dense_169/MatMul/ReadVariableOp!^dense_170/BiasAdd/ReadVariableOp ^dense_170/MatMul/ReadVariableOp!^dense_171/BiasAdd/ReadVariableOp ^dense_171/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp2D
 dense_168/BiasAdd/ReadVariableOp dense_168/BiasAdd/ReadVariableOp2B
dense_168/MatMul/ReadVariableOpdense_168/MatMul/ReadVariableOp2D
 dense_169/BiasAdd/ReadVariableOp dense_169/BiasAdd/ReadVariableOp2B
dense_169/MatMul/ReadVariableOpdense_169/MatMul/ReadVariableOp2D
 dense_170/BiasAdd/ReadVariableOp dense_170/BiasAdd/ReadVariableOp2B
dense_170/MatMul/ReadVariableOpdense_170/MatMul/ReadVariableOp2D
 dense_171/BiasAdd/ReadVariableOp dense_171/BiasAdd/ReadVariableOp2B
dense_171/MatMul/ReadVariableOpdense_171/MatMul/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
»

$__inference_signature_wrapper_964853
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_964445o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 22
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
"
°
I__inference_sequential_43_layer_call_and_return_conditional_losses_964780
normalization_input
normalization_sub_y
normalization_sqrt_x"
dense_167_964754:
dense_167_964756:"
dense_168_964759:
dense_168_964761:"
dense_169_964764:
dense_169_964766:"
dense_170_964769:
dense_170_964771:"
dense_171_964774:
dense_171_964776:
identity¢!dense_167/StatefulPartitionedCall¢!dense_168/StatefulPartitionedCall¢!dense_169/StatefulPartitionedCall¢!dense_170/StatefulPartitionedCall¢!dense_171/StatefulPartitionedCallt
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
:ÿÿÿÿÿÿÿÿÿ
!dense_167/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_167_964754dense_167_964756*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_167_layer_call_and_return_conditional_losses_964470
!dense_168/StatefulPartitionedCallStatefulPartitionedCall*dense_167/StatefulPartitionedCall:output:0dense_168_964759dense_168_964761*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_168_layer_call_and_return_conditional_losses_964487
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_964764dense_169_964766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_169_layer_call_and_return_conditional_losses_964504
!dense_170/StatefulPartitionedCallStatefulPartitionedCall*dense_169/StatefulPartitionedCall:output:0dense_170_964769dense_170_964771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_170_layer_call_and_return_conditional_losses_964521
!dense_171/StatefulPartitionedCallStatefulPartitionedCall*dense_170/StatefulPartitionedCall:output:0dense_171_964774dense_171_964776*
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
GPU 2J 8 *N
fIRG
E__inference_dense_171_layer_call_and_return_conditional_losses_964537y
IdentityIdentity*dense_171/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp"^dense_167/StatefulPartitionedCall"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall"^dense_170/StatefulPartitionedCall"^dense_171/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall2F
!dense_170/StatefulPartitionedCall!dense_170/StatefulPartitionedCall2F
!dense_171/StatefulPartitionedCall!dense_171/StatefulPartitionedCall:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ä

*__inference_dense_167_layer_call_fn_965010

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_167_layer_call_and_return_conditional_losses_964470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
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


ö
E__inference_dense_167_layer_call_and_return_conditional_losses_965021

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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
Ä

*__inference_dense_171_layer_call_fn_965090

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÚ
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
GPU 2J 8 *N
fIRG
E__inference_dense_171_layer_call_and_return_conditional_losses_964537o
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_170_layer_call_and_return_conditional_losses_964521

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 R
¼
__inference__traced_save_965245
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop&
"savev2_count_1_read_readvariableop	/
+savev2_dense_167_kernel_read_readvariableop-
)savev2_dense_167_bias_read_readvariableop/
+savev2_dense_168_kernel_read_readvariableop-
)savev2_dense_168_bias_read_readvariableop/
+savev2_dense_169_kernel_read_readvariableop-
)savev2_dense_169_bias_read_readvariableop/
+savev2_dense_170_kernel_read_readvariableop-
)savev2_dense_170_bias_read_readvariableop/
+savev2_dense_171_kernel_read_readvariableop-
)savev2_dense_171_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_dense_167_kernel_m_read_readvariableop4
0savev2_adam_dense_167_bias_m_read_readvariableop6
2savev2_adam_dense_168_kernel_m_read_readvariableop4
0savev2_adam_dense_168_bias_m_read_readvariableop6
2savev2_adam_dense_169_kernel_m_read_readvariableop4
0savev2_adam_dense_169_bias_m_read_readvariableop6
2savev2_adam_dense_170_kernel_m_read_readvariableop4
0savev2_adam_dense_170_bias_m_read_readvariableop6
2savev2_adam_dense_171_kernel_m_read_readvariableop4
0savev2_adam_dense_171_bias_m_read_readvariableop6
2savev2_adam_dense_167_kernel_v_read_readvariableop4
0savev2_adam_dense_167_bias_v_read_readvariableop6
2savev2_adam_dense_168_kernel_v_read_readvariableop4
0savev2_adam_dense_168_bias_v_read_readvariableop6
2savev2_adam_dense_169_kernel_v_read_readvariableop4
0savev2_adam_dense_169_bias_v_read_readvariableop6
2savev2_adam_dense_170_kernel_v_read_readvariableop4
0savev2_adam_dense_170_bias_v_read_readvariableop6
2savev2_adam_dense_171_kernel_v_read_readvariableop4
0savev2_adam_dense_171_bias_v_read_readvariableop
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
: ¤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*Í
valueÃBÀ)B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¿
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop"savev2_count_1_read_readvariableop+savev2_dense_167_kernel_read_readvariableop)savev2_dense_167_bias_read_readvariableop+savev2_dense_168_kernel_read_readvariableop)savev2_dense_168_bias_read_readvariableop+savev2_dense_169_kernel_read_readvariableop)savev2_dense_169_bias_read_readvariableop+savev2_dense_170_kernel_read_readvariableop)savev2_dense_170_bias_read_readvariableop+savev2_dense_171_kernel_read_readvariableop)savev2_dense_171_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_dense_167_kernel_m_read_readvariableop0savev2_adam_dense_167_bias_m_read_readvariableop2savev2_adam_dense_168_kernel_m_read_readvariableop0savev2_adam_dense_168_bias_m_read_readvariableop2savev2_adam_dense_169_kernel_m_read_readvariableop0savev2_adam_dense_169_bias_m_read_readvariableop2savev2_adam_dense_170_kernel_m_read_readvariableop0savev2_adam_dense_170_bias_m_read_readvariableop2savev2_adam_dense_171_kernel_m_read_readvariableop0savev2_adam_dense_171_bias_m_read_readvariableop2savev2_adam_dense_167_kernel_v_read_readvariableop0savev2_adam_dense_167_bias_v_read_readvariableop2savev2_adam_dense_168_kernel_v_read_readvariableop0savev2_adam_dense_168_bias_v_read_readvariableop2savev2_adam_dense_169_kernel_v_read_readvariableop0savev2_adam_dense_169_bias_v_read_readvariableop2savev2_adam_dense_170_kernel_v_read_readvariableop0savev2_adam_dense_170_bias_v_read_readvariableop2savev2_adam_dense_171_kernel_v_read_readvariableop0savev2_adam_dense_171_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *7
dtypes-
+2)		
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

identity_1Identity_1:output:0*¥
_input_shapes
: ::: ::::::::::: : : : : : : ::::::::::::::::::::: 2(
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

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::  

_output_shapes
::$! 

_output_shapes

:: "

_output_shapes
::$# 

_output_shapes

:: $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::)

_output_shapes
: 
ó!
£
I__inference_sequential_43_layer_call_and_return_conditional_losses_964544

inputs
normalization_sub_y
normalization_sqrt_x"
dense_167_964471:
dense_167_964473:"
dense_168_964488:
dense_168_964490:"
dense_169_964505:
dense_169_964507:"
dense_170_964522:
dense_170_964524:"
dense_171_964538:
dense_171_964540:
identity¢!dense_167/StatefulPartitionedCall¢!dense_168/StatefulPartitionedCall¢!dense_169/StatefulPartitionedCall¢!dense_170/StatefulPartitionedCall¢!dense_171/StatefulPartitionedCallg
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
:ÿÿÿÿÿÿÿÿÿ
!dense_167/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_167_964471dense_167_964473*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_167_layer_call_and_return_conditional_losses_964470
!dense_168/StatefulPartitionedCallStatefulPartitionedCall*dense_167/StatefulPartitionedCall:output:0dense_168_964488dense_168_964490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_168_layer_call_and_return_conditional_losses_964487
!dense_169/StatefulPartitionedCallStatefulPartitionedCall*dense_168/StatefulPartitionedCall:output:0dense_169_964505dense_169_964507*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_169_layer_call_and_return_conditional_losses_964504
!dense_170/StatefulPartitionedCallStatefulPartitionedCall*dense_169/StatefulPartitionedCall:output:0dense_170_964522dense_170_964524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dense_170_layer_call_and_return_conditional_losses_964521
!dense_171/StatefulPartitionedCallStatefulPartitionedCall*dense_170/StatefulPartitionedCall:output:0dense_171_964538dense_171_964540*
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
GPU 2J 8 *N
fIRG
E__inference_dense_171_layer_call_and_return_conditional_losses_964537y
IdentityIdentity*dense_171/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿú
NoOpNoOp"^dense_167/StatefulPartitionedCall"^dense_168/StatefulPartitionedCall"^dense_169/StatefulPartitionedCall"^dense_170/StatefulPartitionedCall"^dense_171/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2F
!dense_168/StatefulPartitionedCall!dense_168/StatefulPartitionedCall2F
!dense_169/StatefulPartitionedCall!dense_169/StatefulPartitionedCall2F
!dense_170/StatefulPartitionedCall!dense_170/StatefulPartitionedCall2F
!dense_171/StatefulPartitionedCall!dense_171/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:


ö
E__inference_dense_169_layer_call_and_return_conditional_losses_964504

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

.__inference_sequential_43_layer_call_fn_964571
normalization_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_43_layer_call_and_return_conditional_losses_964544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 22
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
È	
ö
E__inference_dense_171_layer_call_and_return_conditional_losses_965100

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_169_layer_call_and_return_conditional_losses_965061

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_168_layer_call_and_return_conditional_losses_965041

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ö
E__inference_dense_167_layer_call_and_return_conditional_losses_964470

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
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


ö
E__inference_dense_168_layer_call_and_return_conditional_losses_964487

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È	
ö
E__inference_dense_171_layer_call_and_return_conditional_losses_964537

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

.__inference_sequential_43_layer_call_fn_964882

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_43_layer_call_and_return_conditional_losses_964544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
@
Ö

!__inference__wrapped_model_964445
normalization_input%
!sequential_43_normalization_sub_y&
"sequential_43_normalization_sqrt_xH
6sequential_43_dense_167_matmul_readvariableop_resource:E
7sequential_43_dense_167_biasadd_readvariableop_resource:H
6sequential_43_dense_168_matmul_readvariableop_resource:E
7sequential_43_dense_168_biasadd_readvariableop_resource:H
6sequential_43_dense_169_matmul_readvariableop_resource:E
7sequential_43_dense_169_biasadd_readvariableop_resource:H
6sequential_43_dense_170_matmul_readvariableop_resource:E
7sequential_43_dense_170_biasadd_readvariableop_resource:H
6sequential_43_dense_171_matmul_readvariableop_resource:E
7sequential_43_dense_171_biasadd_readvariableop_resource:
identity¢.sequential_43/dense_167/BiasAdd/ReadVariableOp¢-sequential_43/dense_167/MatMul/ReadVariableOp¢.sequential_43/dense_168/BiasAdd/ReadVariableOp¢-sequential_43/dense_168/MatMul/ReadVariableOp¢.sequential_43/dense_169/BiasAdd/ReadVariableOp¢-sequential_43/dense_169/MatMul/ReadVariableOp¢.sequential_43/dense_170/BiasAdd/ReadVariableOp¢-sequential_43/dense_170/MatMul/ReadVariableOp¢.sequential_43/dense_171/BiasAdd/ReadVariableOp¢-sequential_43/dense_171/MatMul/ReadVariableOp
sequential_43/normalization/subSubnormalization_input!sequential_43_normalization_sub_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 sequential_43/normalization/SqrtSqrt"sequential_43_normalization_sqrt_x*
T0*
_output_shapes

:j
%sequential_43/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *¿Ö3­
#sequential_43/normalization/MaximumMaximum$sequential_43/normalization/Sqrt:y:0.sequential_43/normalization/Maximum/y:output:0*
T0*
_output_shapes

:®
#sequential_43/normalization/truedivRealDiv#sequential_43/normalization/sub:z:0'sequential_43/normalization/Maximum:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_43/dense_167/MatMul/ReadVariableOpReadVariableOp6sequential_43_dense_167_matmul_readvariableop_resource*
_output_shapes

:*
dtype0º
sequential_43/dense_167/MatMulMatMul'sequential_43/normalization/truediv:z:05sequential_43/dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_43/dense_167/BiasAdd/ReadVariableOpReadVariableOp7sequential_43_dense_167_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_43/dense_167/BiasAddBiasAdd(sequential_43/dense_167/MatMul:product:06sequential_43/dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_43/dense_167/ReluRelu(sequential_43/dense_167/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_43/dense_168/MatMul/ReadVariableOpReadVariableOp6sequential_43_dense_168_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_43/dense_168/MatMulMatMul*sequential_43/dense_167/Relu:activations:05sequential_43/dense_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_43/dense_168/BiasAdd/ReadVariableOpReadVariableOp7sequential_43_dense_168_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_43/dense_168/BiasAddBiasAdd(sequential_43/dense_168/MatMul:product:06sequential_43/dense_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_43/dense_168/ReluRelu(sequential_43/dense_168/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_43/dense_169/MatMul/ReadVariableOpReadVariableOp6sequential_43_dense_169_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_43/dense_169/MatMulMatMul*sequential_43/dense_168/Relu:activations:05sequential_43/dense_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_43/dense_169/BiasAdd/ReadVariableOpReadVariableOp7sequential_43_dense_169_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_43/dense_169/BiasAddBiasAdd(sequential_43/dense_169/MatMul:product:06sequential_43/dense_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_43/dense_169/ReluRelu(sequential_43/dense_169/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_43/dense_170/MatMul/ReadVariableOpReadVariableOp6sequential_43_dense_170_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_43/dense_170/MatMulMatMul*sequential_43/dense_169/Relu:activations:05sequential_43/dense_170/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_43/dense_170/BiasAdd/ReadVariableOpReadVariableOp7sequential_43_dense_170_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_43/dense_170/BiasAddBiasAdd(sequential_43/dense_170/MatMul:product:06sequential_43/dense_170/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_43/dense_170/ReluRelu(sequential_43/dense_170/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential_43/dense_171/MatMul/ReadVariableOpReadVariableOp6sequential_43_dense_171_matmul_readvariableop_resource*
_output_shapes

:*
dtype0½
sequential_43/dense_171/MatMulMatMul*sequential_43/dense_170/Relu:activations:05sequential_43/dense_171/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_43/dense_171/BiasAdd/ReadVariableOpReadVariableOp7sequential_43_dense_171_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_43/dense_171/BiasAddBiasAdd(sequential_43/dense_171/MatMul:product:06sequential_43/dense_171/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_43/dense_171/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
NoOpNoOp/^sequential_43/dense_167/BiasAdd/ReadVariableOp.^sequential_43/dense_167/MatMul/ReadVariableOp/^sequential_43/dense_168/BiasAdd/ReadVariableOp.^sequential_43/dense_168/MatMul/ReadVariableOp/^sequential_43/dense_169/BiasAdd/ReadVariableOp.^sequential_43/dense_169/MatMul/ReadVariableOp/^sequential_43/dense_170/BiasAdd/ReadVariableOp.^sequential_43/dense_170/MatMul/ReadVariableOp/^sequential_43/dense_171/BiasAdd/ReadVariableOp.^sequential_43/dense_171/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::: : : : : : : : : : 2`
.sequential_43/dense_167/BiasAdd/ReadVariableOp.sequential_43/dense_167/BiasAdd/ReadVariableOp2^
-sequential_43/dense_167/MatMul/ReadVariableOp-sequential_43/dense_167/MatMul/ReadVariableOp2`
.sequential_43/dense_168/BiasAdd/ReadVariableOp.sequential_43/dense_168/BiasAdd/ReadVariableOp2^
-sequential_43/dense_168/MatMul/ReadVariableOp-sequential_43/dense_168/MatMul/ReadVariableOp2`
.sequential_43/dense_169/BiasAdd/ReadVariableOp.sequential_43/dense_169/BiasAdd/ReadVariableOp2^
-sequential_43/dense_169/MatMul/ReadVariableOp-sequential_43/dense_169/MatMul/ReadVariableOp2`
.sequential_43/dense_170/BiasAdd/ReadVariableOp.sequential_43/dense_170/BiasAdd/ReadVariableOp2^
-sequential_43/dense_170/MatMul/ReadVariableOp-sequential_43/dense_170/MatMul/ReadVariableOp2`
.sequential_43/dense_171/BiasAdd/ReadVariableOp.sequential_43/dense_171/BiasAdd/ReadVariableOp2^
-sequential_43/dense_171/MatMul/ReadVariableOp-sequential_43/dense_171/MatMul/ReadVariableOp:e a
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
-
_user_specified_namenormalization_input:$ 

_output_shapes

::$ 

_output_shapes

:"¿L
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
	dense_1710
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¡
Ð
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
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Ó
	keras_api

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
_adapt_function"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
»
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
»
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
»
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
»
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
~
0
1
2
3
 4
'5
(6
/7
08
79
810
?11
@12"
trackable_list_wrapper
f
0
 1
'2
(3
/4
05
76
87
?8
@9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Ftrace_0
Gtrace_1
Htrace_2
Itrace_32
.__inference_sequential_43_layer_call_fn_964571
.__inference_sequential_43_layer_call_fn_964882
.__inference_sequential_43_layer_call_fn_964911
.__inference_sequential_43_layer_call_fn_964744À
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
 zFtrace_0zGtrace_1zHtrace_2zItrace_3
Ú
Jtrace_0
Ktrace_1
Ltrace_2
Mtrace_32ï
I__inference_sequential_43_layer_call_and_return_conditional_losses_964956
I__inference_sequential_43_layer_call_and_return_conditional_losses_965001
I__inference_sequential_43_layer_call_and_return_conditional_losses_964780
I__inference_sequential_43_layer_call_and_return_conditional_losses_964816À
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
 zJtrace_0zKtrace_1zLtrace_2zMtrace_3
ØBÕ
!__inference__wrapped_model_964445normalization_input"
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

Niter

Obeta_1

Pbeta_2
	Qdecay
Rlearning_ratem} m~'m(m/m0m7m8m?m@mv v'v(v/v0v7v8v?v@v"
	optimizer
,
Sserving_default"
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
Ttrace_02»
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
 zTtrace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Ztrace_02Ñ
*__inference_dense_167_layer_call_fn_965010¢
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
 zZtrace_0

[trace_02ì
E__inference_dense_167_layer_call_and_return_conditional_losses_965021¢
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
 z[trace_0
": 2dense_167/kernel
:2dense_167/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
î
atrace_02Ñ
*__inference_dense_168_layer_call_fn_965030¢
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
 zatrace_0

btrace_02ì
E__inference_dense_168_layer_call_and_return_conditional_losses_965041¢
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
 zbtrace_0
": 2dense_168/kernel
:2dense_168/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
î
htrace_02Ñ
*__inference_dense_169_layer_call_fn_965050¢
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
 zhtrace_0

itrace_02ì
E__inference_dense_169_layer_call_and_return_conditional_losses_965061¢
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
 zitrace_0
": 2dense_169/kernel
:2dense_169/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
î
otrace_02Ñ
*__inference_dense_170_layer_call_fn_965070¢
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
 zotrace_0

ptrace_02ì
E__inference_dense_170_layer_call_and_return_conditional_losses_965081¢
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
 zptrace_0
": 2dense_170/kernel
:2dense_170/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
î
vtrace_02Ñ
*__inference_dense_171_layer_call_fn_965090¢
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
 zvtrace_0

wtrace_02ì
E__inference_dense_171_layer_call_and_return_conditional_losses_965100¢
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
 zwtrace_0
": 2dense_171/kernel
:2dense_171/bias
5
0
1
2"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
'
x0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_43_layer_call_fn_964571normalization_input"À
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
Bý
.__inference_sequential_43_layer_call_fn_964882inputs"À
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
Bý
.__inference_sequential_43_layer_call_fn_964911inputs"À
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
B
.__inference_sequential_43_layer_call_fn_964744normalization_input"À
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
B
I__inference_sequential_43_layer_call_and_return_conditional_losses_964956inputs"À
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
B
I__inference_sequential_43_layer_call_and_return_conditional_losses_965001inputs"À
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
¨B¥
I__inference_sequential_43_layer_call_and_return_conditional_losses_964780normalization_input"À
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
¨B¥
I__inference_sequential_43_layer_call_and_return_conditional_losses_964816normalization_input"À
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
×BÔ
$__inference_signature_wrapper_964853normalization_input"
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
ÞBÛ
*__inference_dense_167_layer_call_fn_965010inputs"¢
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
ùBö
E__inference_dense_167_layer_call_and_return_conditional_losses_965021inputs"¢
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
ÞBÛ
*__inference_dense_168_layer_call_fn_965030inputs"¢
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
ùBö
E__inference_dense_168_layer_call_and_return_conditional_losses_965041inputs"¢
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
ÞBÛ
*__inference_dense_169_layer_call_fn_965050inputs"¢
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
ùBö
E__inference_dense_169_layer_call_and_return_conditional_losses_965061inputs"¢
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
ÞBÛ
*__inference_dense_170_layer_call_fn_965070inputs"¢
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
ùBö
E__inference_dense_170_layer_call_and_return_conditional_losses_965081inputs"¢
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
ÞBÛ
*__inference_dense_171_layer_call_fn_965090inputs"¢
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
ùBö
E__inference_dense_171_layer_call_and_return_conditional_losses_965100inputs"¢
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
N
y	variables
z	keras_api
	{total
	|count"
_tf_keras_metric
.
{0
|1"
trackable_list_wrapper
-
y	variables"
_generic_user_object
:  (2total
:  (2count
':%2Adam/dense_167/kernel/m
!:2Adam/dense_167/bias/m
':%2Adam/dense_168/kernel/m
!:2Adam/dense_168/bias/m
':%2Adam/dense_169/kernel/m
!:2Adam/dense_169/bias/m
':%2Adam/dense_170/kernel/m
!:2Adam/dense_170/bias/m
':%2Adam/dense_171/kernel/m
!:2Adam/dense_171/bias/m
':%2Adam/dense_167/kernel/v
!:2Adam/dense_167/bias/v
':%2Adam/dense_168/kernel/v
!:2Adam/dense_168/bias/v
':%2Adam/dense_169/kernel/v
!:2Adam/dense_169/bias/v
':%2Adam/dense_170/kernel/v
!:2Adam/dense_170/bias/v
':%2Adam/dense_171/kernel/v
!:2Adam/dense_171/bias/v
J
Constjtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant´
!__inference__wrapped_model_964445 '(/078?@E¢B
;¢8
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_171# 
	dense_171ÿÿÿÿÿÿÿÿÿe
__inference_adapt_step_21695E:¢7
0¢-
+(¢
 IteratorSpec 
ª "
 ¥
E__inference_dense_167_layer_call_and_return_conditional_losses_965021\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_167_layer_call_fn_965010O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_168_layer_call_and_return_conditional_losses_965041\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_168_layer_call_fn_965030O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_169_layer_call_and_return_conditional_losses_965061\/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_169_layer_call_fn_965050O/0/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_170_layer_call_and_return_conditional_losses_965081\78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_170_layer_call_fn_965070O78/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
E__inference_dense_171_layer_call_and_return_conditional_losses_965100\?@/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dense_171_layer_call_fn_965090O?@/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÔ
I__inference_sequential_43_layer_call_and_return_conditional_losses_964780 '(/078?@M¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ô
I__inference_sequential_43_layer_call_and_return_conditional_losses_964816 '(/078?@M¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
I__inference_sequential_43_layer_call_and_return_conditional_losses_964956y '(/078?@@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
I__inference_sequential_43_layer_call_and_return_conditional_losses_965001y '(/078?@@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 «
.__inference_sequential_43_layer_call_fn_964571y '(/078?@M¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ«
.__inference_sequential_43_layer_call_fn_964744y '(/078?@M¢J
C¢@
63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_43_layer_call_fn_964882l '(/078?@@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_43_layer_call_fn_964911l '(/078?@@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÎ
$__inference_signature_wrapper_964853¥ '(/078?@\¢Y
¢ 
RªO
M
normalization_input63
normalization_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_171# 
	dense_171ÿÿÿÿÿÿÿÿÿ