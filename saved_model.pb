??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??

?
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_62/kernel
}
$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*&
_output_shapes
: *
dtype0
t
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_62/bias
m
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes
: *
dtype0
?
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_63/kernel
}
$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_63/bias
m
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes
:@*
dtype0
?
conv2d_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_64/kernel
}
$conv2d_64/kernel/Read/ReadVariableOpReadVariableOpconv2d_64/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_64/bias
m
"conv2d_64/bias/Read/ReadVariableOpReadVariableOpconv2d_64/bias*
_output_shapes
:@*
dtype0
z
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_30/kernel
s
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes

:@@*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:@*
dtype0
|
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_31/kernel
u
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel* 
_output_shapes
:
??*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_62/kernel/m
?
+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_62/bias/m
{
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_63/kernel/m
?
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_63/bias/m
{
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_64/kernel/m
?
+Adam/conv2d_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_64/bias/m
{
)Adam/conv2d_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_30/kernel/m
?
*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_30/bias/m
y
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/m
?
*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_62/kernel/v
?
+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_62/bias/v
{
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_63/kernel/v
?
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_63/bias/v
{
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_64/kernel/v
?
+Adam/conv2d_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_64/bias/v
{
)Adam/conv2d_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_30/kernel/v
?
*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_30/bias/v
y
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_31/kernel/v
?
*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?B
value?BB?B B?B
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
 	keras_api
h

!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
R
+trainable_variables
,	variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
R
5trainable_variables
6	variables
7regularization_losses
8	keras_api
R
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem?m?m?m?!m?"m?/m?0m?=m?>m?v?v?v?v?!v?"v?/v?0v?=v?>v?
F
0
1
2
3
!4
"5
/6
07
=8
>9
F
0
1
2
3
!4
"5
/6
07
=8
>9
 
?
	variables
trainable_variables
Hlayer_metrics
regularization_losses
Ilayer_regularization_losses

Jlayers
Kmetrics
Lnon_trainable_variables
 
\Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
Mlayer_metrics
regularization_losses
Nlayer_regularization_losses

Olayers
Pmetrics
Qnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
	variables
Rlayer_metrics
regularization_losses
Slayer_regularization_losses

Tlayers
Umetrics
Vnon_trainable_variables
 
 
 
?
trainable_variables
	variables
Wlayer_metrics
regularization_losses
Xlayer_regularization_losses

Ylayers
Zmetrics
[non_trainable_variables
\Z
VARIABLE_VALUEconv2d_64/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_64/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
#trainable_variables
$	variables
\layer_metrics
%regularization_losses
]layer_regularization_losses

^layers
_metrics
`non_trainable_variables
 
 
 
?
'trainable_variables
(	variables
alayer_metrics
)regularization_losses
blayer_regularization_losses

clayers
dmetrics
enon_trainable_variables
 
 
 
?
+trainable_variables
,	variables
flayer_metrics
-regularization_losses
glayer_regularization_losses

hlayers
imetrics
jnon_trainable_variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
1trainable_variables
2	variables
klayer_metrics
3regularization_losses
llayer_regularization_losses

mlayers
nmetrics
onon_trainable_variables
 
 
 
?
5trainable_variables
6	variables
player_metrics
7regularization_losses
qlayer_regularization_losses

rlayers
smetrics
tnon_trainable_variables
 
 
 
?
9trainable_variables
:	variables
ulayer_metrics
;regularization_losses
vlayer_regularization_losses

wlayers
xmetrics
ynon_trainable_variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
?trainable_variables
@	variables
zlayer_metrics
Aregularization_losses
{layer_regularization_losses

|layers
}metrics
~non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
F
0
1
2
3
4
5
6
7
	8

9

0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/conv2d_62/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_62/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_62_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_62_inputconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_73224
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp$conv2d_64/kernel/Read/ReadVariableOp"conv2d_64/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp+Adam/conv2d_64/kernel/m/Read/ReadVariableOp)Adam/conv2d_64/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp+Adam/conv2d_64/kernel/v/Read/ReadVariableOp)Adam/conv2d_64/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_73743
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/conv2d_64/kernel/mAdam/conv2d_64/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/vAdam/conv2d_64/kernel/vAdam/conv2d_64/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/v*3
Tin,
*2(*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_73870??
?,
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73157
conv2d_62_input)
conv2d_62_73126: 
conv2d_62_73128: )
conv2d_63_73131: @
conv2d_63_73133:@)
conv2d_64_73137:@@
conv2d_64_73139:@ 
dense_30_73144:@@
dense_30_73146:@"
dense_31_73151:
??
dense_31_73153:
identity??!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputconv2d_62_73126conv2d_62_73128*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_727702#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_73131conv2d_63_73133*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_727872#
!conv2d_63/StatefulPartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_727342"
 max_pooling2d_51/PartitionedCall?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_64_73137conv2d_64_73139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_728052#
!conv2d_64/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_727462"
 max_pooling2d_52/PartitionedCall?
dropout_41/PartitionedCallPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_728172
dropout_41/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_41/PartitionedCall:output:0dense_30_73144dense_30_73146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_728502"
 dense_30/StatefulPartitionedCall?
dropout_42/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_728612
dropout_42/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall#dropout_42/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_728692
flatten_16/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_31_73151dense_31_73153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_728822"
 dense_31/StatefulPartitionedCall?
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_73603

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_62_layer_call_and_return_conditional_losses_73438

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_16_layer_call_fn_72912
conv2d_62_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_728892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?,
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_72889

inputs)
conv2d_62_72771: 
conv2d_62_72773: )
conv2d_63_72788: @
conv2d_63_72790:@)
conv2d_64_72806:@@
conv2d_64_72808:@ 
dense_30_72851:@@
dense_30_72853:@"
dense_31_72883:
??
dense_31_72885:
identity??!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_62_72771conv2d_62_72773*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_727702#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_72788conv2d_63_72790*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_727872#
!conv2d_63/StatefulPartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_727342"
 max_pooling2d_51/PartitionedCall?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_64_72806conv2d_64_72808*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_728052#
!conv2d_64/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_727462"
 max_pooling2d_52/PartitionedCall?
dropout_41/PartitionedCallPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_728172
dropout_41/PartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#dropout_41/PartitionedCall:output:0dense_30_72851dense_30_72853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_728502"
 dense_30/StatefulPartitionedCall?
dropout_42/PartitionedCallPartitionedCall)dense_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_728612
dropout_42/PartitionedCall?
flatten_16/PartitionedCallPartitionedCall#dropout_42/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_728692
flatten_16/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_31_72883dense_31_72885*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_728822"
 dense_31/StatefulPartitionedCall?
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_52_layer_call_fn_72752

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_727462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?!
?
C__inference_dense_30_layer_call_and_return_conditional_losses_73545

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????66@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????66@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????66@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????66@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
d
E__inference_dropout_41_layer_call_and_return_conditional_losses_72981

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_62_layer_call_fn_73427

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_727702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_72746

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_62_layer_call_and_return_conditional_losses_72770

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:??????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?S
?
__inference__traced_save_73743
file_prefix/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop/
+savev2_conv2d_64_kernel_read_readvariableop-
)savev2_conv2d_64_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop6
2savev2_adam_conv2d_64_kernel_m_read_readvariableop4
0savev2_adam_conv2d_64_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop6
2savev2_adam_conv2d_64_kernel_v_read_readvariableop4
0savev2_adam_conv2d_64_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop+savev2_conv2d_64_kernel_read_readvariableop)savev2_conv2d_64_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop2savev2_adam_conv2d_64_kernel_m_read_readvariableop0savev2_adam_conv2d_64_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop2savev2_adam_conv2d_64_kernel_v_read_readvariableop0savev2_adam_conv2d_64_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@@:@:@@:@:
??:: : : : : : : : : : : : @:@:@@:@:@@:@:
??:: : : @:@:@@:@:@@:@:
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:&	"
 
_output_shapes
:
??: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
??: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@:&&"
 
_output_shapes
:
??: '

_output_shapes
::(

_output_shapes
: 
?/
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73075

inputs)
conv2d_62_73044: 
conv2d_62_73046: )
conv2d_63_73049: @
conv2d_63_73051:@)
conv2d_64_73055:@@
conv2d_64_73057:@ 
dense_30_73062:@@
dense_30_73064:@"
dense_31_73069:
??
dense_31_73071:
identity??!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?"dropout_41/StatefulPartitionedCall?"dropout_42/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_62_73044conv2d_62_73046*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_727702#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_73049conv2d_63_73051*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_727872#
!conv2d_63/StatefulPartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_727342"
 max_pooling2d_51/PartitionedCall?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_64_73055conv2d_64_73057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_728052#
!conv2d_64/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_727462"
 max_pooling2d_52/PartitionedCall?
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_729812$
"dropout_41/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_41/StatefulPartitionedCall:output:0dense_30_73062dense_30_73064*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_728502"
 dense_30/StatefulPartitionedCall?
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0#^dropout_41/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_729482$
"dropout_42/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall+dropout_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_728692
flatten_16/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_31_73069dense_31_73071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_728822"
 dense_31/StatefulPartitionedCall?
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_16_layer_call_fn_73123
conv2d_62_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_730752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?
c
E__inference_dropout_41_layer_call_and_return_conditional_losses_72817

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????66@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????66@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
d
E__inference_dropout_42_layer_call_and_return_conditional_losses_72948

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_63_layer_call_and_return_conditional_losses_73458

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_64_layer_call_and_return_conditional_losses_72805

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ll@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????ll@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_64_layer_call_fn_73467

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_728052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????ll@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????nn@
 
_user_specified_nameinputs
?
?
(__inference_dense_31_layer_call_fn_73592

inputs
unknown:
??
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_728822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_63_layer_call_and_return_conditional_losses_72787

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
-__inference_sequential_16_layer_call_fn_73249

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_728892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_42_layer_call_fn_73555

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_729482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_64_layer_call_and_return_conditional_losses_73478

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????ll@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????ll@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????nn@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????nn@
 
_user_specified_nameinputs
?!
?
C__inference_dense_30_layer_call_and_return_conditional_losses_72850

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:?????????66@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????66@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????66@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????66@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_63_layer_call_fn_73447

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_727872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_72734

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
E__inference_dropout_41_layer_call_and_return_conditional_losses_73505

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_73870
file_prefix;
!assignvariableop_conv2d_62_kernel: /
!assignvariableop_1_conv2d_62_bias: =
#assignvariableop_2_conv2d_63_kernel: @/
!assignvariableop_3_conv2d_63_bias:@=
#assignvariableop_4_conv2d_64_kernel:@@/
!assignvariableop_5_conv2d_64_bias:@4
"assignvariableop_6_dense_30_kernel:@@.
 assignvariableop_7_dense_30_bias:@6
"assignvariableop_8_dense_31_kernel:
??.
 assignvariableop_9_dense_31_bias:'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: E
+assignvariableop_19_adam_conv2d_62_kernel_m: 7
)assignvariableop_20_adam_conv2d_62_bias_m: E
+assignvariableop_21_adam_conv2d_63_kernel_m: @7
)assignvariableop_22_adam_conv2d_63_bias_m:@E
+assignvariableop_23_adam_conv2d_64_kernel_m:@@7
)assignvariableop_24_adam_conv2d_64_bias_m:@<
*assignvariableop_25_adam_dense_30_kernel_m:@@6
(assignvariableop_26_adam_dense_30_bias_m:@>
*assignvariableop_27_adam_dense_31_kernel_m:
??6
(assignvariableop_28_adam_dense_31_bias_m:E
+assignvariableop_29_adam_conv2d_62_kernel_v: 7
)assignvariableop_30_adam_conv2d_62_bias_v: E
+assignvariableop_31_adam_conv2d_63_kernel_v: @7
)assignvariableop_32_adam_conv2d_63_bias_v:@E
+assignvariableop_33_adam_conv2d_64_kernel_v:@@7
)assignvariableop_34_adam_conv2d_64_bias_v:@<
*assignvariableop_35_adam_dense_30_kernel_v:@@6
(assignvariableop_36_adam_dense_30_bias_v:@>
*assignvariableop_37_adam_dense_31_kernel_v:
??6
(assignvariableop_38_adam_dense_31_bias_v:
identity_40??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*?
value?B?(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_62_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_62_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_63_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_63_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_64_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_64_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_30_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_30_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_31_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_31_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_62_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_62_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_63_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_63_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_64_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_64_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_30_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_30_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_31_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_31_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_62_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_62_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_63_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_63_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_64_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_64_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_30_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_30_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_31_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_31_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39?
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_38AssignVariableOp_382(
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
?W
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73339

inputsB
(conv2d_62_conv2d_readvariableop_resource: 7
)conv2d_62_biasadd_readvariableop_resource: B
(conv2d_63_conv2d_readvariableop_resource: @7
)conv2d_63_biasadd_readvariableop_resource:@B
(conv2d_64_conv2d_readvariableop_resource:@@7
)conv2d_64_biasadd_readvariableop_resource:@<
*dense_30_tensordot_readvariableop_resource:@@6
(dense_30_biasadd_readvariableop_resource:@;
'dense_31_matmul_readvariableop_resource:
??6
(dense_31_biasadd_readvariableop_resource:
identity?? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp? conv2d_64/BiasAdd/ReadVariableOp?conv2d_64/Conv2D/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?!dense_30/Tensordot/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2Dinputs'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_62/BiasAdd?
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_62/Relu?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_63/Conv2D/ReadVariableOp?
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingVALID*
strides
2
conv2d_63/Conv2D?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_63/BiasAdd?
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_63/Relu?
max_pooling2d_51/MaxPoolMaxPoolconv2d_63/Relu:activations:0*/
_output_shapes
:?????????nn@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool?
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_64/Conv2D/ReadVariableOp?
conv2d_64/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@*
paddingVALID*
strides
2
conv2d_64/Conv2D?
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp?
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@2
conv2d_64/BiasAdd~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll@2
conv2d_64/Relu?
max_pooling2d_52/MaxPoolMaxPoolconv2d_64/Relu:activations:0*/
_output_shapes
:?????????66@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPool?
dropout_41/IdentityIdentity!max_pooling2d_52/MaxPool:output:0*
T0*/
_output_shapes
:?????????66@2
dropout_41/Identity?
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes?
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_30/Tensordot/free?
dense_30/Tensordot/ShapeShapedropout_41/Identity:output:0*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape?
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axis?
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2?
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis?
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const?
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod?
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1?
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1?
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axis?
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat?
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stack?
dense_30/Tensordot/transpose	Transposedropout_41/Identity:output:0"dense_30/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Tensordot/transpose?
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_30/Tensordot/Reshape?
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_30/Tensordot/MatMul?
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_30/Tensordot/Const_2?
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axis?
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1?
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Tensordot?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????66@2
dense_30/BiasAdd{
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Relu?
dropout_42/IdentityIdentitydense_30/Relu:activations:0*
T0*/
_output_shapes
:?????????66@2
dropout_42/Identityu
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_16/Const?
flatten_16/ReshapeReshapedropout_42/Identity:output:0flatten_16/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_16/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_16/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/BiasAdd|
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_31/Sigmoid?
IdentityIdentitydense_31/Sigmoid:y:0!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_42_layer_call_and_return_conditional_losses_73560

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????66@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????66@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?n
?

 __inference__wrapped_model_72728
conv2d_62_inputP
6sequential_16_conv2d_62_conv2d_readvariableop_resource: E
7sequential_16_conv2d_62_biasadd_readvariableop_resource: P
6sequential_16_conv2d_63_conv2d_readvariableop_resource: @E
7sequential_16_conv2d_63_biasadd_readvariableop_resource:@P
6sequential_16_conv2d_64_conv2d_readvariableop_resource:@@E
7sequential_16_conv2d_64_biasadd_readvariableop_resource:@J
8sequential_16_dense_30_tensordot_readvariableop_resource:@@D
6sequential_16_dense_30_biasadd_readvariableop_resource:@I
5sequential_16_dense_31_matmul_readvariableop_resource:
??D
6sequential_16_dense_31_biasadd_readvariableop_resource:
identity??.sequential_16/conv2d_62/BiasAdd/ReadVariableOp?-sequential_16/conv2d_62/Conv2D/ReadVariableOp?.sequential_16/conv2d_63/BiasAdd/ReadVariableOp?-sequential_16/conv2d_63/Conv2D/ReadVariableOp?.sequential_16/conv2d_64/BiasAdd/ReadVariableOp?-sequential_16/conv2d_64/Conv2D/ReadVariableOp?-sequential_16/dense_30/BiasAdd/ReadVariableOp?/sequential_16/dense_30/Tensordot/ReadVariableOp?-sequential_16/dense_31/BiasAdd/ReadVariableOp?,sequential_16/dense_31/MatMul/ReadVariableOp?
-sequential_16/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_16/conv2d_62/Conv2D/ReadVariableOp?
sequential_16/conv2d_62/Conv2DConv2Dconv2d_62_input5sequential_16/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2 
sequential_16/conv2d_62/Conv2D?
.sequential_16/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_16/conv2d_62/BiasAdd/ReadVariableOp?
sequential_16/conv2d_62/BiasAddBiasAdd'sequential_16/conv2d_62/Conv2D:output:06sequential_16/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2!
sequential_16/conv2d_62/BiasAdd?
sequential_16/conv2d_62/ReluRelu(sequential_16/conv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
sequential_16/conv2d_62/Relu?
-sequential_16/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_16/conv2d_63/Conv2D/ReadVariableOp?
sequential_16/conv2d_63/Conv2DConv2D*sequential_16/conv2d_62/Relu:activations:05sequential_16/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingVALID*
strides
2 
sequential_16/conv2d_63/Conv2D?
.sequential_16/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_16/conv2d_63/BiasAdd/ReadVariableOp?
sequential_16/conv2d_63/BiasAddBiasAdd'sequential_16/conv2d_63/Conv2D:output:06sequential_16/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2!
sequential_16/conv2d_63/BiasAdd?
sequential_16/conv2d_63/ReluRelu(sequential_16/conv2d_63/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
sequential_16/conv2d_63/Relu?
&sequential_16/max_pooling2d_51/MaxPoolMaxPool*sequential_16/conv2d_63/Relu:activations:0*/
_output_shapes
:?????????nn@*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_51/MaxPool?
-sequential_16/conv2d_64/Conv2D/ReadVariableOpReadVariableOp6sequential_16_conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_16/conv2d_64/Conv2D/ReadVariableOp?
sequential_16/conv2d_64/Conv2DConv2D/sequential_16/max_pooling2d_51/MaxPool:output:05sequential_16/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@*
paddingVALID*
strides
2 
sequential_16/conv2d_64/Conv2D?
.sequential_16/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp7sequential_16_conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_16/conv2d_64/BiasAdd/ReadVariableOp?
sequential_16/conv2d_64/BiasAddBiasAdd'sequential_16/conv2d_64/Conv2D:output:06sequential_16/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@2!
sequential_16/conv2d_64/BiasAdd?
sequential_16/conv2d_64/ReluRelu(sequential_16/conv2d_64/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll@2
sequential_16/conv2d_64/Relu?
&sequential_16/max_pooling2d_52/MaxPoolMaxPool*sequential_16/conv2d_64/Relu:activations:0*/
_output_shapes
:?????????66@*
ksize
*
paddingVALID*
strides
2(
&sequential_16/max_pooling2d_52/MaxPool?
!sequential_16/dropout_41/IdentityIdentity/sequential_16/max_pooling2d_52/MaxPool:output:0*
T0*/
_output_shapes
:?????????66@2#
!sequential_16/dropout_41/Identity?
/sequential_16/dense_30/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_30_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype021
/sequential_16/dense_30/Tensordot/ReadVariableOp?
%sequential_16/dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_16/dense_30/Tensordot/axes?
%sequential_16/dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_16/dense_30/Tensordot/free?
&sequential_16/dense_30/Tensordot/ShapeShape*sequential_16/dropout_41/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_16/dense_30/Tensordot/Shape?
.sequential_16/dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_30/Tensordot/GatherV2/axis?
)sequential_16/dense_30/Tensordot/GatherV2GatherV2/sequential_16/dense_30/Tensordot/Shape:output:0.sequential_16/dense_30/Tensordot/free:output:07sequential_16/dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_16/dense_30/Tensordot/GatherV2?
0sequential_16/dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_30/Tensordot/GatherV2_1/axis?
+sequential_16/dense_30/Tensordot/GatherV2_1GatherV2/sequential_16/dense_30/Tensordot/Shape:output:0.sequential_16/dense_30/Tensordot/axes:output:09sequential_16/dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_30/Tensordot/GatherV2_1?
&sequential_16/dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_30/Tensordot/Const?
%sequential_16/dense_30/Tensordot/ProdProd2sequential_16/dense_30/Tensordot/GatherV2:output:0/sequential_16/dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_30/Tensordot/Prod?
(sequential_16/dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_30/Tensordot/Const_1?
'sequential_16/dense_30/Tensordot/Prod_1Prod4sequential_16/dense_30/Tensordot/GatherV2_1:output:01sequential_16/dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_30/Tensordot/Prod_1?
,sequential_16/dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_30/Tensordot/concat/axis?
'sequential_16/dense_30/Tensordot/concatConcatV2.sequential_16/dense_30/Tensordot/free:output:0.sequential_16/dense_30/Tensordot/axes:output:05sequential_16/dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_30/Tensordot/concat?
&sequential_16/dense_30/Tensordot/stackPack.sequential_16/dense_30/Tensordot/Prod:output:00sequential_16/dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_30/Tensordot/stack?
*sequential_16/dense_30/Tensordot/transpose	Transpose*sequential_16/dropout_41/Identity:output:00sequential_16/dense_30/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????66@2,
*sequential_16/dense_30/Tensordot/transpose?
(sequential_16/dense_30/Tensordot/ReshapeReshape.sequential_16/dense_30/Tensordot/transpose:y:0/sequential_16/dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_16/dense_30/Tensordot/Reshape?
'sequential_16/dense_30/Tensordot/MatMulMatMul1sequential_16/dense_30/Tensordot/Reshape:output:07sequential_16/dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_16/dense_30/Tensordot/MatMul?
(sequential_16/dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_16/dense_30/Tensordot/Const_2?
.sequential_16/dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_30/Tensordot/concat_1/axis?
)sequential_16/dense_30/Tensordot/concat_1ConcatV22sequential_16/dense_30/Tensordot/GatherV2:output:01sequential_16/dense_30/Tensordot/Const_2:output:07sequential_16/dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_30/Tensordot/concat_1?
 sequential_16/dense_30/TensordotReshape1sequential_16/dense_30/Tensordot/MatMul:product:02sequential_16/dense_30/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????66@2"
 sequential_16/dense_30/Tensordot?
-sequential_16/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_16/dense_30/BiasAdd/ReadVariableOp?
sequential_16/dense_30/BiasAddBiasAdd)sequential_16/dense_30/Tensordot:output:05sequential_16/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????66@2 
sequential_16/dense_30/BiasAdd?
sequential_16/dense_30/ReluRelu'sequential_16/dense_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????66@2
sequential_16/dense_30/Relu?
!sequential_16/dropout_42/IdentityIdentity)sequential_16/dense_30/Relu:activations:0*
T0*/
_output_shapes
:?????????66@2#
!sequential_16/dropout_42/Identity?
sequential_16/flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2 
sequential_16/flatten_16/Const?
 sequential_16/flatten_16/ReshapeReshape*sequential_16/dropout_42/Identity:output:0'sequential_16/flatten_16/Const:output:0*
T0*)
_output_shapes
:???????????2"
 sequential_16/flatten_16/Reshape?
,sequential_16/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_16/dense_31/MatMul/ReadVariableOp?
sequential_16/dense_31/MatMulMatMul)sequential_16/flatten_16/Reshape:output:04sequential_16/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_31/MatMul?
-sequential_16/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_31/BiasAdd/ReadVariableOp?
sequential_16/dense_31/BiasAddBiasAdd'sequential_16/dense_31/MatMul:product:05sequential_16/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_16/dense_31/BiasAdd?
sequential_16/dense_31/SigmoidSigmoid'sequential_16/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_16/dense_31/Sigmoid?
IdentityIdentity"sequential_16/dense_31/Sigmoid:y:0/^sequential_16/conv2d_62/BiasAdd/ReadVariableOp.^sequential_16/conv2d_62/Conv2D/ReadVariableOp/^sequential_16/conv2d_63/BiasAdd/ReadVariableOp.^sequential_16/conv2d_63/Conv2D/ReadVariableOp/^sequential_16/conv2d_64/BiasAdd/ReadVariableOp.^sequential_16/conv2d_64/Conv2D/ReadVariableOp.^sequential_16/dense_30/BiasAdd/ReadVariableOp0^sequential_16/dense_30/Tensordot/ReadVariableOp.^sequential_16/dense_31/BiasAdd/ReadVariableOp-^sequential_16/dense_31/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2`
.sequential_16/conv2d_62/BiasAdd/ReadVariableOp.sequential_16/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_62/Conv2D/ReadVariableOp-sequential_16/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_63/BiasAdd/ReadVariableOp.sequential_16/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_63/Conv2D/ReadVariableOp-sequential_16/conv2d_63/Conv2D/ReadVariableOp2`
.sequential_16/conv2d_64/BiasAdd/ReadVariableOp.sequential_16/conv2d_64/BiasAdd/ReadVariableOp2^
-sequential_16/conv2d_64/Conv2D/ReadVariableOp-sequential_16/conv2d_64/Conv2D/ReadVariableOp2^
-sequential_16/dense_30/BiasAdd/ReadVariableOp-sequential_16/dense_30/BiasAdd/ReadVariableOp2b
/sequential_16/dense_30/Tensordot/ReadVariableOp/sequential_16/dense_30/Tensordot/ReadVariableOp2^
-sequential_16/dense_31/BiasAdd/ReadVariableOp-sequential_16/dense_31/BiasAdd/ReadVariableOp2\
,sequential_16/dense_31/MatMul/ReadVariableOp,sequential_16/dense_31/MatMul/ReadVariableOp:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?
?
(__inference_dense_30_layer_call_fn_73514

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_728502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_73583

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
a
E__inference_flatten_16_layer_call_and_return_conditional_losses_72869

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
F
*__inference_dropout_41_layer_call_fn_73483

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_728172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_51_layer_call_fn_72740

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_727342
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?/
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73191
conv2d_62_input)
conv2d_62_73160: 
conv2d_62_73162: )
conv2d_63_73165: @
conv2d_63_73167:@)
conv2d_64_73171:@@
conv2d_64_73173:@ 
dense_30_73178:@@
dense_30_73180:@"
dense_31_73185:
??
dense_31_73187:
identity??!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall?!conv2d_64/StatefulPartitionedCall? dense_30/StatefulPartitionedCall? dense_31/StatefulPartitionedCall?"dropout_41/StatefulPartitionedCall?"dropout_42/StatefulPartitionedCall?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputconv2d_62_73160conv2d_62_73162*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_727702#
!conv2d_62/StatefulPartitionedCall?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_73165conv2d_63_73167*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_727872#
!conv2d_63/StatefulPartitionedCall?
 max_pooling2d_51/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????nn@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_727342"
 max_pooling2d_51/PartitionedCall?
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_51/PartitionedCall:output:0conv2d_64_73171conv2d_64_73173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????ll@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_728052#
!conv2d_64/StatefulPartitionedCall?
 max_pooling2d_52/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_727462"
 max_pooling2d_52/PartitionedCall?
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_52/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_729812$
"dropout_41/StatefulPartitionedCall?
 dense_30/StatefulPartitionedCallStatefulPartitionedCall+dropout_41/StatefulPartitionedCall:output:0dense_30_73178dense_30_73180*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_30_layer_call_and_return_conditional_losses_728502"
 dense_30/StatefulPartitionedCall?
"dropout_42/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0#^dropout_41/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_729482$
"dropout_42/StatefulPartitionedCall?
flatten_16/PartitionedCallPartitionedCall+dropout_42/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_728692
flatten_16/PartitionedCall?
 dense_31/StatefulPartitionedCallStatefulPartitionedCall#flatten_16/PartitionedCall:output:0dense_31_73185dense_31_73187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_31_layer_call_and_return_conditional_losses_728822"
 dense_31/StatefulPartitionedCall?
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall#^dropout_42/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall2H
"dropout_42/StatefulPartitionedCall"dropout_42/StatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?
d
E__inference_dropout_42_layer_call_and_return_conditional_losses_73572

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
F
*__inference_flatten_16_layer_call_fn_73577

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatten_16_layer_call_and_return_conditional_losses_728692
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?j
?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73418

inputsB
(conv2d_62_conv2d_readvariableop_resource: 7
)conv2d_62_biasadd_readvariableop_resource: B
(conv2d_63_conv2d_readvariableop_resource: @7
)conv2d_63_biasadd_readvariableop_resource:@B
(conv2d_64_conv2d_readvariableop_resource:@@7
)conv2d_64_biasadd_readvariableop_resource:@<
*dense_30_tensordot_readvariableop_resource:@@6
(dense_30_biasadd_readvariableop_resource:@;
'dense_31_matmul_readvariableop_resource:
??6
(dense_31_biasadd_readvariableop_resource:
identity?? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp? conv2d_64/BiasAdd/ReadVariableOp?conv2d_64/Conv2D/ReadVariableOp?dense_30/BiasAdd/ReadVariableOp?!dense_30/Tensordot/ReadVariableOp?dense_31/BiasAdd/ReadVariableOp?dense_31/MatMul/ReadVariableOp?
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_62/Conv2D/ReadVariableOp?
conv2d_62/Conv2DConv2Dinputs'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
2
conv2d_62/Conv2D?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? 2
conv2d_62/BiasAdd?
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? 2
conv2d_62/Relu?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_63/Conv2D/ReadVariableOp?
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@*
paddingVALID*
strides
2
conv2d_63/Conv2D?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????@2
conv2d_63/BiasAdd?
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*1
_output_shapes
:???????????@2
conv2d_63/Relu?
max_pooling2d_51/MaxPoolMaxPoolconv2d_63/Relu:activations:0*/
_output_shapes
:?????????nn@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_51/MaxPool?
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_64/Conv2D/ReadVariableOp?
conv2d_64/Conv2DConv2D!max_pooling2d_51/MaxPool:output:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@*
paddingVALID*
strides
2
conv2d_64/Conv2D?
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp?
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????ll@2
conv2d_64/BiasAdd~
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*/
_output_shapes
:?????????ll@2
conv2d_64/Relu?
max_pooling2d_52/MaxPoolMaxPoolconv2d_64/Relu:activations:0*/
_output_shapes
:?????????66@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_52/MaxPooly
dropout_41/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_41/dropout/Const?
dropout_41/dropout/MulMul!max_pooling2d_52/MaxPool:output:0!dropout_41/dropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout_41/dropout/Mul?
dropout_41/dropout/ShapeShape!max_pooling2d_52/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_41/dropout/Shape?
/dropout_41/dropout/random_uniform/RandomUniformRandomUniform!dropout_41/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype021
/dropout_41/dropout/random_uniform/RandomUniform?
!dropout_41/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_41/dropout/GreaterEqual/y?
dropout_41/dropout/GreaterEqualGreaterEqual8dropout_41/dropout/random_uniform/RandomUniform:output:0*dropout_41/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2!
dropout_41/dropout/GreaterEqual?
dropout_41/dropout/CastCast#dropout_41/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout_41/dropout/Cast?
dropout_41/dropout/Mul_1Muldropout_41/dropout/Mul:z:0dropout_41/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout_41/dropout/Mul_1?
!dense_30/Tensordot/ReadVariableOpReadVariableOp*dense_30_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!dense_30/Tensordot/ReadVariableOp|
dense_30/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_30/Tensordot/axes?
dense_30/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_30/Tensordot/free?
dense_30/Tensordot/ShapeShapedropout_41/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_30/Tensordot/Shape?
 dense_30/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/GatherV2/axis?
dense_30/Tensordot/GatherV2GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/free:output:0)dense_30/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2?
"dense_30/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_30/Tensordot/GatherV2_1/axis?
dense_30/Tensordot/GatherV2_1GatherV2!dense_30/Tensordot/Shape:output:0 dense_30/Tensordot/axes:output:0+dense_30/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_30/Tensordot/GatherV2_1~
dense_30/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const?
dense_30/Tensordot/ProdProd$dense_30/Tensordot/GatherV2:output:0!dense_30/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod?
dense_30/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_30/Tensordot/Const_1?
dense_30/Tensordot/Prod_1Prod&dense_30/Tensordot/GatherV2_1:output:0#dense_30/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_30/Tensordot/Prod_1?
dense_30/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_30/Tensordot/concat/axis?
dense_30/Tensordot/concatConcatV2 dense_30/Tensordot/free:output:0 dense_30/Tensordot/axes:output:0'dense_30/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat?
dense_30/Tensordot/stackPack dense_30/Tensordot/Prod:output:0"dense_30/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/stack?
dense_30/Tensordot/transpose	Transposedropout_41/dropout/Mul_1:z:0"dense_30/Tensordot/concat:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Tensordot/transpose?
dense_30/Tensordot/ReshapeReshape dense_30/Tensordot/transpose:y:0!dense_30/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_30/Tensordot/Reshape?
dense_30/Tensordot/MatMulMatMul#dense_30/Tensordot/Reshape:output:0)dense_30/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_30/Tensordot/MatMul?
dense_30/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_30/Tensordot/Const_2?
 dense_30/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_30/Tensordot/concat_1/axis?
dense_30/Tensordot/concat_1ConcatV2$dense_30/Tensordot/GatherV2:output:0#dense_30/Tensordot/Const_2:output:0)dense_30/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_30/Tensordot/concat_1?
dense_30/TensordotReshape#dense_30/Tensordot/MatMul:product:0$dense_30/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Tensordot?
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp?
dense_30/BiasAddBiasAdddense_30/Tensordot:output:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????66@2
dense_30/BiasAdd{
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????66@2
dense_30/Reluy
dropout_42/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_42/dropout/Const?
dropout_42/dropout/MulMuldense_30/Relu:activations:0!dropout_42/dropout/Const:output:0*
T0*/
_output_shapes
:?????????66@2
dropout_42/dropout/Mul
dropout_42/dropout/ShapeShapedense_30/Relu:activations:0*
T0*
_output_shapes
:2
dropout_42/dropout/Shape?
/dropout_42/dropout/random_uniform/RandomUniformRandomUniform!dropout_42/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????66@*
dtype021
/dropout_42/dropout/random_uniform/RandomUniform?
!dropout_42/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_42/dropout/GreaterEqual/y?
dropout_42/dropout/GreaterEqualGreaterEqual8dropout_42/dropout/random_uniform/RandomUniform:output:0*dropout_42/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????66@2!
dropout_42/dropout/GreaterEqual?
dropout_42/dropout/CastCast#dropout_42/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????66@2
dropout_42/dropout/Cast?
dropout_42/dropout/Mul_1Muldropout_42/dropout/Mul:z:0dropout_42/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????66@2
dropout_42/dropout/Mul_1u
flatten_16/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? ? 2
flatten_16/Const?
flatten_16/ReshapeReshapedropout_42/dropout/Mul_1:z:0flatten_16/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_16/Reshape?
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_31/MatMul/ReadVariableOp?
dense_31/MatMulMatMulflatten_16/Reshape:output:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/MatMul?
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp?
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_31/BiasAdd|
dense_31/SigmoidSigmoiddense_31/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_31/Sigmoid?
IdentityIdentitydense_31/Sigmoid:y:0!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp"^dense_30/Tensordot/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2F
!dense_30/Tensordot/ReadVariableOp!dense_30/Tensordot/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_42_layer_call_fn_73550

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_42_layer_call_and_return_conditional_losses_728612
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_73224
conv2d_62_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_62_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_727282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:???????????
)
_user_specified_nameconv2d_62_input
?
c
*__inference_dropout_41_layer_call_fn_73488

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dropout_41_layer_call_and_return_conditional_losses_729812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????66@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
c
E__inference_dropout_41_layer_call_and_return_conditional_losses_73493

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????66@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????66@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?
c
E__inference_dropout_42_layer_call_and_return_conditional_losses_72861

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????66@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????66@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????66@:W S
/
_output_shapes
:?????????66@
 
_user_specified_nameinputs
?

?
C__inference_dense_31_layer_call_and_return_conditional_losses_72882

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
-__inference_sequential_16_layer_call_fn_73274

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:
??
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_16_layer_call_and_return_conditional_losses_730752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:???????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_62_inputB
!serving_default_conv2d_62_input:0???????????<
dense_310
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?S
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"?O
_tf_keras_sequential?O{"name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_62_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 224, 224, 1]}, "float32", "conv2d_62_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_62_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 16}, {"class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 23}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_62", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_62", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 224, 224, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 22}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 224, 224, 1]}}
?


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_63", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_63", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 222, 222, 32]}}
?
trainable_variables
	variables
regularization_losses
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_51", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}}
?


!kernel
"bias
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_64", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 110, 110, 64]}}
?
'trainable_variables
(	variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_52", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 27}}
?
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 12}
?

/kernel
0bias
1trainable_variables
2	variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_30", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 54, 54, 64]}}
?
5trainable_variables
6	variables
7regularization_losses
8	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout_42", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_42", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 16}
?
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_16", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 29}}
?

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_31", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 186624}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 186624]}}
?
Citer

Dbeta_1

Ebeta_2
	Fdecay
Glearning_ratem?m?m?m?!m?"m?/m?0m?=m?>m?v?v?v?v?!v?"v?/v?0v?=v?>v?"
	optimizer
f
0
1
2
3
!4
"5
/6
07
=8
>9"
trackable_list_wrapper
f
0
1
2
3
!4
"5
/6
07
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
Hlayer_metrics
regularization_losses
Ilayer_regularization_losses

Jlayers
Kmetrics
Lnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_62/kernel
: 2conv2d_62/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Mlayer_metrics
regularization_losses
Nlayer_regularization_losses

Olayers
Pmetrics
Qnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_63/kernel
:@2conv2d_63/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Rlayer_metrics
regularization_losses
Slayer_regularization_losses

Tlayers
Umetrics
Vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
	variables
Wlayer_metrics
regularization_losses
Xlayer_regularization_losses

Ylayers
Zmetrics
[non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_64/kernel
:@2conv2d_64/bias
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
?
#trainable_variables
$	variables
\layer_metrics
%regularization_losses
]layer_regularization_losses

^layers
_metrics
`non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'trainable_variables
(	variables
alayer_metrics
)regularization_losses
blayer_regularization_losses

clayers
dmetrics
enon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
,	variables
flayer_metrics
-regularization_losses
glayer_regularization_losses

hlayers
imetrics
jnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_30/kernel
:@2dense_30/bias
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
?
1trainable_variables
2	variables
klayer_metrics
3regularization_losses
llayer_regularization_losses

mlayers
nmetrics
onon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
5trainable_variables
6	variables
player_metrics
7regularization_losses
qlayer_regularization_losses

rlayers
smetrics
tnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9trainable_variables
:	variables
ulayer_metrics
;regularization_losses
vlayer_regularization_losses

wlayers
xmetrics
ynon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_31/kernel
:2dense_31/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
@	variables
zlayer_metrics
Aregularization_losses
{layer_regularization_losses

|layers
}metrics
~non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 31}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}, "shared_object_id": 23}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/conv2d_62/kernel/m
!: 2Adam/conv2d_62/bias/m
/:- @2Adam/conv2d_63/kernel/m
!:@2Adam/conv2d_63/bias/m
/:-@@2Adam/conv2d_64/kernel/m
!:@2Adam/conv2d_64/bias/m
&:$@@2Adam/dense_30/kernel/m
 :@2Adam/dense_30/bias/m
(:&
??2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
/:- 2Adam/conv2d_62/kernel/v
!: 2Adam/conv2d_62/bias/v
/:- @2Adam/conv2d_63/kernel/v
!:@2Adam/conv2d_63/bias/v
/:-@@2Adam/conv2d_64/kernel/v
!:@2Adam/conv2d_64/bias/v
&:$@@2Adam/dense_30/kernel/v
 :@2Adam/dense_30/bias/v
(:&
??2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
?2?
 __inference__wrapped_model_72728?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0
conv2d_62_input???????????
?2?
-__inference_sequential_16_layer_call_fn_72912
-__inference_sequential_16_layer_call_fn_73249
-__inference_sequential_16_layer_call_fn_73274
-__inference_sequential_16_layer_call_fn_73123?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73339
H__inference_sequential_16_layer_call_and_return_conditional_losses_73418
H__inference_sequential_16_layer_call_and_return_conditional_losses_73157
H__inference_sequential_16_layer_call_and_return_conditional_losses_73191?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_conv2d_62_layer_call_fn_73427?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_62_layer_call_and_return_conditional_losses_73438?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_63_layer_call_fn_73447?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_63_layer_call_and_return_conditional_losses_73458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_51_layer_call_fn_72740?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_72734?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2d_64_layer_call_fn_73467?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_64_layer_call_and_return_conditional_losses_73478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_52_layer_call_fn_72752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_72746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_dropout_41_layer_call_fn_73483
*__inference_dropout_41_layer_call_fn_73488?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_41_layer_call_and_return_conditional_losses_73493
E__inference_dropout_41_layer_call_and_return_conditional_losses_73505?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dense_30_layer_call_fn_73514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_30_layer_call_and_return_conditional_losses_73545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dropout_42_layer_call_fn_73550
*__inference_dropout_42_layer_call_fn_73555?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dropout_42_layer_call_and_return_conditional_losses_73560
E__inference_dropout_42_layer_call_and_return_conditional_losses_73572?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_flatten_16_layer_call_fn_73577?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_16_layer_call_and_return_conditional_losses_73583?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_31_layer_call_fn_73592?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_31_layer_call_and_return_conditional_losses_73603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_73224conv2d_62_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_72728?
!"/0=>B??
8?5
3?0
conv2d_62_input???????????
? "3?0
.
dense_31"?
dense_31??????????
D__inference_conv2d_62_layer_call_and_return_conditional_losses_73438p9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_62_layer_call_fn_73427c9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_63_layer_call_and_return_conditional_losses_73458p9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????@
? ?
)__inference_conv2d_63_layer_call_fn_73447c9?6
/?,
*?'
inputs??????????? 
? ""????????????@?
D__inference_conv2d_64_layer_call_and_return_conditional_losses_73478l!"7?4
-?*
(?%
inputs?????????nn@
? "-?*
#? 
0?????????ll@
? ?
)__inference_conv2d_64_layer_call_fn_73467_!"7?4
-?*
(?%
inputs?????????nn@
? " ??????????ll@?
C__inference_dense_30_layer_call_and_return_conditional_losses_73545l/07?4
-?*
(?%
inputs?????????66@
? "-?*
#? 
0?????????66@
? ?
(__inference_dense_30_layer_call_fn_73514_/07?4
-?*
(?%
inputs?????????66@
? " ??????????66@?
C__inference_dense_31_layer_call_and_return_conditional_losses_73603^=>1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? }
(__inference_dense_31_layer_call_fn_73592Q=>1?.
'?$
"?
inputs???????????
? "???????????
E__inference_dropout_41_layer_call_and_return_conditional_losses_73493l;?8
1?.
(?%
inputs?????????66@
p 
? "-?*
#? 
0?????????66@
? ?
E__inference_dropout_41_layer_call_and_return_conditional_losses_73505l;?8
1?.
(?%
inputs?????????66@
p
? "-?*
#? 
0?????????66@
? ?
*__inference_dropout_41_layer_call_fn_73483_;?8
1?.
(?%
inputs?????????66@
p 
? " ??????????66@?
*__inference_dropout_41_layer_call_fn_73488_;?8
1?.
(?%
inputs?????????66@
p
? " ??????????66@?
E__inference_dropout_42_layer_call_and_return_conditional_losses_73560l;?8
1?.
(?%
inputs?????????66@
p 
? "-?*
#? 
0?????????66@
? ?
E__inference_dropout_42_layer_call_and_return_conditional_losses_73572l;?8
1?.
(?%
inputs?????????66@
p
? "-?*
#? 
0?????????66@
? ?
*__inference_dropout_42_layer_call_fn_73550_;?8
1?.
(?%
inputs?????????66@
p 
? " ??????????66@?
*__inference_dropout_42_layer_call_fn_73555_;?8
1?.
(?%
inputs?????????66@
p
? " ??????????66@?
E__inference_flatten_16_layer_call_and_return_conditional_losses_73583b7?4
-?*
(?%
inputs?????????66@
? "'?$
?
0???????????
? ?
*__inference_flatten_16_layer_call_fn_73577U7?4
-?*
(?%
inputs?????????66@
? "?????????????
K__inference_max_pooling2d_51_layer_call_and_return_conditional_losses_72734?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_51_layer_call_fn_72740?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_52_layer_call_and_return_conditional_losses_72746?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_52_layer_call_fn_72752?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_sequential_16_layer_call_and_return_conditional_losses_73157
!"/0=>J?G
@?=
3?0
conv2d_62_input???????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73191
!"/0=>J?G
@?=
3?0
conv2d_62_input???????????
p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73339v
!"/0=>A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_16_layer_call_and_return_conditional_losses_73418v
!"/0=>A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
-__inference_sequential_16_layer_call_fn_72912r
!"/0=>J?G
@?=
3?0
conv2d_62_input???????????
p 

 
? "???????????
-__inference_sequential_16_layer_call_fn_73123r
!"/0=>J?G
@?=
3?0
conv2d_62_input???????????
p

 
? "???????????
-__inference_sequential_16_layer_call_fn_73249i
!"/0=>A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
-__inference_sequential_16_layer_call_fn_73274i
!"/0=>A?>
7?4
*?'
inputs???????????
p

 
? "???????????
#__inference_signature_wrapper_73224?
!"/0=>U?R
? 
K?H
F
conv2d_62_input3?0
conv2d_62_input???????????"3?0
.
dense_31"?
dense_31?????????