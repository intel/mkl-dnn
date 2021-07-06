# Proposal for enabling int8 as a destination type for float convolution

## Introduction

This feature was originally requested by the OpenVINO team in order to improve
performance in benchmarks that include topologies which are partially quantized.
to prevent a drop in accuracy in the quantized portion in comparrison to the 
original model. The proposed feature covers these cases in relevant topologies 
with the pattern: 

\f[
    \dst_{int8}[:] =
        downconvert\_f32\_to\_int8(
            output\_scale \cdot
            conv_{f32}(\src_{f32}, \weights_{f32})
        ),
\f]

Currently this is accomplished by following a float convolution primitive with 
a binary primitive that performs conversion to int8 type. 

This feature is meant to reduce the overhead added in all similar cases, 
eliminating the additional kernel execution. By enabling conversion of the 
results of float convolution to int8 overhead can be reduced. 

## Proposal

###API

This feature alters the existing API by enabling float convolution with int8 
destination. In this and any other mixed convolution cases, the type of the 
computation will be the same as the source and weights tensors.

Use of int8 destination will be supported by the addition of user provided 
output scale and zero point to the float convolution implementations that 
support int8 destination. Scaling will be applied to the float results of 
computation prior to downconversion and the application of post-ops. 

This will further alter the existing API by permitting output scale for 
convolution with float source and weights tensors. In cases where the 
destination type is int8 output scale will be enabled. 

###Implementation

The proposed implementation uses existing float convolution kernels, adding an 
alternative implementation for reading and writing to destination. Kernels with 
this feature will perform standard float convolution with the input and weights 
tensors. When int8 destination is specified the output tensor will be int8 type 
and blocked in an optimal layout for int8 tensors. 

This will require adding the following cfgs to benchdnn to allow testing: 

- f16f16s8
- f32f32s8

Testing will be modified to use a range of values based on destination data type
when it is smaller in size to limit cases where results are saturated. 

The majority of the changes will affect the destination offset calculation and 
any the pattern of reads and writes to account for the different blocked 
layouts used when int8 destination is selcted. 

The proposed implementation will add support for user provided output scales 
and zero point to allow control over conversion from float results to int8. 

## Open Questions


