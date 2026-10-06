#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 16-bit transpose, loading/storing a 4x4 tile of elements
kernel void kernel_transpose_16(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;

    half4 temp0 = read_imageh(input, (j_2+0)*cols+i);
    half4 temp1 = read_imageh(input, (j_2+1)*cols+i);
    half4 temp2 = read_imageh(input, (j_2+2)*cols+i);
    half4 temp3 = read_imageh(input, (j_2+3)*cols+i);

    write_imageh(output, (i_2+0)*rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
    write_imageh(output, (i_2+1)*rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imageh(output, (i_2+2)*rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imageh(output, (i_2+3)*rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
}

// Padded kernel for irregular shape
kernel void kernel_transpose_16_4x1(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int j_2 = j << 2;

    half temp0 = read_imageh(input, (j_2 + 0) * cols + i).x;
    half temp1 = read_imageh(input, (j_2 + 1) * cols + i).x;
    half temp2 = read_imageh(input, (j_2 + 2) * cols + i).x;
    half temp3 = read_imageh(input, (j_2 + 3) * cols + i).x;

    write_imageh(output, i * rows + j, (half4)(temp0, temp1, temp2, temp3));
}

// Buffer transposes, staged through local memory in 32x32 tiles.
//
// The direct form `output[x*ldo + y] = input[y*ldi + x]` reads coalesced but writes
// with a stride of ldo, so each wavefront issues 64 scattered stores and every one
// pulls a whole cache line. Staging the tile in local memory makes both the loads
// and the stores contiguous; the tile is padded to 33 columns so the transposed
// local reads do not all land on one bank.
//
// Launch geometry (see transpose_launch_dims): local {32, 8}, global
// {round_up(ldi, 32), round_up(ldo, 32) / 4}, i.e. four elements per work-item.

kernel void kernel_transpose_8_buf(
    global const uchar * input,
    global uchar * output,
    const int ldi,
    const int ldo
) {
    __local uchar tile[32][33];

    const int bx = get_group_id(0) * 32;
    const int by = get_group_id(1) * 32;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    for (int k = 0; k < 32; k += 8) {
        if (bx + tx < ldi && by + ty + k < ldo) {
            tile[ty + k][tx] = input[(by + ty + k)*ldi + bx + tx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < 32; k += 8) {
        if (by + tx < ldo && bx + ty + k < ldi) {
            output[(bx + ty + k)*ldo + by + tx] = tile[tx][ty + k];
        }
    }
}

kernel void kernel_transpose_16_buf(
    global const ushort * input,
    global ushort * output,
    const int ldi,
    const int ldo
) {
    __local ushort tile[32][33];

    const int bx = get_group_id(0) * 32;
    const int by = get_group_id(1) * 32;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    for (int k = 0; k < 32; k += 8) {
        if (bx + tx < ldi && by + ty + k < ldo) {
            tile[ty + k][tx] = input[(by + ty + k)*ldi + bx + tx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < 32; k += 8) {
        if (by + tx < ldo && bx + ty + k < ldi) {
            output[(bx + ty + k)*ldo + by + tx] = tile[tx][ty + k];
        }
    }
}

kernel void kernel_transpose_32_buf(
    global const uint * input,
    global uint * output,
    const int ldi,
    const int ldo
) {
    __local uint tile[32][33];

    const int bx = get_group_id(0) * 32;
    const int by = get_group_id(1) * 32;
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    for (int k = 0; k < 32; k += 8) {
        if (bx + tx < ldi && by + ty + k < ldo) {
            tile[ty + k][tx] = input[(by + ty + k)*ldi + bx + tx];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < 32; k += 8) {
        if (by + tx < ldo && bx + ty + k < ldi) {
            output[(bx + ty + k)*ldo + by + tx] = tile[tx][ty + k];
        }
    }
}

// 32-bit transpose, loading/storing a 4x4 tile of elements
kernel void kernel_transpose_32(
    __read_only image1d_buffer_t input,
    __write_only image1d_buffer_t output,
    const uint rows,
    const uint cols
) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;

    float4 temp0 = read_imagef(input, (j_2+0)*cols+i);
    float4 temp1 = read_imagef(input, (j_2+1)*cols+i);
    float4 temp2 = read_imagef(input, (j_2+2)*cols+i);
    float4 temp3 = read_imagef(input, (j_2+3)*cols+i);

    write_imagef(output, (i_2+0)*rows+j, (float4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0));
    write_imagef(output, (i_2+1)*rows+j, (float4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imagef(output, (i_2+2)*rows+j, (float4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imagef(output, (i_2+3)*rows+j, (float4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));

}

// 32-bit transpose, loading/storing a 4x4 tile of elements
// Only used for activations
// converts to FP16
// also adds zero padding for non multiple of 8 prompt lengths
kernel void kernel_transpose_32_16(__read_only image1d_buffer_t input, __write_only image1d_buffer_t output, const uint rows, const uint cols, const uint padded_rows) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;
    half4 temp0 = {0,0,0,0}; // initialize outputs to 0
    half4 temp1 = {0,0,0,0};
    half4 temp2 = {0,0,0,0};
    half4 temp3 = {0,0,0,0};

    if((j_2+0)*cols+i*4+3 < rows*cols*16){ // only load from a valid location. Otherwise keep register data as 0
        temp0 = read_imageh(input, (j_2+0)*cols+i);
    }
    if((j_2+1)*cols+i*4+3 < rows*cols*16){
        temp1 = read_imageh(input, (j_2+1)*cols+i);
    }
    if((j_2+2)*cols+i*4+3 < rows*cols*16){
        temp2 = read_imageh(input, (j_2+2)*cols+i);
    }
    if((j_2+3)*cols+i*4+3 < rows*cols*16){
        temp3 = read_imageh(input, (j_2+3)*cols+i);
    }

    write_imageh(output, (i_2+0)*padded_rows+j, (half4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0)); // no conditionals for output, includes zero padding
    write_imageh(output, (i_2+1)*padded_rows+j, (half4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imageh(output, (i_2+2)*padded_rows+j, (half4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imageh(output, (i_2+3)*padded_rows+j, (half4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
}

// F32 twin of kernel_transpose_32_16 for GGML_PREC_F32 callers: identical indexing and
// zero padding, but keeps the activations in float instead of narrowing them to half.
kernel void kernel_transpose_32_32(__read_only image1d_buffer_t input, __write_only image1d_buffer_t output, const uint rows, const uint cols, const uint padded_rows) {

    const int i = get_global_id(0);
    const int j = get_global_id(1);
    const int i_2 = i<<2;
    const int j_2 = j<<2;
    float4 temp0 = {0,0,0,0}; // initialize outputs to 0
    float4 temp1 = {0,0,0,0};
    float4 temp2 = {0,0,0,0};
    float4 temp3 = {0,0,0,0};

    if((j_2+0)*cols+i*4+3 < rows*cols*16){ // only load from a valid location. Otherwise keep register data as 0
        temp0 = read_imagef(input, (j_2+0)*cols+i);
    }
    if((j_2+1)*cols+i*4+3 < rows*cols*16){
        temp1 = read_imagef(input, (j_2+1)*cols+i);
    }
    if((j_2+2)*cols+i*4+3 < rows*cols*16){
        temp2 = read_imagef(input, (j_2+2)*cols+i);
    }
    if((j_2+3)*cols+i*4+3 < rows*cols*16){
        temp3 = read_imagef(input, (j_2+3)*cols+i);
    }

    write_imagef(output, (i_2+0)*padded_rows+j, (float4)(temp0.s0, temp1.s0, temp2.s0, temp3.s0)); // no conditionals for output, includes zero padding
    write_imagef(output, (i_2+1)*padded_rows+j, (float4)(temp0.s1, temp1.s1, temp2.s1, temp3.s1));
    write_imagef(output, (i_2+2)*padded_rows+j, (float4)(temp0.s2, temp1.s2, temp2.s2, temp3.s2));
    write_imagef(output, (i_2+3)*padded_rows+j, (float4)(temp0.s3, temp1.s3, temp2.s3, temp3.s3));
}
