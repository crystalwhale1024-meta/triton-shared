// RUN: triton-shared-opt --triton-to-structured --split-input-file %s | FileCheck %s

// These tests check that loads/stores that exhibit a cmp ge against 0 work
// correctly with the pointer analysis pass

// Example of the triton kernel that generates the loads/stores with cmp ge 0.
// The boundary_check fields of the load/stores, along with preprocessing the
// kernel through --triton-rewrite-tensor-pointer before calling the
// --triton-to-structured pass results in those cmp ge 0 instructions.
//
//  def kernel(in_ptr0, out_ptr0, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
//     yoffset = tl.program_id(1) * YBLOCK
//     xoffset = tl.program_id(0) * XBLOCK
//     tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[16640, 10],
//                     strides=[1, 16640], block_shape=[XBLOCK, YBLOCK],
//                     order=[1, 0], offsets=[xoffset, yoffset]),
//                     boundary_check=[0, 1])
//     tl.store(tl.make_block_ptr(out_ptr0, shape=[16640, 10],
//                     strides=[1, 16640], block_shape=[XBLOCK, YBLOCK],
//                     order=[1, 0], offsets=[xoffset, yoffset]),
//                     tl.broadcast_to(tmp0, [XBLOCK, YBLOCK]).to(tl.float16),
//                     boundary_check=[0, 1])

tt.func public @test_masked_load(%arg0: !tt.ptr<f16>) -> tensor<16x16xf16> {
  %cst = arith.constant dense<0> : tensor<1x16xi64>
  %c16_i32 = arith.constant 16 : i32
  %0 = tt.get_program_id y : i32
  %1 = arith.muli %0, %c16_i32 : i32
  %2 = arith.extsi %1 : i32 to i64
  %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %5 = arith.extsi %4 : tensor<16xi32> to tensor<16xi64>
  %6 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi64> -> tensor<16x16xi64>
  %8 = tt.addptr %3, %7 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %9 = tt.splat %2 : i64 -> tensor<16xi64>
  %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %11 = arith.extsi %10 : tensor<16xi32> to tensor<16xi64>
  %12 = arith.addi %9, %11 : tensor<16xi64>
  %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %14 = arith.cmpi sge, %13, %cst : tensor<1x16xi64>
  %15 = tt.broadcast %14 : tensor<1x16xi1> -> tensor<16x16xi1>
  %16 = tt.load %8, %15 evictionPolicy = evict_last : tensor<16x16x!tt.ptr<f16>>
  tt.return %16 : tensor<16x16xf16>
}

// CHECK:         tt.func public @test_masked_load([[arg0_:%.+]]: !tt.ptr<f16>) -> tensor<16x16xf16> {
// CHECK:           [[VAR_0_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [16, 16], strides: [1, 0], offsets: [0, 0], shape: [0, 0], order: [] : <f16> to tensor<16x16x!tt.ptr<f16>>
// CHECK:           [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64: 16, 16>}> : (tensor<16x16x!tt.ptr<f16>>) -> tensor<16x16xf16>
// CHECK:         }

// -----

tt.func public @test_masked_store(%arg0: !tt.ptr<f16>) {
  %cst = arith.constant dense<0> : tensor<16x1xi64>
  %cst_0 = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
  %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %2 = arith.extsi %1 : tensor<16xi32> to tensor<16xi64>
  %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %4 = tt.broadcast %3 : tensor<16x1xi64> -> tensor<16x16xi64>
  %5 = tt.addptr %0, %4 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>
  %6 = arith.cmpi sge, %3, %cst : tensor<16x1xi64>
  %7 = tt.broadcast %6 : tensor<16x1xi1> -> tensor<16x16xi1>
  tt.store %5, %cst_0, %7 : tensor<16x16x!tt.ptr<f16>>
  tt.return
}

// CHECK:         tt.func public @test_masked_store([[arg0_:%.+]]: !tt.ptr<f16>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.500000e+01> : tensor<16x16xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [16, 16], strides: [1, 0], offsets: [0, 0], shape: [0, 0], order: [] : <f16> to tensor<16x16x!tt.ptr<f16>>
// CHECK:           "tts.store"([[VAR_0_]], [[VAR_cst_]]) <{static_mask_dims = array<i64: 16, 16>}> : (tensor<16x16x!tt.ptr<f16>>, tensor<16x16xf16>) -> ()
// CHECK:         }

// -----

// A range shifted by a negative constant can be below 0, so `>= 0` must keep
// the load masked instead of being treated as always true.
tt.func public @test_negative_shift_lower_bound(%arg0: !tt.ptr<f32>) -> tensor<4xf32> {
  %cst_neg1 = arith.constant dense<-1> : tensor<4xi32>
  %cst_0 = arith.constant dense<0> : tensor<4xi32>
  %cst_4 = arith.constant dense<4> : tensor<4xi32>
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = arith.addi %0, %cst_neg1 : tensor<4xi32>
  %2 = arith.cmpi sge, %1, %cst_0 : tensor<4xi32>
  %3 = arith.cmpi slt, %1, %cst_4 : tensor<4xi32>
  %4 = arith.andi %2, %3 : tensor<4xi1>
  %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %6 = tt.addptr %5, %1 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %7 = tt.load %6, %4 : tensor<4x!tt.ptr<f32>>
  tt.return %7 : tensor<4xf32>
}

// CHECK-LABEL:   tt.func public @test_negative_shift_lower_bound
// CHECK-NOT:       tts.load
// CHECK:           tt.load {{.*}}, {{.*}} : tensor<4x!tt.ptr<f32>>

// -----

// A range shifted by -1 and then offset again, as Inductor emits for padded
// pooling windows, can be negative, so `>= 0` must keep the load masked.
tt.func public @test_nested_negative_shift_lower_bound(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<4xf32> {
  %cst_neg1 = arith.constant dense<-1> : tensor<4xi32>
  %cst_0 = arith.constant dense<0> : tensor<4xi32>
  %cst_6 = arith.constant dense<6> : tensor<4xi32>
  %c2_i32 = arith.constant 2 : i32
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = arith.addi %cst_neg1, %0 : tensor<4xi32>
  %2 = arith.muli %arg1, %c2_i32 : i32
  %3 = tt.splat %2 : i32 -> tensor<4xi32>
  %4 = arith.addi %1, %3 : tensor<4xi32>
  %5 = arith.cmpi sge, %4, %cst_0 : tensor<4xi32>
  %6 = arith.cmpi slt, %4, %cst_6 : tensor<4xi32>
  %7 = arith.andi %5, %6 : tensor<4xi1>
  %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
  %9 = tt.addptr %8, %4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
  %10 = tt.load %9, %7 : tensor<4x!tt.ptr<f32>>
  tt.return %10 : tensor<4xf32>
}

// CHECK-LABEL:   tt.func public @test_nested_negative_shift_lower_bound
// CHECK-NOT:       tts.load
// CHECK:           tt.load {{.*}}, {{.*}} : tensor<4x!tt.ptr<f32>>
