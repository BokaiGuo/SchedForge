module {
  func.func @decoder_layer_moe(
      %x: tensor<4x16xf32>,
      %rms1: tensor<16xf32>,
      %wq: tensor<16x16xf32>,
      %wk: tensor<16x8xf32>,
      %wv: tensor<16x8xf32>,
      %wo: tensor<16x16xf32>,
      %rms2: tensor<16xf32>,
      %router: tensor<16x4xf32>) -> tensor<4x16xf32> {
    %n1 = stablehlo.custom_call @RmsNorm(%x, %rms1) : tensor<4x16xf32>
    %q = stablehlo.dot_general %n1, %wq : tensor<4x16xf32>
    %k = stablehlo.dot_general %n1, %wk : tensor<4x8xf32>
    %v = stablehlo.dot_general %n1, %wv : tensor<4x8xf32>
    %rq = stablehlo.custom_call @RoPE(%q) : tensor<4x16xf32>
    %rk = stablehlo.custom_call @RoPE(%k) : tensor<4x8xf32>
    %o = stablehlo.dot_general %rq, %wo : tensor<4x16xf32>
    %r1 = stablehlo.add %o, %x : tensor<4x16xf32>
    %n2 = stablehlo.custom_call @RmsNorm(%r1, %rms2) : tensor<4x16xf32>
    %router_logits = stablehlo.dot_general %n2, %router : tensor<4x4xf32>
    %moe = stablehlo.custom_call @MoeFFN(%n2, %router_logits) : tensor<4x16xf32>
    %out = stablehlo.add %moe, %r1 : tensor<4x16xf32>
    return %out
  }
}
