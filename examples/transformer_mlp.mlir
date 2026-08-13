module {
  func.func @main(%x: tensor<B*Sx64xf32>, %w1: tensor<64x128xf32>, %b1: tensor<128xf32>, %w2: tensor<128x64xf32>, %b2: tensor<64xf32>) -> tensor<B*Sx64xf32> {
    %0 = stablehlo.dot_general %x, %w1 : tensor<B*Sx128xf32>
    %1 = stablehlo.add %0, %b1 : tensor<B*Sx128xf32>
    %2 = stablehlo.custom_call @Gelu(%1) : tensor<B*Sx128xf32>
    %3 = stablehlo.dot_general %2, %w2 : tensor<B*Sx64xf32>
    %4 = stablehlo.add %3, %b2 : tensor<B*Sx64xf32>
    %5 = stablehlo.add %4, %x : tensor<B*Sx64xf32>
    return %5 : tensor<B*Sx64xf32>
  }
}
