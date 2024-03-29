import os


num_encoder_layers = 48
num_decoder_layers = 12
num_layers = {
    "encoder": 48,
    "decoder": 12,
}

expert_size = 41971712
sparse_step = 2
num_experts = 128
path_prefix = "/data/ft/switch-xxl-128-fp4-random"

buffer = bytearray(expert_size)

for coder, num_layers in num_layers.items():
    for i in range(num_layers):
        if i % sparse_step == 1:
            for e in range(num_experts):
                name = os.path.join(path_prefix, f"{coder}::layer{i}expert{e}.bin")
                with open(name, "wb") as fp:
                    fp.write(buffer)
        else:
            name = os.path.join(path_prefix, f"{coder}::layer{i}.bin")
            with open(name, "wb") as fp:
                fp.write(buffer)
