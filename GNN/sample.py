import arch_sampled as arch
import time

#the adversarial robustness of the sampled architectures
robustness = [81.204, 81.76, 80.464, 82.914, 82.506,
              82.578, 80.946, 80.574, 78.904, 78.422,
              81.54, 81.24, 81.096]

B1 = []
B2 = []
B = []

for arch_index in range(12):
    genotype1 = eval("arch.%s" % 'arch' + str(arch_index))
    genotype2 = eval("arch.%s" % 'arch' + str(arch_index + 1))
    primitives1 = genotype1.split('||')
    primitives2 = genotype2.split('||')
    if robustness[arch_index] > robustness[arch_index + 1]:
        for p in range(7):
            if primitives1[p] != primitives2[p]:
                if (primitives1[p], p) not in B1:
                    B1.append((primitives1[p], p))
    elif robustness[arch_index] < robustness[arch_index + 1]:
        for p in range(7):
            if primitives2[p] != primitives1[p]:
                if(primitives2[p], p) not in B2:
                    B2.append((primitives2[p], p))

for pri in B1:
    if pri in B2:
        B.append(pri)
print(B)
