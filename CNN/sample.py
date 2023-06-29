import arch_sampled as arch

FGSM = [45.38, 45.59, 45.66, 45.35, 45.71, 44.35, 47.13, 45.01, 42.92, 41.54,
        42.07, 43.06, 43.51, 43.21, 44.32, 47.35, 47.87, 49.42, 48.52, 49.39,
        47.67, 48.35, 48.29, 49.70]

PGD = [42.53, 42.03, 42.39, 41.73, 42.15, 41.29, 43.57, 41.35, 39.85, 38.61,
       39.39, 40.21, 40.33, 39.86, 40.95, 43.52, 44.00, 45.48, 44.44, 45.36,
       43.97, 44.84, 44.81, 45.96]

APGD = [42.15, 41.66, 42.12, 41.36, 41.79, 40.90, 43.10, 40.92, 39.49, 38.16,
        38.84, 39.37, 39.90, 39.47, 40.49, 43.11, 43.61, 45.02, 43.97, 44.97,
        43.68, 44.42, 44.40, 45.57]

B1_normal = []
B1_reduce = []
B2_normal = []
B2_reduce = []
B_normal = []
B_reduce = []

for arch_num in range(23):

    genotype1 = eval("arch.%s" % 'arch' + str(arch_num))
    genotype2 = eval("arch.%s" % 'arch' + str(arch_num + 1))

    if FGSM[arch_num] > FGSM[arch_num + 1]:
        for node in range(4):
            if genotype1.normal[2 * node] != genotype2.normal[2 * node]:
                if genotype1.normal[2 * node] != genotype2.normal[2 * node + 1]:
                    if (genotype1.normal[2 * node], node) not in B1_normal:
                        B1_normal.append((genotype1.normal[2 * node], node))

            if genotype1.normal[2 * node + 1] != genotype2.normal[2 * node]:
                if genotype1.normal[2 * node + 1] != genotype2.normal[2 * node + 1]:
                    if (genotype1.normal[2 * node + 1], node) not in B1_normal:
                        B1_normal.append((genotype1.normal[2 * node + 1], node))

            if genotype1.reduce[2 * node] != genotype2.reduce[2 * node]:
                if genotype1.reduce[2 * node] != genotype2.reduce[2 * node + 1]:
                    if (genotype1.reduce[2 * node], node) not in B1_reduce:
                        B1_reduce.append((genotype1.reduce[2 * node], node))

            if genotype1.reduce[2 * node + 1] != genotype2.reduce[2 * node]:
                if genotype1.reduce[2 * node + 1] != genotype2.reduce[2 * node + 1]:
                    if (genotype1.reduce[2 * node + 1], node) not in B1_reduce:
                        B1_reduce.append((genotype1.reduce[2 * node + 1], node))

    if FGSM[arch_num] < FGSM[arch_num + 1]:
        for node in range(4):
            if genotype2.normal[2 * node] != genotype1.normal[2 * node]:
                if genotype2.normal[2 * node] != genotype1.normal[2 * node + 1]:
                    if (genotype2.normal[2 * node], node) not in B2_normal:
                        B2_normal.append((genotype2.normal[2 * node], node))

            if genotype2.normal[2 * node + 1] != genotype1.normal[2 * node]:
                if genotype2.normal[2 * node + 1] != genotype1.normal[2 * node + 1]:
                    if (genotype2.normal[2 * node + 1], node) not in B2_normal:
                        B2_normal.append((genotype2.normal[2 * node + 1], node))

            if genotype2.reduce[2 * node] != genotype1.reduce[2 * node]:
                if genotype2.reduce[2 * node] != genotype1.reduce[2 * node + 1]:
                    if (genotype2.reduce[2 * node], node) not in B2_reduce:
                        B2_reduce.append((genotype2.reduce[2 * node], node))

            if genotype2.reduce[2 * node + 1] != genotype1.reduce[2 * node]:
                if genotype2.reduce[2 * node + 1] != genotype1.reduce[2 * node + 1]:
                    if (genotype2.reduce[2 * node + 1], node) not in B2_reduce:
                        B2_reduce.append((genotype2.reduce[2 * node + 1], node))

print(B1_normal)
print(B1_reduce)
print(B2_normal)
print(B2_reduce)

for primitive in B1_normal:
    if primitive in B2_normal:
        B_normal.append(primitive)

for primitive in B1_reduce:
    if primitive in B2_reduce:
        B_reduce.append(primitive)

print(B_normal)
print(B_reduce)