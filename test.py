import glob

#lora = sorted(glob.glob('./experiment/gpt*-lora_lr1e-3*.txt'))
#slora = sorted(glob.glob('./experiment/gpt*-slora_lr1e-3*.txt'))

#lora = sorted(glob.glob('./experiment/gptj-6b-lora-math-all-r32_*.txt'))
#slora = sorted(glob.glob('./experiment/gptj-6b-slora-math-all-r32_*.txt'))

#model = 'gpt'
model = 'llama3'

dataset = 'metamath'
#dataset = 'commonsense_170k'

rank = 128

lr= 0.0002
seed=1

pos2 = 1
if dataset == 'metamath':
    pos2 = 2

if dataset == 'commonsense_170k':
    lora = sorted(glob.glob(f'./experiment/{model}_{dataset}_lora_r{rank}_lr{lr}_seed{seed}*result.txt'))
    #slora = sorted(glob.glob(f'./experiment/{model}_{dataset}_slora_r{rank}_lr{lr}_seed{seed}*result.txt'))
    plora = sorted(glob.glob(f'./experiment/{model}_{dataset}_lora+pissa_r{rank}_lr{lr}_seed{seed}*result.txt'))
    lmclora = sorted(glob.glob(f'./experiment/{model}_{dataset}_loralmc_r{rank}_lr{lr}_seed{seed}*result.txt'))
    #pslora = sorted(glob.glob(f'./experiment/{model}_{dataset}_slora+pissa_r{rank}_lr{lr}_seed{seed}*result.txt'))
    pos = 2
else:
    lora = sorted(glob.glob(f'./experiment/{model}_{dataset}_lora_r{rank}_lr{lr}_seed{seed}*.txt'))
    #slora = sorted(glob.glob(f'./experiment/{model}_{dataset}_slora_r{rank}_lr{lr}_seed{seed}*.txt'))
    plora = sorted(glob.glob(f'./experiment/{model}_{dataset}_lora+pissa_r{rank}_lr{lr}_seed{seed}*.txt'))
    lmclora = sorted(glob.glob(f'./experiment/{model}_{dataset}_loralmc_r{rank}_lr{lr}_seed{seed}*.txt'))
    #pslora = sorted(glob.glob(f'./experiment/{model}_{dataset}_slora+pissa_r{rank}_lr{lr}_seed{seed}*.txt'))
    pos = 1

print(f'\n{model}_{dataset}_r{rank}_lr{lr}')

lora_results = []
slora_results = []
plora_results = []
pslora_results = []
lmclora_results = []

try:
    print("\n###########LoRA##########\n")
    for i in range(len(lora)):
        l = open(lora[i], 'r')
        
        print(lora[i].split('_')[-pos])

        lora_result = round(float(l.readlines()[-pos2].split(' ')[-1]),3)
        
        lora_results.append(lora_result)

        print(f'LoRA: {lora_result}')

    print(f'AVG LoRA: {sum(lora_results)/ len(lora_results)}')
except:
    pass

# print("\n##########SLoRA##########\n")

# for i in range(len(slora)):
#     sl = open(slora[i], 'r')
    
#     print(slora[i].split('_')[-pos])

#     slora_result = round(float(sl.readlines()[-pos2].split(' ')[-1]),3)
    
#     slora_results.append(slora_result)

#     print(f'SLoRA: {slora_result}')

# print(f'AVG SLoRA: {sum(slora_results)/ len(slora_results)}')

try:
    print("\n##########LoRA+PiSSA##########\n")

    for i in range(len(plora)):
        sl = open(plora[i], 'r')
        
        print(plora[i].split('_')[-pos])

        slora_result = round(float(sl.readlines()[-pos2].split(' ')[-1]),3)
        
        plora_results.append(slora_result)

        print(f'LoRA+PiSSA: {slora_result}')

    print(f'AVG LoRA+PiSSA: {sum(plora_results)/ len(plora_results)}')
except:
    pass

# print("\n##########SLoRA+PiSSA##########\n")

# for i in range(len(pslora)):
#     sl = open(pslora[i], 'r')
    
#     print(pslora[i].split('_')[-pos])

#     slora_result = round(float(sl.readlines()[-pos2].split(' ')[-1]),3)
    
#     pslora_results.append(slora_result)

#     print(f'SLoRA+PiSSA: {slora_result}')

# print(f'AVG SLoRA+PiSSA: {sum(pslora_results)/ len(pslora_results)}')

try:
    print("\n##########LoRA LMC##########\n")

    for i in range(len(lmclora)):
        sl = open(lmclora[i], 'r')
        
        print(lmclora[i].split('_')[-pos])

        slora_result = round(float(sl.readlines()[-pos2].split(' ')[-1]),3)
        
        lmclora_results.append(slora_result)

        print(f'LoRA LMC: {slora_result}')

    print(f'AVG LoRA LMC: {sum(lmclora_results)/ len(lmclora_results)}')
except:
    pass
