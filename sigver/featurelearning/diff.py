import sys
import torch
import sigver.featurelearning.models as models

file1 = sys.argv[1]
file2 = sys.argv[2]

assert len(sys.argv) == 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model1 = models.available_models['signet']().to(device)
base_model2 = models.available_models['signet']().to(device)

parameters1,_,_ = torch.load(file1)
parameters2,_,_ = torch.load(file2)

base_model1.load_state_dict(parameters1)
base_model2.load_state_dict(parameters2)


diff = sum((x - y).abs().sum() for x, y in zip(base_model1.state_dict().values(), base_model2.state_dict().values()))
diff = diff.cpu().numpy()
print('Sum difference between weights:',diff)
if(diff==0):
	print('Base models are the same.')
else:
	print('Base models are different.')


