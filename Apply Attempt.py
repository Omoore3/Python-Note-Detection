import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset, DataLoader
import os


from tqdm import tqdm


#Create dataframe based off of our notes.csv
df = pd.read_csv(r"E:\bunchonotes\notes.csv")
print(df.head())


#Changes notes to be numbers, not just strings of numbers
notes = list(df['note'].unique())
df['note'] = df['note'].apply(lambda x: notes.index(x % 12))
print(df.tail())


#Train test split, because I only have one data set
train_df, test_df = train_test_split(df, train_size = 0.8, random_state = 1) #random state can be whatever you want, just a seed for the splitting randomness

max_length = 220500 # max length of 5 seconds at 44.1khz... will probably have to change this later

'''
#This is mostly just testing with one of the audio things at a reasonable pitch
audio_sample_path = train_data.iloc[300, 0] #grabs from row 60 (a c4 I think, or something around there), column 0 (the file path)
print(audio_sample_path)



max_length = 1716224 #I ran a bit of code that calculates the max length, and since the data doesn't change, I just got the max length and eliminated the code

waveform, sample_rate = torchaudio.load(audio_sample_path)

zeros = torch.zeros(max_length - len(waveform[0]))
waveform = torch.cat((waveform[0], zeros))
print(waveform.shape)



mel_spec = torchaudio.transforms.MelSpectrogram(normalized = True)
spectrogram = mel_spec(waveform)
cmap = cm.cividis #You can choose any color map you prefer
rgb_image = cmap(spectrogram) # Convert single channel spectrogram to RGB using color map

# display rgb image
plt.imshow(rgb_image)
plt.axis("off")
plt.show()

resize = transforms.Resize((224, 224))
print(f"rgb_shape: {rgb_image.shape}")

rgb_image = rgb_image[:, :, :3]
rgb_tensor = torch.from_numpy(rgb_image)
print(f"original_shape: {rgb_tensor.shape}")
rgb_tensor = rgb_tensor.permute(2, 0, 1)
rgb_tensor = rgb_tensor.unsqueeze(0)
print(f"tensor shape: {rgb_tensor.shape}")
resized_rgb_image = resize(torch.tensor(rgb_tensor))
resized_rgb_image = resized_rgb_image.squeeze(0)
print(f"Before permutate: {resized_rgb_image.shape}")
resized_rgb_image = resized_rgb_image.permute(1, 2, 0)
print(f"final shape: {resized_rgb_image}")

plt.imshow(resized_rgb_image)
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 4))
plt.imshow(spectrogram, cmap="copper", origin="lower", aspect="auto")
plt.colorbar(format="%+2, 0f, dB")
plt.title("Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()
'''

class noteDataset(Dataset):
	def __init__(self, df, transform=None):
		self.df = df

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		audio_sample_path = self.df.iloc[index, 0]
		waveform, sample_rate = torchaudio.load(audio_sample_path)
		waveform = waveform.squeeze(0)

		#padding
		if len(waveform) < max_length:
			zeros = torch.zeros(max_length - len(waveform))
			waveform = torch.cat((waveform, zeros))
		else:
			waveform = waveform[:max_length]

		
		#Apply transforms into mel_spectrogram
		mel_spec = torchaudio.transforms.MelSpectrogram(normalized=True)
		resize = torchvision.transforms.Resize((224, 224))
		spectrogram = mel_spec(waveform)


		cmap = cm.viridis
		rgb_image = cmap(spectrogram)

		rgb_image = rgb_image[:, :, :3]
		rgb_tensor = torch.from_numpy(rgb_image).float()
		rgb_tensor = rgb_tensor.permute(2, 0, 1)


		inputs = resize(rgb_tensor)
		targets = self.df.iloc[index, 1]
		return inputs, targets

'''
transform = transforms.Compose([torchaudio.transforms.MelSpectrogram(normalized=True),
								torchvision.transforms.Resize((224, 224)),
								torchvision.transforms.ToTensor()])
'''

train_data = noteDataset(train_df)
test_data = noteDataset(test_df)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

num_classes = 127 #one for each note

model = torchvision.models.vgg19(pretrained=True)

model.classifier[6] = nn.Linear(4096, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
num_epochs = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
	# Training phase
	model.train()
	total_train_loss = 0

	for inputs, labels in tqdm(train_loader):
		inputs, labels = inputs.to(device), labels.to(device)

		optimizer.zero_grad() #zero the gradients
		outputs = model(inputs) #Forward passs
		loss = criterion(outputs, labels) # Calculate the loss
		loss.backward() # Backpropagation
		optimizer.step() # Update Weights
		total_train_loss += loss.item()

	avg_train_loss = total_train_loss / len(train_loader)

	#testing phase
	model.eval()
	total_test_loss = 0
	with torch.no_grad():
		for inputs, labels in test_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			total_test_loss += loss.item()

	avg_test_loss = total_test_loss / len(test_loader)
	print(f"Epoch: {epoch+1} / {num_epochs}, Train loss: {avg_train_loss:.4f}, Test accuracy: {avg_test_loss:.4f}")

torch.save(model.state_dict(), "E:/Note Model.pt")
print("Model Saved - Success !!")