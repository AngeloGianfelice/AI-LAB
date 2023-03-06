# Import dependencies
import torch 
import matplotlib.pyplot as plt
from torch import nn, save, load
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import torchvision.models as models
import numpy as np
import argparse

#hyperparameters
train_dir='D://Github//AI-LAB//data//processed_data//train' 
val_dir='D://Github//AI-LAB//data//processed_data//val'
test_dir='D://Github//AI-LAB//data//processed_data//test'
epochs=25
learning_rate=0.001
image_size=224
batch_size=50
n_features=25088
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = transforms.Compose([
    transforms.Resize((image_size,image_size)),  # scale imported image
    transforms.RandomHorizontalFlip(0.5), #data augmentation
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(np.random.randint(0,360)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])  # transform it into a torch tensor

test_loader = transforms.Compose([
    transforms.Resize((image_size,image_size)),  # scale imported image
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])  # transform it into a torch tensor

#Grapes detection Neural network
class GrapesDetector(nn.Module):
    def __init__(self,n_feature):
        super(GrapesDetector, self).__init__()
        self.linear = nn.Linear(n_feature,1)
        self.in_features=n_feature
        
    def forward(self, x):
        x = x.view(-1, self.in_features)
        y = torch.sigmoid(self.linear(x))
        return y

#feature extractor
def features_extractor(cnn,input):
        out=cnn(input) #vgg19
        return out

#training function
def train_model(train_loader,val_loader,base,model,loss_fn,optimizer,batch_size,epochs):
    
    #initialize list for later plotting results
    train_losses=[]
    train_accs=[]
    val_losses=[]
    val_accs=[]

    #training loop 
    for epoch in range(epochs):

        #training phase
        model.train() 
        print(f"Epoch {epoch+1}")
        print("training phase...")

        for inputs,label in train_loader:
        
            #forward pass
            inputs,label=inputs.to(device),label.to(device)
            features=features_extractor(base,inputs)
            pred=model(features).view(batch_size) 
        

            pred=pred.to(device).to(torch.float)
            target=label.to(device).to(torch.float)
            loss=loss_fn(pred,target)
            train_losses.append(loss.item())
            
            #Calculating training accuracy
            pred=pred.detach().round()
            train_correct=0
            for t in range(batch_size):
                if (pred[t]==target[t]).item():
                    train_correct += 1
            train_accs.append(train_correct/batch_size)
            

            #apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Training Done!")
        
        #validation phase
        model.eval()
        print("validation phase...")
        
        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs=inputs.to(device)
                labels=labels.to(device).to(torch.float)
                out=features_extractor(base,inputs)
                pred = model(out).to(torch.float).view(batch_size)
                loss=loss_fn(pred, labels)
                val_losses.append(loss.item())

                #Calculating validation accuracy
                pred=pred.detach().round()
                val_correct=0
                for t in range(batch_size):
                    if (pred[t]==labels[t]).item():
                        val_correct +=1
                val_accs.append(val_correct/batch_size)
                
        print("Validation Done!")
        print("-------------------------------")
    
    #plotting loss and accuracy history
    plotter(train_losses,'train_loss','Model Training Loss History','Batch Iterations','Loss','train_loss.png')
    plotter(train_accs,'train_accuracy','Model Training Accuracy History','Batch Iterations','Accuracy(%)','train_acc.png')
    plotter(val_losses,'validation_loss','Model Validation Loss History','Batch Iterations','Loss','val_loss.png')
    plotter(val_accs,'validation_accuracy','Model Validation Accuracy History','Batch Iterations','Accuracy(%)','val_acc.png')

#testing function        
def test_model(test_loader,base,model,loss_fn,batch_size):
    model.eval()
    test_size = len(test_loader.dataset)
    test_loss=0
    test_correct=0
    random_number=np.random.randint(0,len(test_loader))

    with torch.no_grad():

        for i,(inputs, labels) in enumerate(test_loader):

            inputs=inputs.to(device)
            labels=labels.to(device).to(torch.float)
            out=features_extractor(base,inputs)
            pred = model(out).to(torch.float).view(batch_size)
            loss=loss_fn(pred, labels)
            loss=loss.detach()
            test_loss += loss

            #select radnom batch to visualize predictions
            if(i==random_number):
                show_random_prediction(inputs,pred,num=6)
            
            #checking testing accuracy
            pred=pred.detach().round()
            for t in range(batch_size):
                if (pred[t]==labels[t]).item():
                    test_correct +=1

    test_loss /= len(test_loader)
    test_correct /= test_size
    print(f"Avg loss: {test_loss} \nAccuracy: {test_correct*100}%")
    
#batch visualizer function
def show_random_prediction(input,pred,num):

    #show num random prediction
    images_counter=0
    fig=plt.figure(figsize=(8,8),facecolor=(0,0,0))
    pred=pred.cpu().numpy()
    for j in range(batch_size):
        images_counter += 1
        ax = plt.subplot(num//2, 2, images_counter)
        ax.axis('off')
        if pred[j] > 0.5:
            title='grapes'
            color='green' 
        else: 
            title='not_grapes'
            color='red' 
        ax.set_title(f'predicted: {title} | prob: {round(pred[j]*100,2)}%',color=color,fontsize=10)#TODO fix padding issue
        imshow(input.cpu().data[j])
        if images_counter == num:
            break 
    plt.ioff()
    plt.show()
   
#show images function
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)

#history data plotter
def plotter(data,data_label,title,xlabel,ylabel,fname):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.plot(data,label=data_label)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('D://Github//AI-LAB//data//plotted_results//'+fname)
    plt.show()

#arg parser function
def arg_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("choice",type=str,help="Operazione consentite: train | test",choices=['train','test'])
    arg=parser.parse_args()
    return arg.choice

#main function
def main(arg):

    if torch.cuda.is_available() == False:
        print("Sorry, cuda not available")
    else:
        print("Cuda available on: ",torch.cuda.get_device_name())
    
    #importing pretrained VGG-19 network
    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    #freezing vgg19 weights
    for param in cnn.parameters():
        param.requires_grad = False

    #defining loss function (Binary cross Entropy Loss)
    loss_fn=nn.BCELoss()

    if arg == 'test':
        #loading test data
        test_dataset = datasets.ImageFolder(test_dir, transform=test_loader)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #testing phase
        #loading model
        loaded_model=GrapesDetector(n_features).to(device)
        loaded_model.load_state_dict(load('model_state.pt'))
    
        #testing model
        print("testing model...")
        test_model(test_dataloader, cnn, loaded_model, loss_fn, batch_size)
        print("Testing Done!")

    else:

        #loading train data
        train_dataset = datasets.ImageFolder(train_dir,transform=train_loader)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #loading validation data
        val_dataset = datasets.ImageFolder(val_dir, transform=train_loader)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #initializing neural network
        grapes_detector=GrapesDetector(n_features).to(device)

        #defining optimizer
        optimizer= torch.optim.SGD(grapes_detector.parameters(),lr=learning_rate)

        #training model (with validation)
        train_model(train_dataloader, val_dataloader, cnn, grapes_detector, loss_fn, optimizer, batch_size, epochs)   
        print("Training Done!")

        #saving trained model as 'model_state.pt'
        print('saving trained model') 
        with open('model_state.pt', 'wb') as fsave: 
            save(grapes_detector.state_dict(), fsave)
        print('model saved succesfully')

if __name__=="__main__":
    arg=arg_parser()
    main(arg)
    