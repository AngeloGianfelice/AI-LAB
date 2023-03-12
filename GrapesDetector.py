# Import dependencies
import torch 
import matplotlib.pyplot as plt
from torch import nn, save, load
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import torchvision.models as models
import numpy as np
import argparse
import os
import time

#hyperparameters
train_dir='data//processed_data//train' 
val_dir='data//processed_data//val'
test_dir='data//processed_data//test'
epochs=25
learning_rate=0.001
image_size=224
batch_size=50
n_features=25088
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#defining loss function (Binary cross Entropy Loss) Note: loss_fn is the same in all cases (vgg/resnet/train/test) 
loss_fn=nn.BCELoss()

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

#feature extractor function
def features_extractor(cnn,input):
        out=cnn(input) #vgg19
        return out

#training function
def train_model(train_loader,val_loader,base,model,loss_fn,optimizer,batch_size,epochs):

    #initialize list for later plotting results
    history=[]
    train_losses=[]
    train_accs=[]
    val_losses=[]
    val_accs=[]
    train_samples=len(train_loader.dataset)
    train_batches=len(train_loader)
    validation_samples=len(val_loader.dataset)
    validation_baches=len(val_loader)

    #training loop 
    for epoch in range(epochs):

        #training phase
        model.train() 
        print(f"Epoch {epoch+1}")
        print("training phase...")
        t_loss=0
        t_correct=0
        for inputs,label in train_loader:
        
            #forward pass
            inputs,label=inputs.to(device),label.to(device)
            features=features_extractor(base,inputs)
            pred=model(features).view(batch_size) 
        

            pred=pred.to(device).to(torch.float)
            target=label.to(device).to(torch.float)
            loss=loss_fn(pred,target)
            t_loss+=loss.item()
            
            #Calculating training accuracy
            pred=pred.detach().round()
            train_correct=0
            for t in range(batch_size):
                if (pred[t]==target[t]).item():
                    t_correct += 1
            
            #apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        t_loss /= train_batches
        t_correct /= train_samples
        train_losses.append(t_loss)
        train_accs.append(t_correct*100)

        print("Training Done!")
        
        #validation phase
        model.eval()
        print("validation phase...")
        
        with torch.no_grad():
            v_loss=0
            v_correct=0
            for inputs, labels in val_loader:

                inputs=inputs.to(device)
                labels=labels.to(device).to(torch.float)
                out=features_extractor(base,inputs)
                pred = model(out).to(torch.float).view(batch_size)
                loss=loss_fn(pred, labels)
                v_loss += loss.item()

                #Calculating validation accuracy
                pred=pred.detach().round()
                for t in range(batch_size):
                    if (pred[t]==labels[t]).item():
                        v_correct +=1

            v_loss /= validation_baches
            v_correct /= validation_samples
            val_losses.append(v_loss)
            val_accs.append(v_correct*100) 

        print("Validation Done!")
        print("-------------------------------")
    
    history.append(train_losses)
    history.append(train_accs)
    history.append(val_losses)
    history.append(val_accs)
    return history
    
#testing function        
def test_model(test_loader,base,model,loss_fn,batch_size):
    model.eval()
    test_samples = len(test_loader.dataset)
    test_batches = len(test_loader)
    t_loss=0
    t_correct=0
    random_number=np.random.randint(0,len(test_loader))

    with torch.no_grad():

        for i,(inputs, labels) in enumerate(test_loader):

            inputs=inputs.to(device)
            labels=labels.to(device).to(torch.float)
            out=features_extractor(base,inputs)
            pred = model(out).to(torch.float).view(batch_size)
            loss=loss_fn(pred, labels)
            loss=loss.item()
            t_loss += loss

            #select random batch to visualize predictions
            #if(i==random_number):
                #show_random_prediction(inputs,pred,num=6)
            
            #checking testing accuracy
            pred=pred.detach().round()
            for t in range(batch_size):
                if (pred[t]==labels[t]).item():
                    t_correct +=1

        t_loss /= test_batches
        t_correct /= test_samples
        t_loss=np.round(t_loss,decimals=3)
        t_correct=np.round(t_correct*100,decimals=2)

    return t_loss,t_correct

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

#plotter helper function
def history_plotter(vgg_history,resnet_history,vgg_time,res_time):

    #plotting training loss
    plt.figure(figsize=(10,5))
    plt.title('Train Loss History')
    plt.plot(vgg_history[0],label='train_loss_vgg')
    plt.plot(resnet_history[0],label='train_loss_resnet')    
    plt.xlabel('epochs')  
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('data//plotted_results//'+'train_loss_comp')
    plt.show()

    #plotting training accuracy
    plt.figure(figsize=(10,5))
    plt.title('Train Accuracy History')
    plt.plot(vgg_history[1],label='train_acc_vgg')
    plt.plot(resnet_history[1],label='train_acc_resnet')    
    plt.xlabel('epochs')  
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.savefig('data//plotted_results//'+'train_acc_comp')
    plt.show()

    #plotting validation loss
    plt.figure(figsize=(10,5))
    plt.title('Validation Loss History')
    plt.plot(vgg_history[2],label='val_loss_vgg')
    plt.plot(resnet_history[2],label='val_loss_resnet')    
    plt.xlabel('epochs')  
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('data//plotted_results//'+'val_loss_comp')
    plt.show()

    #plotting training loss
    plt.figure(figsize=(10,5))
    plt.title('Validation Accuracy History')
    plt.plot(vgg_history[3],label='val_acc_vgg')
    plt.plot(resnet_history[3],label='val_acc_resnet')    
    plt.xlabel('epochs')  
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('data//plotted_results//'+'val_acc')
    plt.show()

    #plotting training time
    categories = ["VGG19", "RESNET18"]
    fig=plt.figure(figsize=(10,5))
    plt.subplot(1,1,1)
    plt.title('training time')
    xs = np.arange(1, len(categories) + 1)
    plt.bar(xs, [vgg_time,res_time], width=1, color=['green','red'], edgecolor='black')
    plt.text(0.75,vgg_time,str(vgg_time),fontweight='bold')
    plt.text(1.75,res_time,str(res_time),fontweight='bold')
    plt.ylabel('time(minutes)',labelpad=0)
    plt.xticks(xs, categories)
    plt.xlim(0, len(categories) + 1)
    plt.suptitle('VGG19 vs RESNET18 (Training)')
    fig.savefig('data//plotted_results//'+'training_time')
    plt.show()

#arg parser function
def arg_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",type=str,help="Operazione consentite: train | test",choices=['train','test'])
    arg=parser.parse_args()
    return arg.mode

#main function
def main(mode):

    if torch.cuda.is_available() == False:
        print("Sorry, cuda not available")
    else:
        print("Cuda available on: ",torch.cuda.get_device_name())

    #importing pretrained VGG-19 network
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval() #Note: IMAGENET1k_V1 weights is the same as pretrained=True
    vgg_fname='grapes_detector_vgg.pt'

    #importing pretrained resnet18 network
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    #strip resnet of its classification layer (as we do the classification ourselves XD)
    modules=list(resnet18.children())[:-2]
    resnet=nn.Sequential(*modules).to(device).eval()
    resnet_fname='grapes_detector_resnet.pt'

    #freezing vgg weights
    for param in vgg.parameters():
        param.requires_grad = False
    
    #freezing resnet weights
    for param in resnet.parameters():
        param.requires_grad = False

    if mode == 'test':

        #loading test data
        test_dataset = datasets.ImageFolder(test_dir, transform=test_loader)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #testing phase
        #loading models
        loaded_model_vgg=GrapesDetector(n_features).to(device)
        loaded_model_resnet=GrapesDetector(n_features).to(device)
        loaded_model_vgg.load_state_dict(load(vgg_fname))
        loaded_model_resnet.load_state_dict(load(resnet_fname))
    
        #testing models
        print("testing model...")
        start=time.time()
        vgg_loss,vgg_acc=test_model(test_dataloader, vgg, loaded_model_vgg, loss_fn, batch_size)
        end=time.time()
        vgg_time=np.round(end-start,decimals=2)
        start=time.time()
        res_loss,res_acc=test_model(test_dataloader, resnet, loaded_model_resnet, loss_fn, batch_size)
        end=time.time()
        res_time=np.round(end-start,decimals=2)

        #plotting testing history
        categories = ["VGG19", "RESNET18"]
        fig=plt.figure(figsize=(10,5))
        #loss graph
        plt.subplot(1,3,1)
        plt.title('average test loss')
        xs = np.arange(1, len(categories) + 1)
        plt.bar(xs, [vgg_loss,res_loss], width=1, color=['green','red'], edgecolor='black')
        plt.text(0.75,vgg_loss,str(vgg_loss),fontweight='bold')
        plt.text(1.75,res_loss,str(res_loss),fontweight='bold')
        plt.ylabel('loss',labelpad=0)
        plt.xticks(xs, categories)
        plt.xlim(0, len(categories) + 1)

        #acc graph
        plt.subplot(1,3,2)
        plt.title('test accuracy')
        xs = np.arange(1, len(categories) + 1)
        plt.bar(xs, [vgg_acc,res_acc], width=1, color=['green','red'], edgecolor='black')
        plt.text(0.75,vgg_acc,str(vgg_acc),fontweight='bold')
        plt.text(1.75,res_acc,str(res_acc),fontweight='bold')
        plt.ylabel('accuracy(%)',labelpad=0)
        plt.xticks(xs, categories)
        plt.xlim(0, len(categories) + 1)

        #time graph
        plt.subplot(1,3,3)
        plt.title('test time')
        xs = np.arange(1, len(categories) + 1)
        plt.bar(xs, [vgg_time,res_time], width=1, color=['green','red'], edgecolor='black')
        plt.text(0.75,vgg_time,str(vgg_time),fontweight='bold')
        plt.text(1.75,res_time,str(res_time),fontweight='bold')
        plt.ylabel('time(seconds)',labelpad=0)
        plt.xticks(xs, categories)
        plt.xlim(0, len(categories) + 1)

        fig.savefig('data//plotted_results//vggVSresnet')
        plt.suptitle('VGG19 vs RESNET18 (Testing)')
        plt.show()
        print("Testing Done!")
    
    elif mode == 'train':

       #deleting existing vgg model
        if os.path.isfile('grapes_detector_vgg.pt'):
            os.remove('grapes_detector_vgg.pt')
        else:
            print("VGG Model is not present in the system.")
         #deleting existing resnet model
        if os.path.isfile('grapes_detector_resnet.pt'):
            os.remove('grapes_detector_resnet.pt')
        else:
            print("RESNET Model is not present in the system.")

        #loading train data
        train_dataset = datasets.ImageFolder(train_dir,transform=train_loader)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #loading validation data
        val_dataset = datasets.ImageFolder(val_dir, transform=train_loader)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #VGG Training
        #initializing neural network
        grapes_detector_vgg=GrapesDetector(n_features).to(device)

        #defining optimizer
        optimizer= torch.optim.SGD(grapes_detector_vgg.parameters(),lr=learning_rate)

        #training model (with validation)
        start=time.time()
        history_vgg=train_model(train_dataloader, val_dataloader, vgg, grapes_detector_vgg, loss_fn, optimizer, batch_size, epochs)   
        end=time.time()
        vgg_time=np.round((end-start)/60,decimals=2)
        print("VGG model Trained!")

        #saving trained model as 'model_state.pt'
        print('saving trained model') 
        with open(vgg_fname, 'wb') as fsave: 
            save(grapes_detector_vgg.state_dict(), fsave)
        print('model saved succesfully')

        #RESNET training
        #initializing neural network
        grapes_detector_resnet=GrapesDetector(n_features).to(device)

        #defining optimizer
        optimizer= torch.optim.SGD(grapes_detector_resnet.parameters(),lr=learning_rate)

        #training model (with validation)
        start=time.time()
        history_resnet=train_model(train_dataloader, val_dataloader, resnet, grapes_detector_resnet, loss_fn, optimizer, batch_size, epochs)
        end=time.time()
        res_time=np.round((end-start)/60,decimals=2)   
        print("RESNET model Trained!")
        print("Training Done!")

        #saving trained model as 'model_state.pt'
        print('saving trained model') 
        with open(resnet_fname, 'wb') as fsave: 
            save(grapes_detector_resnet.state_dict(), fsave)
        print('model saved succesfully')

        #plotter function
        history_plotter(history_vgg,history_resnet,vgg_time,res_time)


    else:
        raise TypeError("Unknown mode: use train or test")

if __name__=="__main__":
    mode=arg_parser()
    main(mode)
    