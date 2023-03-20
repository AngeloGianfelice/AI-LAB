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
batch_size=25
n_features=[102400,409600,1638400,6553600] #feature for each style layers
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
style_layers = ['conv1_1','conv2_1','conv3_1','conv5_1']
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

#Defining a class for the style features extractor model
class VGG(nn.Module):
    def __init__(self,layer):
        super(VGG,self).__init__()
        self.style_layers=['0','5','10','28']
        self.layer=self.style_layers[layer]
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:(int(self.layer)+1)] #model will contain only the necessary layers
        #freezing vgg weights
        for param in self.model.parameters():
            param.requires_grad = False

    #x holds the input tensor(image) that will be feeded to each layer
    def forward(self,x):
        #Iterate over all the layers of the mode
        for layer_num,layer in enumerate(self.model):
            #activation of the layer will stored in x
            x=layer(x)
            #appending the activation of the selected layers and return the gram matrix
            if str(layer_num) == self.layer:
                return gram_matrix(x)

#gram_matrix function
def gram_matrix(input):
    b,c,h,w=input.shape
    # b=batch size
    # c=number of feature maps
    # (h,w)=dimensions of a f. map (N=c*d)
    features=input.view(b*c,h*w)
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b*c*h*w)

#Grapes detection Neural network
class GrapesDetector(nn.Module):
    def __init__(self,n_feature):
        super(GrapesDetector, self).__init__()
        self.linear = nn.Linear(n_feature,1)
        self.in_features=n_feature
        
    def forward(self, x):
        x=x.view(-1,self.in_features) 
        y = torch.sigmoid(self.linear(x))
        return y

#training function
def train_model(train_loader,val_loader,style_extractor,model,loss_fn,optimizer,batch_size,epochs):

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
            inputs,label=inputs.to(device,torch.float),label.to(device,torch.float)
            style_features=style_extractor(inputs)
            pred=model(style_features).view(batch_size)
            
            loss=loss_fn(pred,label)
            t_loss+=loss.item()
            
            #Calculating training accuracy
            pred=pred.detach().round()
            for t in range(batch_size):
                if (pred[t]==label[t]).item():
                    t_correct += 1
            
            #apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        t_loss /= train_batches
        t_correct /= train_samples
        train_losses.append(t_loss)
        train_accs.append(t_correct*100)
        print("training loss: ",t_loss)
        print("training accuracy: ",t_correct*100)

        print("Training Done!")
        
        #validation phase
        model.eval()
        print("validation phase...")
        
        with torch.no_grad():
            v_loss=0
            v_correct=0
            for inputs, labels in val_loader:

                inputs=inputs.to(device,torch.float)
                labels=labels.to(device,torch.float)
                style_features=style_extractor(inputs)
                pred = model(style_features).view(batch_size)

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
            print("validation loss: ",v_loss)
            print("validation accuracy: ",v_correct*100)

        print("Validation Done!")
        print("-------------------------------")
    
    history.append(train_losses)
    history.append(train_accs)
    history.append(val_losses)
    history.append(val_accs)
    return history
    
#testing function        
def test_model(test_loader,style_extractor,model,loss_fn,batch_size):
    model.eval()
    test_samples = len(test_loader.dataset)
    test_batches = len(test_loader)
    t_loss=0
    t_correct=0
    random_number=np.random.randint(0,len(test_loader))

    with torch.no_grad():

        for i,(inputs, labels) in enumerate(test_loader):

            inputs=inputs.to(device,torch.float)
            labels=labels.to(device,torch.float)
            style_features=style_extractor(inputs)
            pred = model(style_features).view(batch_size)
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
def history_plotter(data,times):

    #plotting training loss
    plt.figure(figsize=(10,5))
    plt.title('Train Loss History')
    plt.plot(data[0][0],label='conv1_1')
    plt.plot(data[1][0],label='conv2_1')
    plt.plot(data[2][0],label='conv3_1') 
    plt.plot(data[3][0],label='conv5_1')      
    plt.xlabel('epochs')  
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('data//plotted_results//train//'+'train_loss')
    plt.show()

    #plotting training accuracy
    plt.figure(figsize=(10,5))
    plt.title('Train Accuracy History')
    plt.plot(data[0][1],label='conv1_1')
    plt.plot(data[1][1],label='conv2_1')
    plt.plot(data[2][1],label='conv3_1') 
    plt.plot(data[3][1],label='conv5_1')   
    plt.xlabel('epochs')  
    plt.ylabel('accuracy(%)')
    plt.legend()
    plt.savefig('data//plotted_results//train//'+'train_acc')
    plt.show()

    #plotting validation loss
    plt.figure(figsize=(10,5))
    plt.title('Validation Loss History')
    plt.plot(data[0][2],label='conv1_1')
    plt.plot(data[1][2],label='conv2_1')
    plt.plot(data[2][2],label='conv3_1') 
    plt.plot(data[3][2],label='conv5_1')    
    plt.xlabel('epochs')  
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('data//plotted_results//train//'+'val_loss')
    plt.show()

    #plotting validation accuracy
    plt.figure(figsize=(10,5))
    plt.title('Validation Accuracy History')
    plt.plot(data[0][3],label='conv1_1')
    plt.plot(data[1][3],label='conv2_1')
    plt.plot(data[2][3],label='conv3_1') 
    plt.plot(data[3][3],label='conv5_1')   
    plt.xlabel('epochs')  
    plt.ylabel('accuracy(%)')
    plt.legend(loc='best',fontsize='xx-small')
    plt.savefig('data//plotted_results//train//'+'val_acc')
    plt.show()

    #plotting training time
    fig=plt.figure(figsize=(10,5))
    plt.title('training time')
    plt.bar(style_layers,times, width=0.4,color='green', edgecolor='black')
    plt.text(-0.1,times[0],str(times[0]),fontweight='bold')
    plt.text(0.9,times[1],str(times[1]),fontweight='bold')
    plt.text(1.9,times[2],str(times[2]),fontweight='bold')
    plt.text(2.9,times[3],str(times[3]),fontweight='bold')
    plt.ylabel('time(minutes)',labelpad=0)
    plt.title('Training Time')
    fig.savefig('data//plotted_results//train//'+'training_time')
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

    if mode == 'test':

        #loading test data
        test_dataset = datasets.ImageFolder(test_dir, transform=test_loader)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #testing results lists
        losses=[]
        accs=[]
        times=[]
        for i,elem in enumerate(n_features):

            if len(os.listdir('models//')) < len(n_features): #some of the models are not present in the models folder 
                raise FileNotFoundError("models not found") 
            
            #loading vgg
            style_extractor=VGG(i).to(device).eval()

            #testing phase
            #loading model
            loaded_model_vgg=GrapesDetector(elem).to(device)
            loaded_model_vgg.load_state_dict(load('models//grapes_detector_vgg'+str(i)+'.pt'))
    
            #testing models
            print(f"testing model #{i}...")
            start=time.time()
            vgg_loss,vgg_acc=test_model(test_dataloader, style_extractor, loaded_model_vgg, loss_fn, batch_size)
            end=time.time()
            vgg_time=np.round(end-start,decimals=2)
            losses.append(vgg_loss)
            accs.append(vgg_acc)
            times.append(vgg_time)
            print(f"testing model #{i} finished!")


        #plotting testing history
        #loss graph
        fig=plt.figure(figsize=(10,5))
        plt.title('Test Loss')
        plt.bar(style_layers,losses, width=0.4,color='royalblue', edgecolor='black')
        plt.text(-0.1,losses[0],str(losses[0]),fontweight='bold')
        plt.text(0.9,losses[1],str(losses[1]),fontweight='bold')
        plt.text(1.9,losses[2],str(losses[2]),fontweight='bold')
        plt.text(2.9,losses[3],str(losses[3]),fontweight='bold')
        plt.ylabel('loss(average)')
        fig.savefig('data//plotted_results//test//'+'test_losses')
        plt.show()

        #acc graph
        fig=plt.figure(figsize=(10,5))
        plt.title('Test Accuracy')
        plt.bar(style_layers,accs, width=0.4,color='royalblue', edgecolor='black')
        plt.text(-0.1,accs[0],str(accs[0]),fontweight='bold')
        plt.text(0.9,accs[1],str(accs[1]),fontweight='bold')
        plt.text(1.9,accs[2],str(accs[2]),fontweight='bold')
        plt.text(2.9,accs[3],str(accs[3]),fontweight='bold')
        plt.ylabel('accuracy(%)')
        fig.savefig('data//plotted_results//test//'+'test_accs')
        plt.show()

        #time graph
        fig=plt.figure(figsize=(10,5))
        plt.title('Test Time')
        plt.bar(style_layers,times, width=0.4,color='royalblue', edgecolor='black')
        plt.text(-0.1,times[0],str(times[0]),fontweight='bold')
        plt.text(0.9,times[1],str(times[1]),fontweight='bold')
        plt.text(1.9,times[2],str(times[2]),fontweight='bold')
        plt.text(2.9,times[3],str(times[3]),fontweight='bold')
        plt.ylabel('time(seconds)')
        fig.savefig('data//plotted_results//test//'+'test_times')
        plt.show()

        print("Testing Done!")
    
    elif mode == 'train':

       #deleting existing vggs model
        for i in range(len(n_features)):
            if os.path.isfile('models//grapes_detector_vgg'+str(i)+'.pt'):
                os.remove('models//grapes_detector_vgg'+str(i)+'.pt')
                print(f"removed vgg model #{i}")
        
        #loading train data
        train_dataset = datasets.ImageFolder(train_dir,transform=train_loader)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        #loading validation data
        val_dataset = datasets.ImageFolder(val_dir, transform=train_loader)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

        data=[]
        times=[]
        for i,elem in enumerate(n_features):
            print(f'---------------------------------MODEL #{i}------------------------------')
            #VGG Training
            style_extractor=VGG(i).to(device).eval()
            #initializing neural network
            grapes_detector_vgg=GrapesDetector(elem).to(device)

            #defining optimizer
            optimizer= torch.optim.Adam(grapes_detector_vgg.parameters(),lr=learning_rate)

            #training model (with validation)
            start=time.time()
            history=train_model(train_dataloader, val_dataloader, style_extractor, grapes_detector_vgg, loss_fn, optimizer, batch_size, epochs)   
            end=time.time()
            train_time=np.round((end-start)/60,decimals=2)
            times.append(train_time)
            data.append(history)
            print(f"VGG model #{i} Trained!")

            #saving trained model as 'model_state.pt'
            print('saving trained model') 
            with open('models//grapes_detector_vgg'+str(i)+'.pt', 'wb') as fsave: 
                save(grapes_detector_vgg.state_dict(), fsave)
            print(f'model #{i} saved succesfully')

        history_plotter(data,times)

    else:
        raise TypeError("Unknown mode: use train or test")

if __name__=="__main__":
    mode=arg_parser()
    main(mode)
    