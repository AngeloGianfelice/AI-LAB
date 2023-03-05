# Grapes Detection using VGG-19 Convolutional Neural Network
I used the VGG-19 Convolutional Neural Network to detect images containg grapes (in all sort of shapes and colors). 
## Description  
This is my first Machine Learning project: it was a lot of fun and i learned a lot about CNNs and its usage in image detection/classification
The idea was to view the problem of grapes detection (or even the more general image detection) into a binary classification problem and use the VGG-19 as a simple feature extractor to obtain the 'style' of grapes to then use as input of classification layer which give, as final output, the probability of an image containing grapes. This Approach is often described as [transfer learning](https://cs231n.github.io/transfer-learning/) and in particular in my case I'm using the 'ConvNet as a fixed feature extractor' method of transfer learning, where the base CNN is used with pretrained weights (and all its parameters frozen) and I'm only training the classification layer (in my case I've chosen a simple logistic regression but a two layer classification neural net would work fine as well.
Note: the fully connected classification layers that VGG19 comes with are removed and replaced with logistic regression.
### Model Inputs
Model inputs always have to be 4D array consisting of (batch_size,height,weight,depth) like it's shown in the image below:

![VGG19_input](https://user-images.githubusercontent.com/83078138/222975487-b9b99032-7681-4f64-8b4b-9ccb55fff636.jpg)

In my case the input images are RGB images of size (224,224,3) with a batches of 50 images
For my model I have used in total 4000 images (both images contaning grapes and not) splitted into 75% training samples, and 12,5% for testing and validation samples(each) like shown in the picture below:

![meta-chart](https://user-images.githubusercontent.com/83078138/222976931-e517e9b2-2be8-421b-aeb4-132e2b7efe3a.png)

### Model Architecture
VGG-19 contain a combination of layers which transform an image into output that the model can understand.

![vgg19-1024x173](https://user-images.githubusercontent.com/83078138/222979190-3bb1a4d2-9cf8-4b68-bf79-e5092a60d78a.png)

***-Convolutional layer***: creates a feature map by applying a filter that scans the image several pixels at a time

***-Pooling layer***: scales down the information generated by the convolutional layer to effectively store it

The final classification layer consists of a simple [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) using the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)(see image below) to calculate the probability that the given image (or more precisely, its features) contains grapes:

![1280px-Logistic-curve svg](https://user-images.githubusercontent.com/83078138/222980105-91764ef2-95f8-4765-b116-ac3d594d7bd0.png)

### Code
Raw images are first preprocessed in 'image_preprocessing.py' to assure they are all RGB format (3 channels) and splitted (via the [Split-Folders library](https://pypi.org/project/split-folders/) into the folder structure shown below:

![folder structure](https://user-images.githubusercontent.com/83078138/222977832-9fc3f9e0-a5f2-4cb6-9377-04a484178999.PNG)

          
where Class0 folders contains images without grapes, while Class1 contains images with grapes.
Here's the code for image processing: 
```python
def Convert_to_RGB(path):
    for i,image in enumerate(os.listdir(path)):
        im = Image.open(path+image)
        if im.mode != 'RGB':
            print("image converted: ",image)
            im.convert("RGB").save(path + f"RGB_image{i}.jpg")
            os.remove(path+image)

if __name__ =='__main__':
    #Convert all images into RGB form
    Convert_to_RGB(raw_data_path + 'Class0//')
    Convert_to_RGB(raw_data_path + 'Class1//')
    #perform train,validation,test split
    splitfolders.fixed(input=raw_data_path,output=processed_data_path,fixed=(1500,250,250))
```

The file GrapesDetector.py contains the model definition,initialization,training,validation,testing and results visualization.
Firstly we initialize our hyperparameters:
```python
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
```
where n_feature is the number of features extracted from the input batch of images by the VGG-19 neural net(exactly 7*7*512=25088) by the function feature_extractor.
```python
#feature extractor
def features_extractor(cnn,input):
        out=cnn(input) #vgg19
        return out
```
Note that the training data is randomly flipped and rotated just the help avoid overfitting (with a bit of data augmentation)
and all images are first normalized before passing it through the CNN.
Then we define our module:
```python
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
```
Here we have simple Neural Net with only one layer which takes as input the features extracted by the CNN and generate, through the sigmoid function (see above), 
a probability(before 0 and 1) that the input features contains grapes.
After that we have the classic training and testing function:
```python
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
```
Here we can see the classic training (and validation pipeline):
Firstly we use the built-in pytorch dataloaders to iterate through the input images in batches.
The model trains throughout many epochs (25 in our case are more than sufficient) by taking one forward and one backward pass of all training samples each time
Forward propagation calculates the loss and cost functions by comparing the difference between the actual and predicted target for each labeled image
Backward propagation uses [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to update the weights and bias for each neuron, attributing more impact on the neurons which have the most predictive power, until it arrives to an optimal activation combination (**global minimum**)(see image below).

![gradient descent](https://user-images.githubusercontent.com/83078138/222984187-2655f905-ce66-4a89-a3a5-5cb3f4b41b58.jpg)

As the model sees more examples, it learns to better predict the target causing the loss measure to decrease
The cost function takes the average loss across all samples indicating overall performance. After every training epoch the model is tested with the validation dataset 
to check model performance and improvements through the epochs (this process is called validation).
Finally after the training phase has finished we plot our results (in particular loss and accuracy history of our model).  

Then we have the testing function, where we test our already trained model with the test dataset:
```python
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
```
In my testing the model has reached an accuracy above 97% an the test dataset (500 images) 
The remaining function are all helper function for plotting and visualizing results and images
### Results
    todo



