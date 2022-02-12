""" 
--------------------- RESNET FILE ---------------------
Author: Reza Tanakizadeh
Year  : 2022
P_name: Sima face verification project
-------------------------------------------------------
"""


# --- Import pkgs file and other classes
import import_pkgs
from resnet import ResNet18, ResNet34
from loss import TripletLoss
from utils import make_triplets, TripletFaceDatset, save_model

# --- Traning parameters
DatasetFolder   = "./CASIA-WebFace/"        # path to Dataset folder
ResNet_sel      = "18"                      # select ResNet type
NumberID        = 10575                     # Number of ID in CASIA-WebFace dataset 
batch_size      = 256                       # size of batch size
Triplet_size    = 10000 * batch_size        # size of total Triplets
loss_margin     = 0.6                       # Margin for Triplet loss
learning_rate   = 0.075                     # choose Learning Rate(note that this value will be change during training)
epochs          = 200                       # number of iteration over total dataset
load_last_epoch = False                     # If you need to continiue pre. model make this True
lfw_pairs_path  ='lfw_pairs_path.npy'       # LFW Pairs that we make it in LFW folder (.npy file)
device          = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_PATH = "./gdrive/MyDrive/best_trained.pth"
LAST_MODEL_PATH = "./gdrive/MyDrive/last_trained.pth"



# --- Instance of model
if ResNet_sel=="18": 
    model = ResNet18() 
else: 
    model = ResNet34()

# --- Model Summary
model.to(device)
summary(model, (3, 114, 114))
del model

# --- Make Triplet pairs and save then in a folder for future use
TripletList = make_triplets(Triplet_size, NumberID, DatasetFolder)

# --- Define Tansforms
transform_list =transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize((140,140)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std =[0.229, 0.224, 0.225])
                 ])

# --- Test Dataset
triplet_dataset = TripletFaceDatset(TripletList, transform_list, DatasetFolder)
print("Shape of images", triplet_dataset[0]['anc_img'].shape)

# --- LFW validation needed
l2_dist = PairwiseDistance(2)
valid_thresh = 0.96


# Make Trianing Faster in Pytorch(Cuda):
# 1. use number of worker
# 2. set pin_memory
# 3. Enable cuDNN for optimizing Conv
# 4. using AMP
# 5. set bias=False in conv layer if you set batch normalizing in model
# source: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b

# --- DataLoader
face_data = torch.utils.data.DataLoader(triplet_dataset, 
                                       batch_size= batch_size,
                                       shuffle=True,
                                       num_workers=2,
                                       pin_memory= True)

# --- Enable cuDNN
torch.backends.cudnn.benchmark = True

# --- Make model, optimizer and other variable that need during training loop
model       = ResNet18().to(device)                                                 # load model                                                       
tiplet_loss = TripletLoss(loss_margin)                                              # load Tripletloss
optimizer   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=learning_rate)                                          # load optimizer
epoch_check = 0
valid_arr   = []
loss_arr    = []


# --- Load prev. model if you need
if (load_last_epoch == True):
  # --- load last model
  # define model objects before this
  checkpoint  = torch.load(LAST_MODEL_PATH, map_location=device)   # load model path
  epoch_check = checkpoint['epoch']                                # load epoch
  loss        = checkpoint['loss']                                 # load loss value
  valid_arr   = checkpoint['accu_sv_list']                         # load Acc. values
  loss_arr    = checkpoint['loss_sv_list']                         # load loss values
  model.load_state_dict(checkpoint['model_state_dict'])            # load state dict
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    # load optimizer



# ------------------------------------------------------------------------------------------------
# ---------------------------------------  Training  ---------------------------------------------
# ------------------------------------------------------------------------------------------------
model.train()

temp = 0.075
for epoch in range(epoch_check,200):
  print(80*'=')
  
  # --- For saving imformation
  triplet_loss_sum = 0.0
  len_face_data = len(face_data)

  # -- set starting time
  time0 = time.time()

  # --- make learning rate update
  if 50 < len(loss_arr):
    for g in optimizer.param_groups:
      g['lr'] = 0.001
      temp = 0.001

  # --- loop on batches
  for batch_idx, batch_faces in enumerate(face_data):
    # --- Extract face triplets and send them to CPU or GPU
    anc_img = batch_faces['anc_img'].to(device)
    pos_img = batch_faces['pos_img'].to(device)
    neg_img = batch_faces['neg_img'].to(device)

    # --- Get embedded values for each triplet
    anc_embed = model(anc_img)
    pos_embed = model(pos_img)
    neg_embed = model(neg_img)
    

    # --- Find Distance
    pos_dist = l2_dist.forward(anc_embed, pos_embed)
    neg_dist = l2_dist.forward(anc_embed, neg_embed)

    # --- Select hard triplets
    all = (neg_dist - pos_dist < 0.8).cpu().numpy().flatten()
    hard_triplets = np.where(all == 1)

    if len(hard_triplets[0]) == 0: # --- Check number of hard triplets
      continue
    
    # --- select hard embeds
    anc_hard_embed = anc_embed[hard_triplets]
    pos_hard_embed = pos_embed[hard_triplets]
    neg_hard_embed = neg_embed[hard_triplets]

    # --- Loss
    loss_value = tiplet_loss.forward(anc_hard_embed, pos_hard_embed, neg_hard_embed)

    # --- backward path
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    if (batch_idx % 200 == 0) : print("Epoch: [{}/{}]  ,Batch index: [{}/{}], Loss Value:[{:.8f}]".format(epoch+1, epochs, batch_idx+1, len_face_data,loss_value))
    # --- save information
        triplet_loss_sum += loss_value.item()
  
  print("Learning Rate: ", temp)


  # --- Find Avg. loss value
  avg_triplet_loss = triplet_loss_sum / len_face_data
  loss_arr.append(avg_triplet_loss)

  
  # --- Validation part besed on LFW Dataset
  validation_acc = lfw_validation(model, l2_dist, valid_thresh)
  valid_arr.append(validation_acc)
  model.train()

  # --- Save model with checkpoints
  save_model(model, avg_triplet_loss, epoch+1, optimizer, validation_acc, valid_arr, loss_arr)

  # --- Print information for each epoch
  print(" Train set - Triplet Loss    =  {:.8f}".format(avg_triplet_loss))
  print(' Train set - Accuracy        = {:.8f}'.format(validation_acc))
  print(f' Execution time              = {time.time() - time0}')