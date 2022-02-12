
from torch.utils.data import Dataset

def make_triplets(Triplet_size, NumberID, DatasetFolder):
    # --- make a list of ids and folders
    selected_ids = np.uint32(np.round((np.random.rand(int(Triplet_size))) * (NumberID-1)))
    folders = os.listdir(DatasetFolder)

    # --- Itrate on each id and make Triplets list
    TripletList = []

    for index,id in enumerate(selected_ids):

        # --- find name of id faces folder
        id_str = str(folders[id])

        # --- find list of faces in this folder
        number_faces = os.listdir(DatasetFolder+id_str)

        # --- Get two Random number for Anchor and Positive
        while(True):
            two_random = np.uint32(np.round(np.random.rand(2) * (len(number_faces)-1))) 
            if (two_random[0] != two_random[1]):
                break

        # --- Make Anchor and Positive image
        Anchor   = str(number_faces[two_random[0]])
        Positive = str(number_faces[two_random[1]])

        # --- Make Negative image
        while(True):
            neg_id = np.uint32(np.round(np.random.rand(1) * (NumberID-1)))
            if (neg_id != id):
                break
        # --- number of images in negative Folder
        neg_id_str   = str(folders[neg_id[0]])
        number_faces = os.listdir(DatasetFolder+neg_id_str)
        one_random   = np.uint32(np.round(np.random.rand(1) * (len(number_faces)-1))) 
        Negative     = str(number_faces[one_random[0]])
        
        # --- insert Anchor, Positive and Negative image path to TripletList
        TempList = ["","",""]
        TempList[0] =  id_str + "/" + Anchor
        TempList[1] =  id_str + "/" + Positive
        TempList[2] =  neg_id_str + "/" + Negative
        TripletList.append(TempList)

    return TripletList



# --- Make Pytorch Dataset Class for Triplets 
class TripletFaceDatset(Dataset):
  def __init__(self, list_of_triplets, transform=None, DatasetFolder='./CASIA-WebFace/'):
    # --- initializing values
    print("Start Creating Triplets Dataset from CASIA-WebFace")
    self.list_of_triplets = list_of_triplets
    self.transform = transform
    self.DatasetFolder = DatasetFolder

  # --- getitem function
  def __getitem__(self, index):
    # --- get images path and read faces
    anc_img_path, pos_img_path, neg_img_path = self.list_of_triplets[index]
    anc_img = cv2.imread(self.DatasetFolder+anc_img_path)
    pos_img = cv2.imread(self.DatasetFolder+pos_img_path)
    neg_img = cv2.imread(self.DatasetFolder+neg_img_path)


    # --- set transform
    if self.transform:
      anc_img = self.transform(anc_img)
      pos_img = self.transform(pos_img)
      neg_img = self.transform(neg_img)
    
    return {'anc_img' : anc_img,
            'pos_img' : pos_img,
            'neg_img' : neg_img}
  
  # --- return len of triplets
  def __len__(self):
    return len(self.list_of_triplets)


def save_model(model_sv, loss_sv, epoch_sv, optimizer_state_sv, accuracy, accu_sv_list, loss_sv_list):
  # --- Inputs:
  #     1. model_sv           : Orginal model that trained
  #     2. loss_sv            : Current loss
  #     3. epoch_sv           : Current epoch
  #     4. optimizer_state_sv : Current value of optimizer 
  #     5. accuracy           : Current accuracy
  #     6. accu_sv_list       : Accuracy list
  #     7. loss_sv_list       : Loss list
  global BEST_MODEL_PATH
  global LAST_MODEL_PATH
  
  # --- save last epoch
  if accuracy >= max(accu_sv_list):
      torch.save(model.state_dict(), BEST_MODEL_PATH)
  
  # --- save this model for checkpoint
  torch.save({
            'epoch': epoch_sv,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_state_sv.state_dict(),
            'loss': loss_sv,
            'accu_sv_list': accu_sv_list,
            'loss_sv_list' : loss_sv_list
             }, LAST_MODEL_PATH)